"""
Offline test-suite for the CS[02] codebase. No network or API key is needed:
LLM calls are replaced by fakes so each module's logic is checked in isolation.

Run:  python3 -m unittest eval/test_cv_sorting.py -v
"""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout, redirect_stderr
from types import SimpleNamespace
from unittest import mock

CODEBASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Capstone_Project-CS[02]", "Codebase")
)
sys.path.insert(0, CODEBASE)

import document_loader as dl  # noqa: E402
import extractor as ex  # noqa: E402
import llm_client as lc  # noqa: E402
import pipeline as pl  # noqa: E402
import ranker as rk  # noqa: E402


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


class DocumentLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_text_and_latin1_fallback(self):
        p = os.path.join(self.tmp, "a.txt")
        with open(p, "wb") as f:
            f.write("Jos\xe9 Python".encode("latin-1"))
        self.assertIn("Python", dl.load_document(p))

    def test_empty_file_raises(self):
        p = _write(os.path.join(self.tmp, "empty.txt"), "   \n")
        with self.assertRaises(RuntimeError):
            dl.load_document(p)

    def test_unsupported_extension_raises(self):
        p = _write(os.path.join(self.tmp, "x.csv"), "a,b")
        with self.assertRaises(RuntimeError):
            dl.load_document(p)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            dl.load_document(os.path.join(self.tmp, "nope.txt"))

    def test_corrupt_docx_raises(self):
        p = _write(os.path.join(self.tmp, "bad.docx"), "not a zip")
        with self.assertRaises(RuntimeError):
            dl.load_document(p)

    def test_docx_paragraphs(self):
        p = os.path.join(self.tmp, "c.docx")
        xml = (
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>Line one</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>Line </w:t></w:r><w:r><w:t>two</w:t></w:r></w:p></w:body></w:document>"
        )
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("word/document.xml", xml)
        self.assertEqual(dl.load_document(p), "Line one\nLine two")

    def test_directory_scan_ignores_hidden_and_unsupported(self):
        _write(os.path.join(self.tmp, "a.txt"), "cv")
        _write(os.path.join(self.tmp, ".DS_Store"), "x")
        _write(os.path.join(self.tmp, "n.csv"), "x")
        paths = dl.collect_cv_paths([], self.tmp)
        self.assertEqual([os.path.basename(p) for p in paths], ["a.txt"])

    def test_dedupe_and_jd_exclusion(self):
        a = _write(os.path.join(self.tmp, "a.txt"), "cv")
        jd = _write(os.path.join(self.tmp, "job.txt"), "jd")
        paths = dl.collect_cv_paths([a], self.tmp, exclude_paths=[jd])
        self.assertEqual(paths, [os.path.abspath(a)])

    def test_no_cvs_raises(self):
        with self.assertRaises(RuntimeError):
            dl.collect_cv_paths([], self.tmp)

    def test_missing_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            dl.collect_cv_paths([], os.path.join(self.tmp, "missing"))


class JsonParsingTests(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(lc.parse_json_object('{"a": 1}'), {"a": 1})

    def test_fenced_with_prose(self):
        self.assertEqual(lc.parse_json_object('Here:\n```json\n{"a": 1}\n```\nDone'), {"a": 1})

    def test_embedded(self):
        self.assertEqual(lc.parse_json_object('Sure! {"a": {"b": 2}} hope it helps'), {"a": {"b": 2}})

    def test_array_rejected(self):
        with self.assertRaises(ValueError):
            lc.parse_json_object("[1, 2]")

    def test_garbage_rejected(self):
        with self.assertRaises(ValueError):
            lc.parse_json_object("no json here")


def _fake_response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class LLMClientTests(unittest.TestCase):
    def _client(self, side_effect):
        c = lc.LLMClient(api_key="k", model="m", max_retries=3)
        c._client = mock.MagicMock()
        c._client.chat.completions.create.side_effect = side_effect
        return c

    def test_rejects_empty_key_and_model(self):
        with self.assertRaises(ValueError):
            lc.LLMClient(api_key=" ", model="m")
        with self.assertRaises(ValueError):
            lc.LLMClient(api_key="k", model="")

    def test_response_format_fallback(self):
        calls = []

        def side(**kw):
            calls.append("response_format" in kw)
            if "response_format" in kw:
                raise Exception("Unsupported parameter: response_format")
            return _fake_response('{"ok": true}')

        self.assertEqual(self._client(side).complete_json("s", "u"), {"ok": True})
        self.assertEqual(calls, [True, False])

    def test_repair_pass(self):
        replies = iter([_fake_response("not json"), _fake_response('{"fixed": 1}')])
        c = self._client(lambda **kw: next(replies))
        self.assertEqual(c.complete_json("s", "u"), {"fixed": 1})

    @mock.patch("llm_client.time.sleep", lambda s: None)
    def test_transient_error_retried(self):
        replies = iter([Exception("503"), _fake_response('{"a": 1}')])

        def side(**kw):
            r = next(replies)
            if isinstance(r, Exception):
                raise r
            return r

        self.assertEqual(self._client(side).complete_json("s", "u"), {"a": 1})

    @mock.patch("llm_client.time.sleep", lambda s: None)
    def test_gives_up_after_retries(self):
        def side(**kw):
            raise Exception("boom")

        with self.assertRaises(RuntimeError):
            self._client(side).complete("s", "u")

    def test_auth_error_not_retried(self):
        calls = []

        class AuthError(Exception):
            status_code = 401

        def side(**kw):
            calls.append(1)
            raise AuthError("invalid api key")

        with self.assertRaises(RuntimeError):
            self._client(side).complete("s", "u", json_mode=False)
        self.assertEqual(len(calls), 1)

    def test_key_resolution_order(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "GROQ_API_KEY": "g", "LLM_API_KEY": "l"}):
            self.assertEqual(lc.resolve_api_key(None), "g")
            self.assertEqual(lc.resolve_api_key(" cli "), "cli")


class ExtractorTests(unittest.TestCase):
    def test_normalize_handles_bad_types(self):
        job = ex.normalize_job({"must_have_skills": "Python", "required_experience_years": "5+ yrs"})
        self.assertEqual(job["must_have_skills"], ["Python"])
        self.assertEqual(job["job_title"], "Untitled role")
        cv = ex.normalize_cv({"experience": "oops", "years_experience": "7.5"}, "f.txt")
        self.assertEqual(cv["candidate_name"], "f.txt")
        self.assertEqual(cv["experience"], [])
        self.assertEqual(cv["years_experience"], 7)

    def test_overlap_basic(self):
        job = {"must_have_skills": ["Python", "SQL", "Spark"], "nice_to_have_skills": ["Kafka"]}
        cv = {"skills": ["python 3", "PySpark", "sql"]}
        ov = ex.overlap_evidence(job, cv)
        self.assertEqual(ov["must_have_missing"], [])
        self.assertEqual(ov["must_have_coverage_pct"], 100.0)
        self.assertEqual(ov["nice_to_have_missing"], ["Kafka"])

    def test_overlap_short_skill_no_false_positive(self):
        # A one-letter skill like "R" or "C" must not match "Spark" / "Docker".
        job = {"must_have_skills": ["Spark", "Docker", "Go"], "nice_to_have_skills": []}
        cv = {"skills": ["R", "C", "Django"]}
        ov = ex.overlap_evidence(job, cv)
        self.assertEqual(ov["must_have_matched"], [], ov)

    def test_vendor_prefix_and_compounds(self):
        self.assertTrue(ex.skills_match("Apache Spark", "PySpark"))
        self.assertTrue(ex.skills_match("Apache Airflow", "Airflow"))
        self.assertTrue(ex.skills_match("AWS", "AWS (S3, Glue, EMR)"))
        self.assertFalse(ex.skills_match("Java", "JavaScript"))
        self.assertFalse(ex.skills_match("C", "C++"))

    def test_overlap_no_must_haves(self):
        ov = ex.overlap_evidence({"must_have_skills": [], "nice_to_have_skills": []}, {"skills": []})
        self.assertEqual(ov["must_have_coverage_pct"], 0.0)


class RankerTests(unittest.TestCase):
    def test_clip_and_defaults(self):
        s = rk.normalize_score({"overall_score": 150, "skills_score": "-5", "experience_score": "n/a"})
        self.assertEqual((s["overall_score"], s["skills_score"], s["experience_score"]), (100, 0, 0))

    def test_sort_tie_breaks(self):
        rows = [
            {"candidate_name": "B", "overall_score": 80, "skills_score": 70, "missing_must_have": []},
            {"candidate_name": "A", "overall_score": 80, "skills_score": 70, "missing_must_have": ["x"]},
            {"candidate_name": "C", "overall_score": 80, "skills_score": 90, "missing_must_have": ["x"]},
        ]
        self.assertEqual([r["candidate_name"] for r in rk.sort_results(rows)], ["C", "B", "A"])

    def test_calibration_ignores_unknown_ids_and_missing_scores(self):
        results = [
            {"candidate_id": "cv_001", "candidate_name": "A", "overall_score": 70, "skills_score": 70},
            {"candidate_id": "cv_002", "candidate_name": "B", "overall_score": 60, "skills_score": 60},
        ]
        fake = mock.MagicMock()
        fake.complete_json.return_value = {
            "rankings": [
                {"candidate_id": "cv_001", "overall_score": 75, "tie_break_note": "n"},
                {"candidate_id": "cv_002"},
                {"candidate_id": "cv_999", "overall_score": 1},
            ]
        }
        out = rk.recalibrate_scores(fake, {"job_title": "t"}, results)
        self.assertEqual([r["overall_score"] for r in out], [75, 60])
        self.assertEqual(out[0]["pre_calibration_score"], 70)

    def test_calibration_skipped_for_single(self):
        fake = mock.MagicMock()
        rows = [{"candidate_id": "cv_001", "overall_score": 5}]
        self.assertIs(rk.recalibrate_scores(fake, {}, rows), rows)
        fake.complete_json.assert_not_called()


class FakeLLM:
    """Deterministic stand-in for both LLM stages, keyed on prompt content."""

    def __init__(self, model, fail_on=None):
        self.model = model
        self.fail_on = fail_on

    def complete_json(self, system, user):
        if self.fail_on and self.fail_on in user:
            raise RuntimeError("simulated LLM failure")
        if "JOB DESCRIPTION:" in user:
            return {"job_title": "DE", "must_have_skills": ["Python", "SQL"], "nice_to_have_skills": []}
        if "RESUME TEXT:" in user:
            good = "Anita" in user
            return {"candidate_name": "Anita" if good else "Ben", "skills": ["Python", "SQL"] if good else ["Java"]}
        if "independently" in user:
            return {"rankings": []}
        good = '"Anita"' in user
        return {"overall_score": 90 if good else 20, "skills_score": 90 if good else 10,
                "missing_must_have": [] if good else ["Python", "SQL"], "explanation": "x"}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.jd = _write(os.path.join(self.tmp, "job.txt"), "JD text")
        self.cvs = os.path.join(self.tmp, "cvs")
        os.makedirs(self.cvs)
        _write(os.path.join(self.cvs, "a.txt"), "Anita Python SQL")
        _write(os.path.join(self.cvs, "b.txt"), "Ben Java")

    def _run(self, **overrides):
        kw = dict(jd_path=self.jd, cv_paths=[], cv_dir=self.cvs, extractor=FakeLLM("e"),
                  ranker=FakeLLM("r"), required_skills=[], min_score=0, calibrate=True,
                  output_path=os.path.join(self.tmp, "out.json"))
        kw.update(overrides)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return pl.run_pipeline(**kw)

    def test_end_to_end_ranking_and_json(self):
        payload = self._run()
        self.assertEqual([c["candidate_name"] for c in payload["ranked_candidates"]], ["Anita", "Ben"])
        with open(os.path.join(self.tmp, "out.json")) as f:
            self.assertEqual(json.load(f)["candidate_count"], 2)

    def test_filters(self):
        self.assertEqual(self._run(min_score=50)["candidate_count"], 1)
        self.assertEqual(self._run(required_skills=["java"])["ranked_candidates"][0]["candidate_name"], "Ben")

    def test_one_unreadable_cv_does_not_abort_run(self):
        _write(os.path.join(self.cvs, "empty.txt"), "")
        payload = self._run()
        self.assertEqual(payload["candidate_count"], 2)
        self.assertEqual(len(payload["skipped"]), 1)

    def test_one_failed_llm_call_does_not_abort_run(self):
        payload = self._run(ranker=FakeLLM("r", fail_on='"Ben"'))
        self.assertEqual([c["candidate_name"] for c in payload["ranked_candidates"]], ["Anita"])
        self.assertEqual(len(payload["skipped"]), 1)


class CliTests(unittest.TestCase):
    def _cli(self, *args, env=None):
        e = dict(os.environ, OPENAI_API_KEY="", GROQ_API_KEY="", LLM_API_KEY="")
        e.update(env or {})
        return subprocess.run([sys.executable, "main.py", *args], cwd=CODEBASE,
                              capture_output=True, text=True, env=e)

    def test_help(self):
        self.assertEqual(self._cli("-h").returncode, 0)

    def test_missing_cvs(self):
        self.assertNotEqual(self._cli("--jd", "x.txt").returncode, 0)

    def test_missing_key(self):
        p = self._cli("--jd", "x.txt", "--cvs", "a.txt")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("API key", p.stderr)

    def test_missing_jd_is_clean_error(self):
        p = self._cli("--jd", "nope.txt", "--cvs", "a.txt", "--api-key", "k")
        self.assertEqual(p.returncode, 1)
        self.assertNotIn("Traceback", p.stderr)
        self.assertIn("Document not found", p.stderr)

    def test_same_model_warning(self):
        import main as app
        args = app.build_parser().parse_args(
            ["--jd", "x", "--cvs", "a", "--extractor-model", "m", "--ranker-model", "m"])
        err = io.StringIO()
        with redirect_stderr(err):
            app.validate_args(args)
        self.assertIn("identical", err.getvalue())

    def test_bad_min_score(self):
        self.assertNotEqual(self._cli("--jd", "x", "--cvs", "a", "--min-score", "101").returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
