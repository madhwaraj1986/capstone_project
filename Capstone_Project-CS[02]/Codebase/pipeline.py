"""
End-to-end CV sorting pipeline: ingest, extract (LLM-1), score (LLM-2), rank.

The pipeline is intentionally linear so evaluators can map each step in the
report to a function in this module.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from document_loader import collect_cv_paths, load_document, load_named_documents
from extractor import extract_cv, extract_job
from llm_client import LLMClient
from ranker import recalibrate_scores, score_candidate, sort_results


def filter_by_required_skills(
    results: List[Dict[str, Any]], required_skills: Sequence[str]
) -> List[Dict[str, Any]]:
    """
    Recruiter-side filter: drop candidates missing any extra required skill.

    Matching is case-insensitive and uses substring comparison so that
    "SQL" matches "PostgreSQL" when that is the recruiter's intent.
    """
    if not required_skills:
        return results

    needles = [skill.strip().lower() for skill in required_skills if skill.strip()]
    if not needles:
        return results

    kept: List[Dict[str, Any]] = []
    for item in results:
        haystack = " ".join(item.get("skills") or []).lower()
        if all(needle in haystack for needle in needles):
            kept.append(item)
    return kept


def apply_min_score(results: List[Dict[str, Any]], min_score: int) -> List[Dict[str, Any]]:
    """Keep candidates whose overall_score is at least min_score."""
    return [item for item in results if int(item.get("overall_score") or 0) >= min_score]


def _fit(text: str, width: int) -> str:
    """Truncate text to width, marking the cut with '...' instead of chopping mid-word silently."""
    return text if len(text) <= width else text[: width - 3].rstrip(" ,") + "..."


def render_table(results: List[Dict[str, Any]]) -> str:
    """Render a fixed-width ranked table for terminal evaluation."""
    if not results:
        return "No candidates remained after ranking/filtering."

    headers = ("RANK", "SCORE", "CANDIDATE", "SOURCE", "MISSING MUST-HAVES")
    rows = [headers]
    for item in results:
        missing = ", ".join(item.get("missing_must_have") or []) or "-"
        rows.append(
            (
                str(item.get("rank")),
                str(item.get("overall_score")),
                _fit(str(item.get("candidate_name") or ""), 32),
                _fit(str(item.get("source_file") or ""), 24),
                _fit(missing, 48),
            )
        )

    widths = [max(len(row[col]) for row in rows) for col in range(len(headers))]
    lines = []
    for index, row in enumerate(rows):
        line = " | ".join(row[col].ljust(widths[col]) for col in range(len(headers)))
        lines.append(line)
        if index == 0:
            lines.append("-+-".join("-" * widths[col] for col in range(len(headers))))
    return "\n".join(lines)


def render_explanations(results: List[Dict[str, Any]]) -> str:
    """Print per-candidate evidence so ranking is inspectable, not a black box."""
    blocks: List[str] = []
    for item in results:
        strengths = "; ".join(item.get("strengths") or []) or "-"
        risks = "; ".join(item.get("risks") or []) or "-"
        blocks.append(
            "\n".join(
                [
                    f"[{item.get('rank')}] {item.get('candidate_name')} "
                    f"(overall={item.get('overall_score')}, "
                    f"skills={item.get('skills_score')}, "
                    f"experience={item.get('experience_score')}, "
                    f"domain={item.get('domain_score')}, "
                    f"education={item.get('education_score')})",
                    f"  Source: {item.get('source_file')}",
                    f"  Strengths: {strengths}",
                    f"  Risks: {risks}",
                    f"  Explanation: {item.get('explanation') or '-'}",
                ]
            )
        )
    return "\n\n".join(blocks)


def run_pipeline(
    jd_path: str,
    cv_paths: List[str],
    cv_dir: Optional[str],
    extractor: LLMClient,
    ranker: LLMClient,
    required_skills: Sequence[str],
    min_score: int,
    calibrate: bool,
    output_path: Optional[str],
) -> Dict[str, Any]:
    """
    Execute the full sorting workflow and return a serializable result object.

    Steps
    -----
    1. Load JD and CV documents from disk.
    2. Extract structured JD and CV profiles with LLM-1.
    3. Score each CV against the JD with LLM-2.
    4. Optionally recalibrate scores with a second LLM-2 pass.
    5. Filter, sort, print, and optionally write JSON.
    """
    job_text = load_document(jd_path)
    paths = collect_cv_paths(cv_paths, cv_dir, exclude_paths=[jd_path])
    named_cvs, skipped = load_named_documents(paths)

    print(f"Loaded job description: {jd_path}", file=sys.stderr)
    print(f"Loaded {len(named_cvs)} CV file(s).", file=sys.stderr)
    for item in skipped:
        print(f"Warning: skipped {item['source_file']}: {item['reason']}", file=sys.stderr)
    print(f"Extractor model (LLM-1): {extractor.model}", file=sys.stderr)
    print(f"Ranker model (LLM-2): {ranker.model}", file=sys.stderr)

    job = extract_job(extractor, job_text)
    print(f"Parsed job title: {job.get('job_title')}", file=sys.stderr)

    scored: List[Dict[str, Any]] = []
    for index, (source_name, cv_text) in enumerate(named_cvs, start=1):
        # A single failed LLM call (after retries) should cost one candidate,
        # not the whole ranking; the failure is reported in `skipped`.
        try:
            print(f"Extracting CV {index}/{len(named_cvs)}: {source_name}", file=sys.stderr)
            cv = extract_cv(extractor, source_name, cv_text)
            print(f"Scoring CV {index}/{len(named_cvs)}: {cv.get('candidate_name')}", file=sys.stderr)
            verdict = score_candidate(ranker, job, cv)
        except (RuntimeError, ValueError) as exc:
            print(f"Warning: skipped {source_name}: {exc}", file=sys.stderr)
            skipped.append({"source_file": source_name, "reason": str(exc)})
            continue
        record = {
            "candidate_id": f"cv_{index:03d}",
            "candidate_name": cv.get("candidate_name"),
            "source_file": source_name,
            "skills": cv.get("skills") or [],
            "structured_cv": cv,
            **verdict,
        }
        scored.append(record)

    if calibrate:
        print("Recalibrating scores across the shortlist (LLM-2).", file=sys.stderr)
        try:
            scored = recalibrate_scores(ranker, job, scored)
        except (RuntimeError, ValueError) as exc:
            # Independent scores are still valid, so keep them.
            print(f"Warning: calibration failed, using independent scores: {exc}", file=sys.stderr)

    scored_before_filters = scored
    scored = filter_by_required_skills(scored, required_skills)
    scored = apply_min_score(scored, min_score)
    ranked = sort_results(scored)

    payload = {
        "job": job,
        "models": {
            "extractor": extractor.model,
            "ranker": ranker.model,
        },
        "candidate_count": len(ranked),
        "scored_count": len(scored_before_filters),
        "ranked_candidates": ranked,
        "skipped": skipped,
    }

    print()
    print("=== Ranked CVs ===")
    print(render_table(ranked))
    print()
    print("=== Match explanations ===")
    print(render_explanations(ranked))
    if skipped:
        print()
        print("=== Skipped CVs ===")
        for item in skipped:
            print(f"- {item['source_file']}: {item['reason']}")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"\nWrote JSON results to {output_path}", file=sys.stderr)

    return payload
