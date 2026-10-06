"""
LLM-1 stage: turn raw job descriptions and resumes into structured JSON.

A smaller/faster model is preferred here because the task is information
extraction, not open-ended reasoning.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from llm_client import LLMClient
from prompts import CV_EXTRACT_USER, EXTRACTOR_SYSTEM, JD_EXTRACT_USER


def _as_list(value: Any) -> List[str]:
    """Coerce a JSON field into a list of stripped strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _as_int(value: Any, default: int = 0) -> int:
    """Best-effort integer conversion for years-of-experience style fields."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def normalize_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fill defaults for a parsed job-description object."""
    return {
        "job_title": str(payload.get("job_title") or "Untitled role").strip(),
        "seniority": str(payload.get("seniority") or "").strip(),
        "must_have_skills": _as_list(payload.get("must_have_skills")),
        "nice_to_have_skills": _as_list(payload.get("nice_to_have_skills")),
        "responsibilities": _as_list(payload.get("responsibilities")),
        "required_experience_years": _as_int(payload.get("required_experience_years")),
        "education": str(payload.get("education") or "").strip(),
        "keywords": _as_list(payload.get("keywords")),
        "summary": str(payload.get("summary") or "").strip(),
    }


def normalize_cv(payload: Dict[str, Any], source_name: str) -> Dict[str, Any]:
    """Validate and fill defaults for a parsed resume object."""
    name = str(payload.get("candidate_name") or "").strip()
    if not name:
        name = source_name
    experience = payload.get("experience") or []
    if not isinstance(experience, list):
        experience = []
    clean_experience = []
    for item in experience:
        if not isinstance(item, dict):
            continue
        clean_experience.append(
            {
                "title": str(item.get("title") or "").strip(),
                "company": str(item.get("company") or "").strip(),
                "duration": str(item.get("duration") or "").strip(),
                "highlights": _as_list(item.get("highlights")),
            }
        )
    return {
        "candidate_name": name,
        "source_file": source_name,
        "headline": str(payload.get("headline") or "").strip(),
        "years_experience": _as_int(payload.get("years_experience")),
        "education": _as_list(payload.get("education")),
        "skills": _as_list(payload.get("skills")),
        "experience": clean_experience,
        "projects": _as_list(payload.get("projects")),
        "achievements": _as_list(payload.get("achievements")),
        "summary": str(payload.get("summary") or "").strip(),
    }


def extract_job(client: LLMClient, job_text: str) -> Dict[str, Any]:
    """Run LLM-1 on a job description and return a normalized JSON profile."""
    payload = client.complete_json(
        EXTRACTOR_SYSTEM,
        JD_EXTRACT_USER.format(document=job_text),
    )
    return normalize_job(payload)


def extract_cv(client: LLMClient, source_name: str, cv_text: str) -> Dict[str, Any]:
    """Run LLM-1 on a resume and return a normalized JSON profile."""
    payload = client.complete_json(
        EXTRACTOR_SYSTEM,
        CV_EXTRACT_USER.format(source_name=source_name, document=cv_text),
    )
    return normalize_cv(payload, source_name)


# Vendor prefixes that LLM extraction adds inconsistently ("Apache Spark" vs "Spark").
_VENDOR_PREFIXES = ("apache ", "amazon ", "aws ", "microsoft ", "google ")


def _strip_vendor(skill: str) -> str:
    """Lowercase a skill name and drop a leading vendor prefix if something remains."""
    name = skill.strip().lower()
    for prefix in _VENDOR_PREFIXES:
        if name.startswith(prefix) and len(name) > len(prefix):
            return name[len(prefix):].strip()
    return name


def skills_match(required: str, owned: str) -> bool:
    """
    Decide whether a CV skill evidences a required skill.

    Plain substring matching is unsafe for short names ("r" is inside "spark",
    "go" is inside "django"), so containment is only accepted on word
    boundaries ("python" in "python 3", "aws" in "aws (s3, glue)"). Suffix
    compounds such as "pyspark" vs "spark" are accepted only when the
    shorter name has at least 4 characters. Both the raw and vendor-stripped
    names are tried, so "AWS" still matches "AWS (S3, Glue)".
    """
    raw_a, raw_b = required.strip().lower(), owned.strip().lower()
    return _names_match(raw_a, raw_b) or _names_match(
        _strip_vendor(required), _strip_vendor(owned)
    )


def _names_match(a: str, b: str) -> bool:
    """Core comparison used by skills_match on already-normalized names."""
    if not a or not b:
        return False
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < 2:
        return False
    # '+' and '#' count as word characters so "c" never matches "c++" or "c#".
    pattern = rf"(?<![a-z0-9+#]){re.escape(short)}(?![a-z0-9+#])"
    if re.search(pattern, long_):
        return True
    # Suffix only: prefixes give false positives such as "java" -> "javascript".
    return len(short) >= 4 and long_.endswith(short)


def overlap_evidence(job: Dict[str, Any], cv: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a transparent keyword-overlap signal for the ranker.

    This is not the final ranking; it is evidence so LLM-2 cannot ignore
    obvious skill matches or gaps.
    """
    cv_skills = [skill for skill in cv.get("skills", []) if skill.strip()]

    def _match(required: List[str]) -> Dict[str, List[str]]:
        """Split required skills into those evidenced by any CV skill and those missing."""
        matched, missing = [], []
        for skill in required:
            found = any(skills_match(skill, owned) for owned in cv_skills)
            (matched if found else missing).append(skill)
        return {"matched": matched, "missing": missing}

    must = _match(job.get("must_have_skills") or [])
    nice = _match(job.get("nice_to_have_skills") or [])
    must_total = max(len(job.get("must_have_skills") or []), 1)
    coverage = round(100.0 * len(must["matched"]) / must_total, 1)
    return {
        "must_have_matched": must["matched"],
        "must_have_missing": must["missing"],
        "nice_to_have_matched": nice["matched"],
        "nice_to_have_missing": nice["missing"],
        "must_have_coverage_pct": coverage,
    }


def format_overlap_block(overlap: Dict[str, Any]) -> str:
    """Pretty-print overlap evidence inside the ranker prompt."""
    return json.dumps(overlap, indent=2)
