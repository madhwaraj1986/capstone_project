"""
LLM-2 stage: score and rank structured candidates against a structured job.

A stronger reasoning model is preferred here because the task requires
trade-off judgement, not just field copying.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from extractor import format_overlap_block, overlap_evidence
from llm_client import LLMClient
from prompts import RANKER_SYSTEM, RANKER_USER, RERANK_SYSTEM, RERANK_USER


def _clip_score(value: Any, default: int = 0) -> int:
    """Force a score into the inclusive integer range 0-100."""
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        number = default
    return max(0, min(100, number))


def _as_list(value: Any) -> List[str]:
    """Coerce bullets/skills into a list of strings."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def normalize_score(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the ranker JSON and clip numeric fields."""
    return {
        "skills_score": _clip_score(payload.get("skills_score")),
        "experience_score": _clip_score(payload.get("experience_score")),
        "education_score": _clip_score(payload.get("education_score")),
        "domain_score": _clip_score(payload.get("domain_score")),
        "overall_score": _clip_score(payload.get("overall_score")),
        "matched_must_have": _as_list(payload.get("matched_must_have")),
        "missing_must_have": _as_list(payload.get("missing_must_have")),
        "strengths": _as_list(payload.get("strengths")),
        "risks": _as_list(payload.get("risks")),
        "explanation": str(payload.get("explanation") or "").strip(),
    }


def score_candidate(
    client: LLMClient, job: Dict[str, Any], cv: Dict[str, Any]
) -> Dict[str, Any]:
    """Ask LLM-2 for a rubric score of one candidate versus the job."""
    overlap = overlap_evidence(job, cv)
    payload = client.complete_json(
        RANKER_SYSTEM,
        RANKER_USER.format(
            overlap_block=format_overlap_block(overlap),
            job_json=json.dumps(job, indent=2),
            cv_json=json.dumps(cv, indent=2),
        ),
    )
    scored = normalize_score(payload)
    scored["keyword_overlap"] = overlap
    return scored


def recalibrate_scores(
    client: LLMClient, job: Dict[str, Any], results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Optional second pass of LLM-2: keep independent scores comparable.

    Independent scoring can drift when candidates are processed one-by-one.
    This calibration step looks at the whole shortlist at once.
    """
    if len(results) < 2:
        return results

    compact = []
    for item in results:
        compact.append(
            {
                "candidate_id": item["candidate_id"],
                "candidate_name": item["candidate_name"],
                "overall_score": item["overall_score"],
                "skills_score": item["skills_score"],
                "missing_must_have": item.get("missing_must_have", []),
                "summary": (item.get("explanation") or "")[:400],
            }
        )

    payload = client.complete_json(
        RERANK_SYSTEM,
        RERANK_USER.format(
            job_title=job.get("job_title", ""),
            candidates_block=json.dumps(compact, indent=2),
        ),
    )
    adjustments = {}
    for row in payload.get("rankings") or []:
        if not isinstance(row, dict) or not row.get("candidate_id"):
            continue
        if "overall_score" not in row:
            continue
        adjustments[str(row["candidate_id"])] = {
            "overall_score": _clip_score(row.get("overall_score")),
            "tie_break_note": str(row.get("tie_break_note") or "").strip(),
        }

    calibrated: List[Dict[str, Any]] = []
    for item in results:
        updated = dict(item)
        adj = adjustments.get(item["candidate_id"])
        if adj:
            updated["pre_calibration_score"] = item["overall_score"]
            updated["overall_score"] = adj["overall_score"]
            if adj["tie_break_note"]:
                updated["calibration_note"] = adj["tie_break_note"]
        calibrated.append(updated)
    return calibrated


def sort_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort by overall_score descending.

    Ties are broken by skills_score, then fewer missing must-have skills,
    then candidate name for a stable, reproducible order.
    """
    def key(item: Dict[str, Any]) -> tuple:
        """Ascending sort key; scores are negated so higher scores come first."""
        missing = len(item.get("missing_must_have") or [])
        return (
            -int(item.get("overall_score") or 0),
            -int(item.get("skills_score") or 0),
            missing,
            str(item.get("candidate_name") or ""),
        )

    ranked = sorted(results, key=key)
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    return ranked
