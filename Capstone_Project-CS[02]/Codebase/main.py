"""
CV Sorting using LLMs (Capstone_Project-CS[02])

Entry point for terminal evaluation. This program ranks candidate CVs against
a job description using two complementary LLMs:

- LLM-1 (extractor): converts unstructured JD/CV text into structured JSON.
- LLM-2 (ranker): scores each structured CV against structured requirements
  and optionally recalibrates the shortlist.

API keys must be supplied at runtime (--api-key or environment variables).
They are never stored in source files.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Sequence

from llm_client import LLMClient, mask_key, resolve_api_key_with_source
from pipeline import run_pipeline


def parse_csv_list(value: str) -> List[str]:
    """Split a comma-separated CLI list into stripped tokens."""
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface used by evaluators."""
    parser = argparse.ArgumentParser(
        description=(
            "Rank candidate CVs against a job description using two LLMs "
            "(extractor + ranker)."
        )
    )
    parser.add_argument(
        "--jd",
        required=True,
        help="Path to the job description file (.txt, .md, .pdf, or .docx).",
    )
    parser.add_argument(
        "--cvs",
        nargs="*",
        default=[],
        help="Explicit CV file paths (.txt, .md, .pdf, or .docx).",
    )
    parser.add_argument(
        "--cv-dir",
        default=None,
        help="Directory containing CV files. Can be combined with --cvs.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "API key for the OpenAI-compatible provider. "
            "If omitted, OPENAI_API_KEY, GROQ_API_KEY, or LLM_API_KEY is used."
        ),
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help=(
            "Optional OpenAI-compatible base URL. Examples: "
            "https://api.groq.com/openai/v1 or http://localhost:11434/v1"
        ),
    )
    parser.add_argument(
        "--extractor-model",
        default="gpt-4o-mini",
        help="LLM-1 used for JD/CV information extraction (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--ranker-model",
        default="gpt-4o",
        help="LLM-2 used for scoring and ranking (default: gpt-4o).",
    )
    parser.add_argument(
        "--required-skills",
        default="",
        help="Optional comma-separated skills that every remaining candidate must have.",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=0,
        help="Drop candidates whose overall score is below this integer (0-100).",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip the second LLM-2 pass that recalibrates scores across the shortlist.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the full ranked JSON result.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Fail fast on combinations that cannot satisfy the project objective."""
    if not args.cvs and not args.cv_dir:
        raise SystemExit("Provide at least one CV via --cvs and/or --cv-dir.")
    if args.min_score < 0 or args.min_score > 100:
        raise SystemExit("--min-score must be between 0 and 100.")
    if args.extractor_model.strip() == args.ranker_model.strip():
        print(
            "Warning: extractor and ranker models are identical. "
            "The project requires two LLMs; pass different --extractor-model "
            "and --ranker-model values.",
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments, construct two LLM clients, and run ranking."""
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)

    api_key, key_source = resolve_api_key_with_source(args.api_key)
    if not api_key:
        parser.error(
            "Missing API key. Pass --api-key YOUR_KEY or export OPENAI_API_KEY / GROQ_API_KEY."
        )
    print(f"API key source: {key_source} ({mask_key(api_key)})", file=sys.stderr)

    # LLM-1 is deterministic extraction; LLM-2 is allowed mild sampling for judgement.
    extractor = LLMClient(
        api_key=api_key,
        model=args.extractor_model,
        api_base=args.api_base,
        temperature=0.0,
    )
    ranker = LLMClient(
        api_key=api_key,
        model=args.ranker_model,
        api_base=args.api_base,
        temperature=0.2,
    )

    try:
        payload = run_pipeline(
            jd_path=args.jd,
            cv_paths=list(args.cvs),
            cv_dir=args.cv_dir,
            extractor=extractor,
            ranker=ranker,
            required_skills=parse_csv_list(args.required_skills),
            min_score=args.min_score,
            calibrate=not args.no_calibrate,
            output_path=args.output,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        # Fatal for the whole run: unreadable JD, bad key/model, or unwritable output.
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    # Filters may legitimately leave zero candidates; zero *scored* means every CV failed.
    if payload["scored_count"] == 0:
        print("Error: no CV could be scored. See warnings above.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
