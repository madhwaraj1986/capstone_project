"""
Prompt templates for the two-LLM pipeline.

LLM-1 (extractor) converts messy resume/JD text into comparable JSON.
LLM-2 (ranker) scores a structured candidate against structured requirements.
Keeping prompts in one module makes the dual-model design easy to inspect.
"""

# LLM-1 system prompt shared by job and CV extraction; forbids invented facts.
EXTRACTOR_SYSTEM = """You are an information-extraction engine for recruiting.
Return ONLY a valid JSON object. Do not invent employers, degrees, or skills that
are not supported by the source text. If a field is missing, use an empty string
or an empty list. Normalize skill names to concise lowercase tokens where possible.
"""

# LLM-1: job description -> requirements JSON. Doubled braces are literal JSON
# braces escaped for str.format(); {document} is the only placeholder.
JD_EXTRACT_USER = """Extract hiring requirements from the job description below.

Return JSON with this exact shape:
{{
  "job_title": "string",
  "seniority": "string",
  "must_have_skills": ["string"],
  "nice_to_have_skills": ["string"],
  "responsibilities": ["string"],
  "required_experience_years": 0,
  "education": "string",
  "keywords": ["string"],
  "summary": "2-4 sentence summary of the role"
}}

JOB DESCRIPTION:
{document}
"""

# LLM-1: resume text -> candidate profile JSON (same schema for every CV so profiles are comparable).
CV_EXTRACT_USER = """Extract a structured candidate profile from the resume text.

Return JSON with this exact shape:
{{
  "candidate_name": "string",
  "headline": "string",
  "years_experience": 0,
  "education": ["string"],
  "skills": ["string"],
  "experience": [
    {{"title": "string", "company": "string", "duration": "string", "highlights": ["string"]}}
  ],
  "projects": ["string"],
  "achievements": ["string"],
  "summary": "3-6 sentence factual summary of the candidate"
}}

RESUME SOURCE FILE: {source_name}

RESUME TEXT:
{document}
"""

# LLM-2 system prompt: score only from provided evidence, with one rubric for all candidates.
RANKER_SYSTEM = """You are a rigorous technical recruiter. Score candidates only from
the provided structured evidence. Do not reward unstated claims. Apply the rubric
consistently across candidates. Return ONLY a valid JSON object.
"""

# LLM-2: one candidate vs the job. The weights here are the scoring rubric described in the report.
RANKER_USER = """Score this candidate against the job requirements.

Scoring rubric (integers 0-100):
- skills_score: coverage of must-have skills, then nice-to-have skills
- experience_score: relevance and depth of work history vs required years
- education_score: degree/field fit
- domain_score: similarity of past domains/responsibilities to this role
- overall_score: weighted judgement (skills 40%, experience 30%, domain 20%, education 10%)

Also return:
- matched_must_have: skills clearly evidenced
- missing_must_have: must-have skills not evidenced
- strengths: short bullets
- risks: short bullets
- explanation: 4-8 sentences a recruiter can trust

Deterministic keyword overlap (pre-computed, use as evidence, not as the final score):
{overlap_block}

JOB (structured JSON):
{job_json}

CANDIDATE (structured JSON):
{cv_json}
"""

# LLM-2 calibration pass: sees the whole shortlist at once to fix drift between independent scores.
RERANK_SYSTEM = """You are a calibration judge. Given independently scored candidates
for the SAME job, you may slightly adjust overall scores to keep them comparable.
Do not change the ranking dramatically without evidence. Return ONLY JSON.
"""

# LLM-2 calibration: returns adjusted overall scores keyed by candidate_id.
RERANK_USER = """The following candidates were scored independently. Recalibrate
overall_score values if the relative ordering is inconsistent, keeping each score
in 0-100. Preserve candidate_id values.

Return:
{{
  "rankings": [
    {{"candidate_id": "string", "overall_score": 0, "tie_break_note": "string"}}
  ]
}}

JOB TITLE: {job_title}

CANDIDATES:
{candidates_block}
"""
