"""Regenerates Report/report.pdf (text-only, Times New Roman 12 pt, 1.5 spacing, <= 3 pages)."""

import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle

FONTS = "/System/Library/Fonts/Supplemental/"
pdfmetrics.registerFont(TTFont("TNR", FONTS + "Times New Roman.ttf"))
pdfmetrics.registerFont(TTFont("TNR-Bold", FONTS + "Times New Roman Bold.ttf"))
pdfmetrics.registerFont(TTFont("TNR-Italic", FONTS + "Times New Roman Italic.ttf"))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "Capstone_Project-CS[02]", "Report", "report.pdf")


def style(name, font="TNR", size=12, leading=18, before=0, after=6, align=4):
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=leading,
                          spaceBefore=before, spaceAfter=after, alignment=align)


TITLE = style("t", "TNR-Bold", 16, 20, 0, 6, 1)
SUB = style("s", "TNR-Italic", 12, 16, 0, 10, 1)
H = style("h", "TNR-Bold", 12, 18, 8, 3, 0)
BODY = style("b")
BULLET = style("bl", after=1)
CELL = style("c", size=12, leading=15, after=0, align=0)
CELL_B = style("cb", "TNR-Bold", 12, 15, 0, 0, 0)

ABSTRACT = (
    "Recruiters spend substantial time reading unstructured resumes against loosely written "
    "job descriptions. This capstone implements a terminal-based CV sorting system that ranks "
    "candidate documents for a role using two complementary large language models. The first "
    "model (LLM-1) extracts a structured job profile and structured candidate profiles from "
    "PDF, DOCX, Markdown, or plain-text files. The second model (LLM-2) scores each candidate "
    "with an explicit rubric covering skills, experience, domain fit, and education, and then "
    "recalibrates scores across the shortlist so independently judged CVs stay comparable. "
    "A deterministic, word-boundary keyword-overlap signal is injected into the ranker prompt "
    "as evidence, so obvious gaps cannot be ignored while the final judgement remains with the "
    "reasoning model. The program starts from main.py, takes the API key only at runtime, "
    "prints a ranked table with per-candidate explanations, and supports recruiter filters and "
    "JSON export. Unreadable CVs or failed model calls are reported and skipped instead of "
    "aborting the run. On a labelled test set of five CVs in four formats, including a "
    "keyword-stuffed decoy, the system reproduced the expected ordering exactly in three "
    "independent runs, with per-candidate score variation of at most six points."
)

INTRO = (
    "Applicant tracking still depends on keyword filters or manual reading. Both miss context: "
    "a candidate may describe relevant work without naming a tool, or list a skill with no "
    "evidence of using it. Instruction-tuned LLMs can parse messy documents and justify a match, "
    "but a single model asked to rank CVs mixes extraction errors with scoring bias. This "
    "project therefore separates extraction from judgement, which also gives each of the two "
    "required LLMs a clear role. The work is a command-line tool because evaluation is "
    "terminal-based and a GUI would not improve the ranking objective."
)

PROBLEM = (
    "Given one job description and N candidate CVs in common document formats, produce a "
    "stable ranked list with overall scores, missing must-have skills, and a written rationale "
    "per candidate, without embedded secrets or a front-end, runnable with the evaluator's own "
    "API key and test files."
)

METHOD = (
    "The pipeline is linear and started by main.py. document_loader.py reads UTF-8 text and "
    "Markdown, PDFs via pdfplumber, and DOCX via ZIP/XML parsing (no python-docx dependency); "
    "hidden and unsupported files are ignored, and the JD is excluded if it sits in the CV "
    "folder. llm_client.py wraps the OpenAI chat-completions SDK with an optional base URL, so "
    "the same code runs against OpenAI, Gemini, Groq, or a local Ollama server. It requests "
    "JSON mode (falling back when unsupported), repairs malformed JSON once, retries transient "
    "errors, and fails immediately on authentication or unknown-model errors. LLM-1 fills a "
    "fixed JSON schema for the job and for each CV at temperature 0. Before scoring, "
    "extractor.overlap_evidence computes must-have and nice-to-have coverage. Matching is "
    "case-insensitive on word boundaries, tolerates vendor prefixes (Apache Spark vs Spark) and "
    "suffix compounds (PySpark vs Spark), and rejects false positives such as R in Spark or "
    "Java in JavaScript. LLM-2 receives this evidence with both JSON profiles and returns "
    "0-100 sub-scores and an overall score weighted skills 40%, experience 30%, domain 20%, "
    "education 10%, plus strengths, risks, and an explanation. A calibration call then lets "
    "LLM-2 see the whole shortlist and adjust overall scores for consistency. Results are "
    "filtered (--required-skills, --min-score) and sorted by overall score, then skills score, "
    "then fewer missing must-haves. A failure on one CV is logged and skipped; the program "
    "exits with an error only if the JD cannot be processed or no CV could be scored."
)

MODELS = (
    "Model choice follows the task split. Extraction is schema filling, where latency and cost "
    "dominate, so a small fast model is used: gpt-4o-mini by default, gemini-3.5-flash-lite in "
    "the validation runs below. Ranking needs trade-off reasoning and trustworthy explanations, "
    "so a stronger model is used: gpt-4o by default, gemini-3.8-flash in validation. The same "
    "small-extractor / strong-ranker split is available on Groq (Llama 3.1 8B and Llama 3.3 "
    "70B) or Ollama for fully local, privacy-preserving use."
)

RESULTS = (
    "The system was evaluated on a labelled set: one Senior Data Engineer JD and five CVs in "
    "PDF, DOCX, TXT and MD, with a human reference order. Sara Khan is a deliberate decoy who "
    "lists every JD keyword (100% keyword coverage) but whose history is entirely non-technical. "
    "Table 1 shows overall scores across two full runs and one run without calibration."
)

ANALYSIS = (
    "All three runs reproduced the reference order exactly (Kendall tau = 1.0), and no score "
    "moved by more than six points between runs. The decoy was ranked fourth with a score of "
    "20-25: LLM-2 explicitly flagged her skills as self-reported with no supporting work, which "
    "shows the value of separating keyword evidence from judgement. Calibration changed scores "
    "by at most six points and did not change the order on this set, so its benefit is mainly "
    "for larger shortlists. With a corrupt DOCX and an empty file added, both were skipped with "
    "warnings and the other CVs were still ranked; the filters --required-skills airflow "
    "--min-score 50 correctly left only the top candidate. An offline suite of 41 unit tests "
    "(fake LLMs) covers loaders, JSON repair, retries, matching, ranking, filters, and CLI "
    "errors. A run takes about 80-100 seconds for five CVs because calls are sequential."
)

CONCLUSION = (
    "The system meets the capstone objective: CVs and a job description in, a ranked and "
    "explained list out, using two role-specialised LLMs. Future work: parallel model calls "
    "for large CV sets, embedding pre-filtering, OCR for scanned PDFs, and a recruiter "
    "feedback file that re-runs only the ranking stage."
)

TABLE = [
    ["Candidate (format)", "Expected", "Run 1", "Run 2", "No calibration"],
    ["Priya Sharma (PDF)", "1", "96", "97", "97"],
    ["Rahul Verma (DOCX)", "2", "71", "72", "70"],
    ["Emily Chen (TXT)", "3", "39", "43", "37"],
    ["Sara Khan, decoy (TXT)", "4", "25", "22", "20"],
    ["John Doe (MD)", "5", "13", "16", "17"],
]


def build_table():
    rows = [[Paragraph(c, CELL_B if r == 0 else CELL) for c in row] for r, row in enumerate(TABLE)]
    t = Table(rows, colWidths=[2.3 * inch, 0.9 * inch, 0.8 * inch, 0.8 * inch, 1.3 * inch])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def main():
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=inch, rightMargin=inch,
                            topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                            title="CV Sorting using LLMs", author="Capstone Project CS[02]")
    story = [
        Paragraph("CV Sorting using LLMs", TITLE),
        Paragraph("Capstone Project CS[02]", SUB),
        Paragraph("Abstract", H), Paragraph(ABSTRACT, BODY),
        Paragraph("1. Introduction", H), Paragraph(INTRO, BODY),
        Paragraph("2. Problem Statement", H), Paragraph(PROBLEM, BODY),
        Paragraph("3. Objectives", H),
        Paragraph("• Rank multiple CVs against one job description from the terminal.", BULLET),
        Paragraph("• Use two LLMs with distinct roles: extraction (LLM-1) and scoring (LLM-2).", BULLET),
        Paragraph("• Accept PDF, DOCX, Markdown and text; tolerate bad files without aborting.", BULLET),
        Paragraph("• Return inspectable scores, missing skills, and optional JSON output.", BULLET),
        Paragraph("4. Methodology and Implementation", H), Paragraph(METHOD, BODY),
        Paragraph("5. Model Selection", H), Paragraph(MODELS, BODY),
        Paragraph("6. Results and Analysis", H), Paragraph(RESULTS, BODY),
        Paragraph("Table 1. Overall score (0-100) per candidate; all runs ranked in expected order.",
                  style("cap", "TNR-Italic", 12, 15, 2, 4, 0)),
        build_table(),
        Paragraph(ANALYSIS, style("b2", before=8)),
        Paragraph("7. Conclusion and Future Work", H), Paragraph(CONCLUSION, BODY),
        Paragraph("References", H),
        Paragraph("1. OpenAI Python SDK, Chat Completions API. 2. Google Gemini OpenAI-compatible "
                  "API. 3. Groq and Ollama OpenAI-compatible endpoints. 4. pdfplumber. "
                  "5. Capstone brief: CV Sorting using LLMs (CS[02]).", BODY),
    ]

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("TNR", 10)
        canvas.drawCentredString(A4[0] / 2.0, 0.5 * inch, str(d.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("Wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
