"""
Builds a labelled evaluation set (1 JD + 6 CVs in mixed formats) under eval/data/.

Expected ordering (human judgement) is written to eval/data/expected_ranking.json
so the live run can be scored with rank-correlation metrics.
"""

import json
import os
import zipfile

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CVS = os.path.join(DATA, "cvs")

JD = """Senior Data Engineer - Retail Analytics Platform

We are hiring a Senior Data Engineer to build and operate batch and streaming
data pipelines for our retail analytics platform.

Must have:
- 5+ years of professional data engineering experience
- Strong Python and SQL
- Apache Spark (PySpark) for large-scale processing
- Workflow orchestration with Apache Airflow
- Cloud data services on AWS (S3, Glue, EMR or Redshift)

Nice to have:
- Kafka or other streaming platforms
- dbt, Docker, Terraform
- Retail or e-commerce domain experience

Education: Bachelor's degree in Computer Science or a related field.

Responsibilities: design ETL/ELT pipelines, optimise Spark jobs, own data quality
checks, mentor junior engineers, and partner with analysts.
"""

PRIYA = [
    "Priya Sharma",
    "Senior Data Engineer | priya.sharma@example.com",
    "Summary: 7 years building data platforms for e-commerce and retail.",
    "Experience:",
    "ShopMart (2020-2025) Senior Data Engineer: Built PySpark pipelines on AWS EMR",
    "processing 4 TB/day; orchestrated 120+ Airflow DAGs; migrated warehouse to",
    "Redshift; introduced dbt models and Great Expectations data-quality checks;",
    "mentored 4 engineers. Streaming ingestion with Kafka.",
    "DataCorp (2018-2020) Data Engineer: Python and SQL ETL into S3 and Glue.",
    "Skills: Python, SQL, PySpark, Apache Airflow, AWS (S3, Glue, EMR, Redshift),",
    "Kafka, dbt, Docker, Terraform",
    "Education: B.Tech Computer Science, IIT Delhi",
]

RAHUL = [
    "Rahul Verma",
    "Data Engineer",
    "Experience: 5 years.",
    "FinServe Ltd (2021-2025) Data Engineer - Spark batch jobs in Python on AWS EMR,",
    "SQL transformations in Redshift, scheduling with cron and AWS Step Functions.",
    "Analytics Hub (2020-2021) Junior Data Engineer - SQL reporting pipelines.",
    "Skills: Python, SQL, Apache Spark, AWS, Redshift, Docker",
    "Education: B.E. Information Technology",
]

EMILY = """Emily Chen
Data Analyst

3 years of experience in retail analytics.

RetailCo (2022-2025) Data Analyst: wrote SQL queries in Snowflake, built Tableau
dashboards for merchandising, automated weekly reports with Python pandas scripts.

Skills: SQL, Python (pandas), Tableau, Excel, Snowflake
Education: B.Sc. Statistics
"""

JOHN = """# John Doe
Frontend Engineer

6 years building web applications.

- WebWorks (2019-2025): React and TypeScript single-page apps, Node.js APIs,
  Jest testing, CI with GitHub Actions.

**Skills:** JavaScript, TypeScript, React, Node.js, CSS, Jest
**Education:** B.S. Computer Science
"""

# Keyword-stuffed CV: every JD keyword is listed but nothing in the work history
# supports it. A good ranker should not place this candidate near the top.
SARA = """Sara Khan

Skills: Python, SQL, Spark, PySpark, Airflow, AWS, S3, Glue, EMR, Redshift,
Kafka, dbt, Docker, Terraform, data engineering, ETL, big data

Experience:
Bean Street Cafe (2021-2025) Shift Supervisor: managed staff rota, handled
cash reconciliation, trained new baristas.
City Mart (2019-2021) Retail Associate: stocked shelves, customer service.

Education: High school diploma
"""

EXPECTED = [
    "Priya Sharma",
    "Rahul Verma",
    "Emily Chen",
    "Sara Khan",
    "John Doe",
]


def write_pdf(path, lines):
    """Render lines into a single-page text PDF (has a real text layer)."""
    pdf = canvas.Canvas(path, pagesize=A4)
    y = A4[1] - 60
    for line in lines:
        pdf.drawString(50, y, line)
        y -= 16
    pdf.save()


def write_docx(path, paragraphs):
    """Write a minimal but valid DOCX containing one paragraph per line."""
    body = "".join(
        f"<w:p><w:r><w:t xml:space=\"preserve\">{p}</w:t></w:r></w:p>" for p in paragraphs
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/></Relationships>'
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document)


def main():
    os.makedirs(CVS, exist_ok=True)
    with open(os.path.join(DATA, "job.txt"), "w", encoding="utf-8") as f:
        f.write(JD)
    write_pdf(os.path.join(CVS, "priya_sharma.pdf"), PRIYA)
    write_docx(os.path.join(CVS, "rahul_verma.docx"), RAHUL)
    with open(os.path.join(CVS, "emily_chen.txt"), "w", encoding="utf-8") as f:
        f.write(EMILY)
    with open(os.path.join(CVS, "john_doe.md"), "w", encoding="utf-8") as f:
        f.write(JOHN)
    with open(os.path.join(CVS, "sara_khan.txt"), "w", encoding="utf-8") as f:
        f.write(SARA)
    # Noise that the loader must ignore.
    with open(os.path.join(CVS, ".DS_Store"), "wb") as f:
        f.write(b"\x00\x01")
    with open(os.path.join(CVS, "notes.csv"), "w") as f:
        f.write("not,a,cv\n")
    with open(os.path.join(DATA, "expected_ranking.json"), "w") as f:
        json.dump(EXPECTED, f, indent=2)
    print("Wrote evaluation data to", DATA)


if __name__ == "__main__":
    main()
