# AI Resume Screening System

An end-to-end NLP project that ranks candidate resumes against a job description using semantic embeddings and cosine similarity.

## Features

- Parse resume PDFs into text
- Normalize job/resume text with lightweight preprocessing
- Generate semantic embeddings with `all-MiniLM-L6-v2`
- Compute match scores (%) with cosine similarity
- Rank candidates from best to worst fit
- Visualize ranking in a Streamlit dashboard

## Project Structure

```text
AI-Resume-Screening-System/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── resumes/
│   └── job_description.txt
├── models/
├── src/
│   ├── embedding.py
│   ├── parser.py
│   ├── preprocess.py
│   ├── ranking.py
│   └── similarity.py
├── tests/
│   ├── test_preprocess.py
│   ├── test_ranking.py
│   └── test_similarity.py
├── requirements.txt
└── README.md
```

## Workflow

1. Upload resumes (PDF) and provide a job description.
2. Parse resumes into raw text.
3. Preprocess text (lowercase, token cleanup, stopword removal).
4. Create embedding vectors using sentence-transformers.
5. Calculate cosine similarity against the job vector.
6. Convert to percentage and rank candidates.
7. Show sorted results and a bar chart.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run App

```bash
streamlit run app/streamlit_app.py
```

## Run Tests

```bash
pytest -q
```

## Notes

- If `sentence-transformers` is not installed, the UI shows a clear error message instead of crashing.
- The current preprocessing intentionally avoids runtime corpus downloads, making it friendlier for restricted environments.
