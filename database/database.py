from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "screening_history.db"


def init_history_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS screenings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                job_role TEXT NOT NULL,
                candidate_names TEXT NOT NULL,
                scores TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_screening_history(job_role: str, candidate_names: list[str], scores: list[float]) -> None:
    init_history_db()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO screenings (created_at, job_role, candidate_names, scores) VALUES (?, ?, ?, ?)",
            (
                timestamp,
                job_role,
                " | ".join(candidate_names),
                " | ".join(f"{score:.2f}%" for score in scores),
            ),
        )
        conn.commit()


def load_history() -> pd.DataFrame:
    init_history_db()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT created_at AS Date, job_role AS Job_Role, candidate_names AS Candidates, scores AS Scores "
            "FROM screenings ORDER BY id DESC",
            conn,
        )
