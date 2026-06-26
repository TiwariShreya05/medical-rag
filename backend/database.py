import os
import json
import mysql.connector 
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "") 
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "medical_rag")

# ✅ prints AFTER variables are define
print("MYSQL_USER =", MYSQL_USER)
print("MYSQL_PASSWORD =", repr(MYSQL_PASSWORD)) 
print("MYSQL_DATABASE =", MYSQL_DATABASE)

def _connect(use_database=True):
    config = {
        "host": MYSQL_HOST,
        "port": MYSQL_PORT,
        "user": MYSQL_USER,
        "password": MYSQL_PASSWORD,
    }
    if use_database:
        config["database"] = MYSQL_DATABASE
    return mysql.connector.connect(**config)

def get_db_connection():
    return _connect()


def init_db():
    # Step 1: make sure the database itself exists (connect without selecting one)
    conn = _connect(use_database=False)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE}")
    conn.commit()
    cur.close()
    conn.close()

    # Step 2: create tables inside that database
    conn = _connect(use_database=True)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            filename TEXT,
            patient_info TEXT,
            thought TEXT,
            observation TEXT,
            search_terms TEXT,
            analysis TEXT,
            sources TEXT,
            extraction_method TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    conn.commit()
    cur.close()
    conn.close()


@contextmanager
def get_db():
    conn = _connect(use_database=True)
    try:
        yield conn
    finally:
        conn.close()


#Users

def get_user_by_username(username: str):
    with get_db() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        row = cur.fetchone()
        cur.close()
        return row
    
def get_user_count() -> int:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users")
        count = cur.fetchone()[0]
        cur.close()
        return count    


def create_user(username: str, password_hash: str):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
            (username, password_hash),
        )
        conn.commit()
        new_id = cur.lastrowid
        cur.close()
        return new_id


#Reports

def save_report(user_id, filename, patient_info, thought, observation,
                 search_terms, analysis, sources, extraction_method):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO reports
               (user_id, filename, patient_info, thought, observation,
                search_terms, analysis, sources, extraction_method)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                user_id,
                filename,
                json.dumps(patient_info),
                thought,
                observation,
                json.dumps(search_terms),
                analysis,
                json.dumps(sources),
                extraction_method,
            ),
        )
        conn.commit()
        new_id = cur.lastrowid
        cur.close()
        return new_id


def get_reports_for_user(user_id):
    """List (without full analysis text) of everything this user has analyzed — newest first."""
    with get_db() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """SELECT id, filename, patient_info, created_at
               FROM reports WHERE user_id = %s ORDER BY created_at DESC""",
            (user_id,),
        )
        rows = cur.fetchall()
        cur.close()
        results = []
        for row in rows:
            row["patient_info"] = json.loads(row["patient_info"]) if row["patient_info"] else None
            results.append(row)
        return results


def get_report_by_id(report_id, user_id):
    """
    Filters by user_id too — so even if someone guesses another person's
    report id, they can't fetch it. This is what keeps reports inside the
    same account they were created under.
    """
    with get_db() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT * FROM reports WHERE id = %s AND user_id = %s",
            (report_id, user_id),
        )
        row = cur.fetchone()
        cur.close()
        if not row:
            return None
        row["patient_info"] = json.loads(row["patient_info"]) if row["patient_info"] else None
        row["search_terms"] = json.loads(row["search_terms"]) if row["search_terms"] else []
        row["sources"] = json.loads(row["sources"]) if row["sources"] else []
        return row
    
def delete_report(report_id: int, user_id: int):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM reports WHERE id = %s AND user_id = %s",
            (report_id, user_id)
        )
        conn.commit()
        cur.close()
