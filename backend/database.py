import os
import json
from datetime import datetime
import mysql.connector
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "medical_rag")


print("MYSQL_USER =", MYSQL_USER)
print("MYSQL_PASSWORD =", repr(MYSQL_PASSWORD))
print("MYSQL_DATABASE =", MYSQL_DATABASE)

TOKEN_LIMIT = 50000


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
    # Step 1: make sure the database itself exists
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

    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_activity_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            action_type VARCHAR(50) NOT NULL,
            metadata JSON,
            status ENUM('success', 'error') DEFAULT 'success',
            error_msg TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    cur.close()
    conn.close()

    # Step 3: add token-tracking columns to users (idempotent, for existing DBs)
    _ensure_token_columns()


def _ensure_token_columns():
    """
    Adds tokens_used / tokens_month columns to users if they don't exist yet.
    Safe to call every startup — swallows the "duplicate column" error.
    """
    with get_db() as conn:
        cur = conn.cursor()
        for stmt in [
            "ALTER TABLE users ADD COLUMN tokens_used INT DEFAULT 0",
            "ALTER TABLE users ADD COLUMN tokens_month VARCHAR(7) DEFAULT NULL",
        ]:
            try:
                cur.execute(stmt)
            except mysql.connector.Error:
                pass  # column already exists
        conn.commit()
        cur.close()


@contextmanager
def get_db():
    conn = _connect(use_database=True)
    try:
        yield conn
    finally:
        conn.close()



# Users

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


# Token usage (estimated, resets monthly)


def _current_month() -> str:
    return datetime.now().strftime("%Y-%m")


def estimate_tokens(text: str) -> int:
    """Rough char/4 estimate. Not exact — good enough for a soft usage cap."""
    return max(1, len(text or "") // 4)


def get_token_usage(user_id: int) -> int:
    """
    Tokens used so far this calendar month. Auto-resets the counter
    (in the DB) the first time it's checked after a month rollover.
    """
    with get_db() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT tokens_used, tokens_month FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
        cur.close()
        if not row:
            return 0

        this_month = _current_month()
        if row["tokens_month"] != this_month:
            cur2 = conn.cursor()
            cur2.execute(
                "UPDATE users SET tokens_used = 0, tokens_month = %s WHERE id = %s",
                (this_month, user_id),
            )
            conn.commit()
            cur2.close()
            return 0

        return row["tokens_used"] or 0


def add_tokens(user_id: int, count: int):
    """
    Adds `count` estimated tokens to the user's running total for the
    current month. Call get_token_usage() first if you need the rollover
    check to happen before adding (add_tokens itself doesn't reset).
    """
    if count <= 0:
        return
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET tokens_used = tokens_used + %s, tokens_month = %s WHERE id = %s",
            (count, _current_month(), user_id),
        )
        conn.commit()
        cur.close()



# Reports

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


def log_activity(
    user_id: int,
    action_type: str,
    metadata: dict = None,
    status: str = "success",
    error_msg: str = None,
):
    """
    Call this anywhere in your FastAPI routes to record what a user did.

    Examples:
        log_activity(user_id, "query", {"query": "What is diabetes?"})
        log_activity(user_id, "upload", {"filename": "report.pdf"})
        log_activity(user_id, "report_generated", {"report_id": 42, "filename": "ecg.pdf"})
        log_activity(user_id, "report_deleted", {"report_id": 42}, status="success")
        log_activity(user_id, "upload", {"filename": "bad.pdf"}, status="error", error_msg="OCR failed")
    """
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO user_activity_logs
                   (user_id, action_type, metadata, status, error_msg)
                   VALUES (%s, %s, %s, %s, %s)""",
                (
                    user_id,
                    action_type,
                    json.dumps(metadata) if metadata else None,
                    status,
                    error_msg,
                ),
            )
            conn.commit()
            cur.close()
    except Exception as e:
        # Logging should never crash your main app
        print(f"[activity_log] Failed to log activity for user {user_id}: {e}")


def get_activity_for_user(user_id: int, limit: int = 50):
    """
    Returns the last `limit` activity entries for a user — newest first.
    Use this in a /my-activity FastAPI endpoint.
    """
    with get_db() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """SELECT id, action_type, metadata, status, error_msg, created_at
               FROM user_activity_logs
               WHERE user_id = %s
               ORDER BY created_at DESC
               LIMIT %s""",
            (user_id, limit),
        )
        rows = cur.fetchall()
        cur.close()
        for row in rows:
            row["metadata"] = json.loads(row["metadata"]) if row["metadata"] else {}
        return rows


def get_activity_summary_for_user(user_id: int):
    """
    Returns a quick count breakdown per action_type.
    Useful for a stats card on the frontend.

    Returns something like:
        [
            {"action_type": "query", "count": 24},
            {"action_type": "upload", "count": 5},
            {"action_type": "report_generated", "count": 5},
        ]
    """
    with get_db() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """SELECT action_type, COUNT(*) as count
               FROM user_activity_logs
               WHERE user_id = %s
               GROUP BY action_type
               ORDER BY count DESC""",
            (user_id,),
        )
        rows = cur.fetchall()
        cur.close()
        return rows
    

def reset_all_tokens():

        """
    Resets tokens_used to 0 for every user and stamps tokens_month
    to the current month. Meant to be called by a scheduled job
    at midnight on the 1st of each month.
    """
        
        this_month = _current_month()
        with get_db() as conn:
            cur = conn.cursor()
        cur.execute(
            "UPDATE users SET tokens_used = 0, tokens_month = %s",
            (this_month,),
        )
        affected = cur.rowcount
        conn.commit()
        cur.close()

        return affected