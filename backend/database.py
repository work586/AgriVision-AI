from contextlib import contextmanager
from pathlib import Path
import hashlib
import hmac
import os
import re
import sqlite3
import threading
from datetime import datetime


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATABASE_DIR = ROOT / "data"
DATABASE_FILE = DATABASE_DIR / "agrivision.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    """Create and return a SQLite database connection."""

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_FILE, timeout=10)
    connection.row_factory = sqlite3.Row

    return connection


@contextmanager
def _db():
    """Commit on success, roll back on error, ALWAYS close."""

    connection = get_connection()

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


# ============================================================
# INITIALIZE DATABASE (runs the real work only once per process)
# ============================================================

_INIT_LOCK = threading.Lock()
_INITIALIZED = False


def initialize_database():
    """Create tables if missing and add new columns to older databases."""

    global _INITIALIZED

    if _INITIALIZED and DATABASE_FILE.exists():
        return

    with _INIT_LOCK:

        if _INITIALIZED and DATABASE_FILE.exists():
            return

        with _db() as connection:
            cursor = connection.cursor()

            # ---------- USERS ----------
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

            # ---------- ANALYSES ----------
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    analyzed_at TEXT NOT NULL,
                    crop TEXT,
                    crop_confidence REAL,
                    disease TEXT,
                    disease_confidence REAL,
                    status TEXT,
                    disease_model_available INTEGER,
                    price_available INTEGER,
                    price_date TEXT,
                    min_price REAL,
                    max_price REAL,
                    avg_modal_price REAL,
                    predicted_price REAL,
                    predicted_price_date TEXT,
                    price_trend TEXT,
                    price_recommendation TEXT
                )
                """
            )

            # Older databases: add the missing columns in place.
            new_columns = {
                "user_id": "INTEGER",
                "predicted_price": "REAL",
                "predicted_price_date": "TEXT",
                "price_trend": "TEXT",
                "price_recommendation": "TEXT",
            }

            cursor.execute("PRAGMA table_info(analyses)")
            existing_columns = {row["name"] for row in cursor.fetchall()}

            for column_name, column_type in new_columns.items():
                if column_name not in existing_columns:
                    cursor.execute(
                        f"ALTER TABLE analyses ADD COLUMN {column_name} {column_type}"
                    )

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_analyses_user ON analyses(user_id)"
            )

        _INITIALIZED = True


# ============================================================
# USERS: REGISTER + LOGIN
# ============================================================

_PBKDF2_ROUNDS = 200_000
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.]{3,30}$")


def _hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _PBKDF2_ROUNDS,
    ).hex()


def register_user(username, password):
    """
    Create a new account.

    Returns:
        (ok: bool, message: str)
    """

    initialize_database()

    username = (username or "").strip().lower()
    password = password or ""

    if not _USERNAME_RE.match(username):
        return False, "Username must be 3-30 characters: letters, numbers, _ or . only."

    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    salt = os.urandom(16)
    password_hash = _hash_password(password, salt)

    try:
        with _db() as connection:
            connection.execute(
                """
                INSERT INTO users (username, password_hash, salt, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    username,
                    password_hash,
                    salt.hex(),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
    except sqlite3.IntegrityError:
        return False, "That username is already taken."

    return True, "Account created. You can log in now."


def authenticate_user(username, password):
    """
    Check credentials.

    Returns:
        dict {"id", "username"} on success, otherwise None.
    """

    initialize_database()

    username = (username or "").strip().lower()
    password = password or ""

    with _db() as connection:
        row = connection.execute(
            "SELECT id, username, password_hash, salt FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if row is None:
        return None

    expected = row["password_hash"]
    actual = _hash_password(password, bytes.fromhex(row["salt"]))

    if not hmac.compare_digest(expected, actual):
        return None

    return {"id": row["id"], "username": row["username"]}


# ============================================================
# SAVE ANALYSIS (per user)
# ============================================================

def save_analysis(result, user_id):
    """Save a completed analysis for one user. Returns the new row ID."""

    initialize_database()

    analyzed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with _db() as connection:
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO analyses (
                user_id,
                analyzed_at,
                crop,
                crop_confidence,
                disease,
                disease_confidence,
                status,
                disease_model_available,
                price_available,
                price_date,
                min_price,
                max_price,
                avg_modal_price,
                predicted_price,
                predicted_price_date,
                price_trend,
                price_recommendation
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(user_id),
                analyzed_at,
                result.get("crop"),
                result.get("crop_confidence"),
                result.get("disease"),
                result.get("disease_confidence"),
                result.get("status"),
                int(bool(result.get("disease_model_available", False))),
                int(bool(result.get("price_available", False))),
                result.get("price_date"),
                result.get("min_price"),
                result.get("max_price"),
                result.get("avg_modal_price"),
                result.get("predicted_price"),
                result.get("predicted_price_date"),
                result.get("price_trend"),
                result.get("price_recommendation"),
            ),
        )

        return cursor.lastrowid


# ============================================================
# HISTORY (always filtered by user)
# ============================================================

def get_analysis_history(limit, user_id):
    """Most recent analyses of ONE user."""

    initialize_database()

    with _db() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM analyses
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(user_id), int(limit)),
        ).fetchall()

    return [dict(row) for row in rows]


def get_analysis_by_id(analysis_id, user_id):
    """One analysis, only if it belongs to this user."""

    initialize_database()

    with _db() as connection:
        row = connection.execute(
            "SELECT * FROM analyses WHERE id = ? AND user_id = ?",
            (int(analysis_id), int(user_id)),
        ).fetchone()

    return dict(row) if row else None


def get_analysis_count(user_id):
    """Number of saved analyses of ONE user."""

    initialize_database()

    with _db() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS total FROM analyses WHERE user_id = ?",
            (int(user_id),),
        ).fetchone()

    return int(row["total"])


def clear_analysis_history(user_id):
    """Delete only this user's saved analyses."""

    initialize_database()

    with _db() as connection:
        connection.execute(
            "DELETE FROM analyses WHERE user_id = ?",
            (int(user_id),),
        )


# ============================================================
# DATABASE TEST
# ============================================================

if __name__ == "__main__":

    initialize_database()

    print(f"Database: {DATABASE_FILE}")
    print(f"Exists  : {DATABASE_FILE.exists()}")
    print("Database initialization successful.")