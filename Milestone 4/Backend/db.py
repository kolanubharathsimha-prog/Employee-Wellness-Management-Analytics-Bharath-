import os, psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from dotenv import load_dotenv
load_dotenv()

CFG = dict(host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", "5432"),
           dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
           password=os.getenv("DB_PASSWORD"), sslmode="require")

@contextmanager
def cursor(commit=False):
    conn = psycopg2.connect(**CFG)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        yield cur
        if commit: conn.commit()
    finally:
        cur.close(); conn.close()

def init_db():
    with cursor(commit=True) as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY, username VARCHAR(50) UNIQUE, email VARCHAR(255) UNIQUE,
            password_hash VARCHAR(255), is_verified BOOLEAN DEFAULT FALSE,
            role VARCHAR(20) NOT NULL DEFAULT 'employee')""")
        # Safe to run repeatedly: adds the column if this table already existed pre-role.
        cur.execute("""ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(20) NOT NULL DEFAULT 'employee'""")
        cur.execute("""CREATE TABLE IF NOT EXISTS otp_codes (
            id SERIAL PRIMARY KEY, email VARCHAR(255), code VARCHAR(6),
            purpose VARCHAR(20), expires_at TIMESTAMP, used BOOLEAN DEFAULT FALSE)""")

        # ---- One row per mood entry (manual pick OR journal/NLP submission) ----
        # All entries — manual emoji picks and journal/file submissions — go into
        # this ONE table, tied to the same user_id, so a person's whole history
        # (calendar, journal log, dashboard) always comes from one place.
        # `mood_date` is the calendar day the entry belongs to (defaults to submission day).
        # `created_at` is the full DATE+TIME the row was written — this is the
        # "emoji/journal saved with time" piece: every row already carries a timestamp.
        # `sentiment` stores one of the 5 labels: Amazing, Happy, Normal, Sad, Angry.
        # `compound_score` is VADER's -1..1 score, handy for charts (NULL for manual picks).
        # `source` marks how the row was created: 'manual' (emoji picker) or 'nlp' (journal/upload).
        cur.execute("""CREATE TABLE IF NOT EXISTS mood_logs (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            mood_date DATE NOT NULL DEFAULT CURRENT_DATE,
            sentiment VARCHAR(20),
            emotion VARCHAR(30),
            compound_score REAL,
            journal_text TEXT,
            source VARCHAR(10) NOT NULL DEFAULT 'manual',
            created_at TIMESTAMP NOT NULL DEFAULT NOW())""")
        cur.execute("""ALTER TABLE mood_logs ADD COLUMN IF NOT EXISTS source VARCHAR(10) NOT NULL DEFAULT 'manual'""")
        cur.execute("""CREATE INDEX IF NOT EXISTS idx_mood_logs_user_date
            ON mood_logs(user_id, mood_date)""")



# ---- The 5-point mood scale used everywhere (picker, calendar, dashboard, reports) ----
MOOD_LABELS = ["Amazing", "Happy", "Normal", "Sad", "Angry"]

# Maps the NLP pipeline's 3-way sentiment onto the closest of the 5 labels,
# so journal/file analysis results can still be plotted on the same scale.
NLP_TO_MOOD_LABEL = {
    "Positive": "Happy",
    "Neutral": "Normal",
    "Negative": "Sad",
}


# ---- Mood log helpers ----

def save_manual_mood(user_id, mood_label):
    """Employee taps an emoji on the 'How Do You Feel?' picker — saves
    immediately (with the current date+time via created_at), no NLP involved."""
    with cursor(commit=True) as cur:
        cur.execute(
            """INSERT INTO mood_logs (user_id, sentiment, source)
               VALUES (%s, %s, 'manual')""",
            (user_id, mood_label),
        )

def save_mood_log(user_id, sentiment, emotion, compound_score, journal_text):
    """Call this right after the NLP pipeline returns a result, so every
    journal entry (typed or uploaded) leaves a row — with date+time and the
    full journal text — for the calendar/journal-history/dashboard/report.
    `sentiment` here is the pipeline's Positive/Neutral/Negative label; it's
    mapped onto the 5-point scale so it plots consistently everywhere."""
    mood_label = NLP_TO_MOOD_LABEL.get(sentiment, "Normal")
    with cursor(commit=True) as cur:
        cur.execute(
            """INSERT INTO mood_logs (user_id, sentiment, emotion, compound_score, journal_text, source)
               VALUES (%s, %s, %s, %s, %s, 'nlp')""",
            (user_id, mood_label, emotion, compound_score, journal_text),
        )

def get_mood_logs_for_month(user_id, year, month):
    """Returns one row per day for a given user/month, latest entry per day.
    Used by the Home tab's calendar grid."""
    with cursor() as cur:
        cur.execute(
            """SELECT DISTINCT ON (mood_date) mood_date, sentiment, emotion, compound_score, created_at
               FROM mood_logs
               WHERE user_id = %s
                 AND EXTRACT(YEAR FROM mood_date) = %s
                 AND EXTRACT(MONTH FROM mood_date) = %s
               ORDER BY mood_date, created_at DESC""",
            (user_id, year, month),
        )
        return cur.fetchall()

def get_user_mood_history(user_id, limit=200):
    with cursor() as cur:
        cur.execute(
            """SELECT mood_date, sentiment, emotion, compound_score, journal_text, source, created_at
               FROM mood_logs
               WHERE user_id = %s
               ORDER BY created_at DESC
               LIMIT %s""",
            (user_id, limit),
        )
        return cur.fetchall()

def get_all_employee_mood_logs(limit_days=30):

    with cursor() as cur:
        cur.execute(
            """SELECT u.username, u.email, m.mood_date, m.sentiment, m.emotion, m.compound_score, m.created_at
               FROM mood_logs m
               JOIN users u ON u.id = m.user_id
               WHERE u.role = 'employee'
                 AND m.mood_date >= CURRENT_DATE - (%s || ' days')::interval
               ORDER BY m.mood_date DESC, u.username""",
            (limit_days,),
        )
        return cur.fetchall()

def get_latest_mood_per_employee():
    """For managers: each employee's single most recent mood entry."""
    with cursor() as cur:
        cur.execute(
            """SELECT DISTINCT ON (u.id) u.username, u.email, m.mood_date, m.sentiment, m.emotion, m.created_at
               FROM users u
               JOIN mood_logs m ON m.user_id = u.id
               WHERE u.role = 'employee'
               ORDER BY u.id, m.created_at DESC"""
        )
        return cur.fetchall()
