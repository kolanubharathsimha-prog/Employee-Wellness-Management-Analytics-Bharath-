import os, jwt, bcrypt, random, string
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from db import cursor
load_dotenv()

SECRET = os.getenv("JWT_SECRET")

def hash_pw(pw): return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
def check_pw(pw, h): return bcrypt.checkpw(pw.encode(), h.encode())
def make_token(user):
    payload = {"id": user["id"], "username": user["username"], "email": user["email"],
               "role": user.get("role", "employee"),
               "exp": datetime.now(timezone.utc) + timedelta(hours=1)}
    return jwt.encode(payload, SECRET, algorithm="HS256")

def read_token(token):
    try: return jwt.decode(token, SECRET, algorithms=["HS256"])
    except jwt.PyJWTError: return None

def get_user(email):
    with cursor() as cur:
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        return cur.fetchone()

def username_taken(username):
    with cursor() as cur:
        cur.execute("SELECT 1 FROM users WHERE username=%s", (username,))
        return cur.fetchone() is not None

def create_user(username, email, pw, role="employee"):
    with cursor(commit=True) as cur:
        cur.execute("INSERT INTO users (username,email,password_hash,role) VALUES (%s,%s,%s,%s)",
                    (username, email, hash_pw(pw), role))

def verify_user(email):
    with cursor(commit=True) as cur:
        cur.execute("UPDATE users SET is_verified=TRUE WHERE email=%s", (email,))

def set_password(email, pw):
    with cursor(commit=True) as cur:
        cur.execute("UPDATE users SET password_hash=%s WHERE email=%s", (hash_pw(pw), email))

def new_otp():
    return "".join(random.choices(string.digits, k=6))

def save_otp(email, code, purpose):
    exp = datetime.now(timezone.utc) + timedelta(minutes=10)
    with cursor(commit=True) as cur:
        cur.execute("UPDATE otp_codes SET used=TRUE WHERE email=%s AND purpose=%s", (email, purpose))
        cur.execute("INSERT INTO otp_codes (email,code,purpose,expires_at) VALUES (%s,%s,%s,%s)",
                    (email, code, purpose, exp))

def check_otp(email, code, purpose):
    with cursor(commit=True) as cur:
        cur.execute("""SELECT * FROM otp_codes WHERE email=%s AND purpose=%s AND used=FALSE
                       ORDER BY id DESC LIMIT 1""", (email, purpose))
        row = cur.fetchone()
        if not row or row["code"] != code:
            return False
        now = datetime.now(row["expires_at"].tzinfo) if row["expires_at"].tzinfo else datetime.now()
        if now > row["expires_at"]:
            return False
        cur.execute("UPDATE otp_codes SET used=TRUE WHERE id=%s", (row["id"],))
        return True
