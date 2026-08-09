import os, io, jwt, csv, logging
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from nlp_pipeline import process_employee_feedback, wellness_chat_reply

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

load_dotenv()

SECRET = os.getenv("JWT_SECRET")
if not SECRET:
    logger.error("❌ JWT_SECRET not found in environment variables!")
else:
    logger.info(f"✅ JWT_SECRET loaded (length: {len(SECRET)})")

app = FastAPI(title="Upload API")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                    allow_methods=["*"], allow_headers=["*"])

def get_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing token")
    token = authorization.split(" ", 1)[1]
    try:
        return jwt.decode(token, SECRET, algorithms=["HS256"])
    except jwt.PyJWTError as e:
        logger.warning(f"Auth failed: {str(e)}")
        raise HTTPException(401, f"Invalid or expired token: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload(file: UploadFile = File(...), authorization: str = Header(None)):
    user = get_user(authorization)

    name = file.filename or ""
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if ext not in ("csv", "txt"):
        raise HTTPException(400, "Only .csv or .txt files are allowed.")

    raw = await file.read()
    max_bytes = 5 * 1024 * 1024  # 5 MB cap
    if len(raw) > max_bytes:
        raise HTTPException(400, "File too large (max 5 MB).")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "File must be UTF-8 text.")

    lines = text.splitlines()
    row_count = len(lines)
    preview_lines = lines[:20]

    columns = None
    preview_rows = None
    if ext == "csv":
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if rows:
            columns = rows[0]
            preview_rows = rows[1:21]
            row_count = max(len(rows) - 1, 0)

    return {
        "filename": name,
        "type": ext,
        "uploaded_by": user["username"],
        "row_count": row_count,
        "columns": columns,
        "preview_rows": preview_rows,
        "preview_lines": None if ext == "csv" else preview_lines,
    }

def _extract_text_blob(raw: bytes, ext: str, column: str | None) -> tuple[str, str | None]:
    text = raw.decode("utf-8")
    if ext == "txt":
        return text.strip(), None
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows: raise HTTPException(400, "CSV file has no rows.")
    header = rows[0]
    data_rows = rows[1:]
    if not data_rows: raise HTTPException(400, "CSV file has a header but no data rows.")
    col_index = header.index(column) if column and column in header else len(header) - 1
    values = [row[col_index] for row in data_rows if len(row) > col_index and row[col_index].strip()]
    blob = " ".join(values).strip()
    if not blob: raise HTTPException(400, f"Column '{header[col_index]}' has no readable text.")
    return blob, header[col_index]

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), column: str = Form(None), authorization: str = Header(None)):
    get_user(authorization)
    name = file.filename or ""
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if ext not in ("csv", "txt"): raise HTTPException(400, "Only .csv or .txt files are allowed.")
    raw = await file.read()
    try:
        text_blob, used_column = _extract_text_blob(raw, ext, column)
    except UnicodeDecodeError: raise HTTPException(400, "File must be UTF-8 text.")
    results = process_employee_feedback(text_blob)
    results.update({"filename": name, "file_type": ext.upper(), "used_column": used_column})
    return results

class TextIn(BaseModel):
    text: str

@app.post("/analyze-text")
async def analyze_text(payload: TextIn, authorization: str = Header(None)):
    get_user(authorization)
    text_blob = payload.text.strip()
    if not text_blob: raise HTTPException(400, "Text cannot be empty.")
    results = process_employee_feedback(text_blob)
    results.update({"filename": None, "file_type": "TEXT", "used_column": None})
    return results

class ChatTurn(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = []

@app.post("/chat")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    get_user(authorization)
    message = payload.message.strip()
    if not message: raise HTTPException(400, "Message cannot be empty.")
    history = [turn.dict() for turn in payload.history]
    return wellness_chat_reply(message, history=history)
