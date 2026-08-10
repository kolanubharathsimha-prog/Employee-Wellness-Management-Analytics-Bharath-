# MoodMentor — Employee Wellness Management & Analytics

## Project Objective

One platform to understand how employees are really feeling — and act on it early.
Employees write journal entries, chat with a wellness assistant, or scan their face
for a quick mood check-in. A multilingual NLP pipeline detects sentiment and emotion
from that input, stores it per user, and surfaces trends, breakdowns, and
personalized recommendations on a live dashboard — for individuals and for managers
overseeing a team.

## MoodMentor Architecture

```
Text Input → Preprocessing → Emotion & Sentiment Analysis → Recommendation → Database → Report
```

| Layer      | Tech                                                       | Folder      |
|------------|-------------------------------------------------------------|-------------|
| Frontend   | Streamlit — auth, journal, chat, face scanner, dashboard     | `frontend/` |
| Backend    | FastAPI — auth, `/analyze`, `/analyze-text`, `/chat`, JWT     | `backend/`  |
| ML / NLP   | Language detection → translation → sentiment → emotion → LLM recommendation | `models/` |
| Database   | PostgreSQL — users, OTP codes, mood/journal history           | `backend/db.py` |

The Streamlit frontend calls the FastAPI backend over HTTP with a JWT issued at
login. The backend runs text through `models/nlp_pipeline.py`, writes the result to
Postgres, and returns sentiment/emotion scores + a recommendation, which the
frontend renders as charts and exports.

## Key Features

- Email/OTP signup, login, and password reset
- Journal entries and file uploads (CSV/TXT), analyzed for sentiment + emotion
- Multilingual input — auto-detected and translated before analysis
- Wellness Chat assistant with crisis-keyword safety check
- Live Face Scanner — mood check-in from a webcam snapshot
- Personalized, LLM-generated recommendations grounded in what was written
- Dashboard: mood distribution, mood trend, emotion breakdown, KPI tiles
- Date-range, emotion, and mood filters; text search over journal history
- CSV and PDF report export
- Manager view aggregating team-level wellness analytics

## Technology Stack

- **Frontend:** Streamlit, matplotlib
- **Backend:** FastAPI, Uvicorn
- **Database:** PostgreSQL (psycopg2)
- **Auth:** JWT (PyJWT), bcrypt password hashing
- **NLP/ML:** langdetect, Google Translate, spaCy (`xx_sent_ud_sm`), VADER,
  BERT (`bert-base-go-emotion`), Qwen2.5-0.5B-Instruct, GPT-OSS-20B (via Groq),
  DeepFace
- **Dev/deploy:** Google Colab + pyngrok (see `Milestone4/colab_notebook.ipynb`)

## Testing & Validation

See `Milestone4/testing.md` for the full checklist — auth, journal (text + file,
multilingual, empty/edge-case input), API error handling, dashboard filters/exports,
and recommendation relevance across all six emotion states. `Milestone4/README.md`
tracks which of the Milestone 4 objectives are implemented vs. still to verify.

## Application Highlights

- Emotion detection covers 6 states — Happy, Sad, Stress, Angry, Fear, Neutral —
  each with a consistent color and emoji used across every chart in the app
- Recommendations have no generic fallback: if the LLM call fails, nothing shows
  rather than showing irrelevant boilerplate advice
- A single emotion palette drives the Dashboard bar chart, Journal bar chart, and
  PDF report, so visuals stay consistent everywhere emotion data appears

## Running the Application

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download xx_sent_ud_sm
   ```
2. Copy `.env.example` to `.env` in `backend/` and fill in your Postgres, JWT,
   SMTP, and Groq credentials.
3. Run the backend:
   ```bash
   cd backend
   uvicorn backend:app --host 0.0.0.0 --port 8000
   ```
4. Run the frontend (separate terminal):
   ```bash
   cd frontend
   BACKEND_URL=http://localhost:8000 streamlit run app.py
   ```

> Originally developed and demoed from Google Colab using `pyngrok` to tunnel both
> services — see `Milestone4/colab_notebook.ipynb`. The steps above are the
> equivalent local/production setup; the source files are identical either way.

## Security

- Passwords are hashed with bcrypt — never stored in plain text
- Sessions use JWT (1-hour expiry), signed with `JWT_SECRET`
- Signup and password reset require a time-limited, single-use OTP sent by email
- Secrets (DB credentials, JWT secret, SMTP, Groq API key) are loaded from `.env`,
  which is git-ignored — nothing is hard-coded in source
- **Known dev-only gap:** the backend currently allows CORS from any origin
  (`allow_origins=["*"]` in `backend/backend.py`). Restrict this to your actual
  frontend origin before any real deployment.

## Project

Employee Wellness Management Analytics — MoodMentor
Final Milestone: Milestone 4 — Final Integration, Testing & Enhancement
