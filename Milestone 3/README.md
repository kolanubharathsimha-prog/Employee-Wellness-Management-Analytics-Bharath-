# Milestone 3: Emotion Detection & Journal Analytics

## Project Objective

This milestone extends the Employee Wellness Management Analytics platform with automated
emotion detection and journal analytics. Employees write a daily journal entry (or upload a
CSV/TXT file of feedback), and the system automatically detects the dominant emotion with a
confidence score, computes sentiment scores, and stores the results so employees can track
their wellness trends over time.

## Model Used

- **Emotion detection:** `bhadresh-savani/bert-base-go-emotion`, a transformer-based text
  classification model, run through Hugging Face's `transformers` pipeline.
- **Sentiment analysis:** VADER (`vaderSentiment`), a lexicon-based sentiment scorer.
- **Language detection:** `langdetect`.
- **Translation:** `deep-translator` (Google Translate backend), used to translate non-English
  entries to English before scoring.
- **Preprocessing:** spaCy's multilingual pipeline (`xx_sent_ud_sm`) for tokenization and
  sentence splitting.

## Emotion Detection Pipeline

Each journal entry passes through the full multilingual NLP pipeline before emotion
prediction:

1. Normalize the text and extract any emoji.
2. Clean the text — remove URLs, emails, mentions, and hashtags.
3. Detect the language.
4. Tokenize and split into sentences with spaCy.
5. Remove stopwords for the detected language.
6. Translate the cleaned text to English.
7. Lemmatize the translated text.
8. Pass the translated text into the emotion classification model, which scores it against
   GoEmotions' fine-grained labels. These are grouped into six app-level emotions — Happy,
   Sad, Stress, Angry, Fear, and Neutral — by combining the underlying labels that map to
   each group.
9. The emotion with the highest combined score becomes the entry's predicted emotion.

## Confidence Score Calculation

The raw scores from the emotion model are grouped into the six app-level emotions, then
normalized so all six add up to 1. The confidence score shown for the predicted emotion is
its share of that total — essentially, how strongly the model favored that emotion relative
to the other five. All six scores are displayed as a bar chart so employees can see the full
spread, not just the winning label.

## Sentiment Analysis

Sentiment is computed on the translated English text using VADER, which produces four scores:
a positive score, a negative score, a neutral score, and a compound score (a single value
from -1 to +1 summarizing overall sentiment). The compound score is used to assign a simple
label — Positive, Negative, or Neutral — based on standard VADER thresholds. All four scores
are shown to the employee; the compound score is the one saved to the database, since it's
used for charting mood trends over time.

## Database Schema

Journal entries are stored in a `mood_logs` table, alongside manually picked moods, so a
person's full history lives in one place. Each row records who the entry belongs to, the date
and timestamp, the predicted sentiment and emotion labels, the compound sentiment score, the
full journal text, and whether the entry came from a manual mood pick or from the NLP
pipeline. This lets the calendar, journal history, and dashboard all pull from a single
consistent source, and lets journal-based sentiment be mapped onto the same mood scale used
for manual check-ins.

## API Endpoints

- A text-analysis endpoint that runs the full NLP pipeline on a single journal entry and
  returns the language, sentiment scores, emotion scores, and final labels.
- A file-analysis endpoint that runs the same pipeline on a column of an uploaded CSV or TXT
  file.
- A chat endpoint that powers the Wellness Chat feature.
- A transcription endpoint that converts recorded voice messages into text for the Wellness
  Chat's microphone input.
- A health-check endpoint used to confirm the backend is running.

All endpoints besides the health check require an authenticated user.

## Sample Input & Output

**Input:** "Finished the big client presentation today and it went really well! Feeling
relieved and proud of the team."

**Output:** The system detects the entry as English, scores it as strongly Positive with a
high compound sentiment score, and predicts Happy as the dominant emotion with high
confidence, alongside much lower scores for the other five emotions. The entry, its sentiment
label, its emotion label, and its compound score are saved to the employee's journal history,
and the employee sees a summary along with a bar chart of all six emotion confidence scores.

## Observations

- Translating non-English entries to English before scoring keeps a single consistent model
  in the loop, but translation quality — and therefore emotion/sentiment accuracy — varies by
  language and can shift nuance in short or idiomatic entries.
- Confidence scores are well-separated for clearly emotional entries but flatten out toward
  Neutral for short, factual, or mixed-sentiment ones, which is expected given how the scores
  are combined and normalized.
- Journal-based sentiment only ever lands on Happy, Normal, or Sad on the shared mood scale —
  Amazing and Angry are reserved for manual emoji check-ins — which is a deliberate
  simplification rather than a limitation of the model.
- Models are loaded once and reused, so the first analysis after the backend starts is
  noticeably slower than every analysis after it.

## Submission

Project: Employee Wellness Management Analytics — Milestone 3 (Emotion Detection & Journal
Analytics), developed and tested in Google Colaboratory.
