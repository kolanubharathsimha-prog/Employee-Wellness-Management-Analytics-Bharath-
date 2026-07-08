# User Authentication and Session Management Infrastructure

## Project Objective

- **Secure Auth Flows:** User registration, login, and session tracking via a clean Streamlit interface.
- **Modern Cryptography:** Cleartext passwords are never saved; they are salted and hashed using bcrypt before hitting the database.
- **Smart Sessions:** Implements JSON Web Tokens (JWT) for secure, time-expiring state tracking.
- **Cloud Persistence:** Complete data storage via a serverless Neon PostgreSQL database.
- **Password Recoveries:** Secure password resets handled via SMTP email utilizing One-Time Passwords (OTP).

## Technologies Used

- **Frontend:** Streamlit
- **Database:** Neon PostgreSQL (Cloud/Serverless)
- **Security:** bcrypt (Hashing) & PyJWT (Tokenization)
- **Tunneling:** pyngrok (Exposes local ports to public URLs)

## Google Colab Setup Instructions

1. **Set Up Environment Secrets:**
   - Add `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` (Take these from our Neon console)
   - `JWT_SECRET` (Secure string for token signing)
   - `NGROK_AUTHTOKEN` (From our personal Ngrok dashboard)
   - `SMTP_EMAIL` and `SMTP_APP_PASSWORD` (our Email credentials for sending OTPs)

2. **Install Dependencies:**
   ```bash
   !pip install -q streamlit psycopg2-binary PyJWT bcrypt python-dotenv email-validator pyngrok
   ```
