import os, smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
load_dotenv()

HOST, PORT = "smtp.gmail.com", 587
EMAIL = os.getenv("SMTP_EMAIL")
APP_PW = os.getenv("SMTP_APP_PASSWORD")

def send_otp(to_email, code, purpose):
    subject = "Your Verification Code" if purpose == "signup" else "Your Password Reset Code"
    msg = MIMEText(f"Your code is: {code}\nExpires in 10 minutes.")
    msg["From"], msg["To"], msg["Subject"] = EMAIL, to_email, subject
    try:
        with smtplib.SMTP(HOST, PORT, timeout=15) as s:
            s.starttls()
            s.login(EMAIL, APP_PW)
            s.sendmail(EMAIL, to_email, msg.as_string())
        return True, "sent"
    except Exception as e:
        return False, str(e)
