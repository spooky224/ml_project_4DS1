import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys

# ------------------------------

# Configuration

# ------------------------------

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "[abderrahmen.oueslati@esprit.tn](mailto:abderrahmen.oueslati@esprit.tn)"
SENDER_PASSWORD = "221JMT6246@"  # Use App Password
RECEIVER_EMAIL = "[spookys392@gmail.com](mailto:spookys392@gmail.com)"

def send_email(subject, body):
try:
msg = MIMEMultipart()
msg['From'] = SENDER_EMAIL
msg['To'] = RECEIVER_EMAIL
msg['Subject'] = subject
msg.attach(MIMEText(body, 'plain'))

```
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_PASSWORD)
    server.send_message(msg)
    server.quit()
    print("✅ Email notification sent successfully!")
except Exception as e:
    print(f"❌ Failed to send email: {e}")
```

if **name** == "**main**":
subject = sys.argv[1] if len(sys.argv) > 1 else "MLOps Pipeline Notification"
body = sys.argv[2] if len(sys.argv) > 2 else "Pipeline execution finished successfully."
send_email(subject, body)
