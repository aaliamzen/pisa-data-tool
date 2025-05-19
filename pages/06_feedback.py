import streamlit as st
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

# Add logo that persists across all pages
try:
    st.logo("assets/logo.png")  # Replace with the path to your logo file, e.g., "assets/logo.png"
except Exception as e:
    st.error(f"Failed to load logo: {e}")

# Streamlit app configuration
st.set_page_config(page_title="Feedback - PISA Data Exploration Tool", layout="wide")

# Title
st.title("Feedback and Recommendations")

# Email setup
def send_feedback_email(feedback):
    try:
        # Load email credentials from Streamlit secrets
        sender_email = st.secrets["EMAIL"]["sender_email"]
        sender_password = st.secrets["EMAIL"]["sender_password"]
        recipient_email = st.secrets["EMAIL"]["recipient_email"]

        # Create email message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = "PISA Tool Feedback Submission"
        body = f"Timestamp: {timestamp}\n\nFeedback:\n{feedback}"
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = recipient_email

        # Send email via Gmail SMTP
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Error sending feedback email: {str(e)}")
        return False

# Feedback Section
st.header("Provide Feedback")
feedback = st.text_area("We’d love to hear your feedback! Please share your thoughts, suggestions, or issues below:", height=150)

if 'feedback_list' not in st.session_state:
    st.session_state.feedback_list = []

if st.button("Submit Feedback"):
    if feedback.strip():
        # Send feedback via email
        if send_feedback_email(feedback):
            st.session_state.feedback_list.append(feedback)
            st.success("Thank you for your feedback! It has been sent.")
        else:
            st.warning("Feedback submitted locally, but there was an issue sending it via email.")
    else:
        st.warning("Please enter some feedback before submitting.")

# Display submitted feedback (optional)
if st.session_state.feedback_list:
    st.subheader("Submitted Feedback")
    for idx, fb in enumerate(st.session_state.feedback_list, 1):
        st.write(f"Feedback {idx}: {fb}")