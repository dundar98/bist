"""
Email Notification Service.

Sends trading signal notifications via email using SMTP.
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EmailConfig:
    """Email configuration."""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = ""
    sender_password: str = ""  # App password for Gmail
    recipient_emails: List[str] = None
    
    def __post_init__(self):
        if self.recipient_emails is None:
            self.recipient_emails = []


class EmailNotifier:
    """
    Sends email notifications for trading signals.
    
    Supports Gmail, Outlook, and custom SMTP servers.
    
    For Gmail, you need to:
    1. Enable 2-Factor Authentication
    2. Create an App Password: https://myaccount.google.com/apppasswords
    3. Use the app password, not your regular password
    """
    
    def __init__(self, config: EmailConfig):
        """
        Initialize email notifier.
        
        Args:
            config: Email configuration
        """
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate email configuration."""
        if not self.config.sender_email:
            logger.warning("Sender email not configured")
        if not self.config.sender_password:
            logger.warning("Sender password not configured")
        if not self.config.recipient_emails:
            logger.warning("No recipient emails configured")
    
    def send(
        self,
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        recipients: Optional[List[str]] = None,
    ) -> bool:
        """
        Send an email.
        
        Args:
            subject: Email subject
            body: Plain text body
            html_body: Optional HTML body
            recipients: Optional override for recipients
            
        Returns:
            True if sent successfully
        """
        recipients = recipients or self.config.recipient_emails
        
        if not recipients:
            logger.error("No recipients configured")
            return False
        
        if not self.config.sender_email or not self.config.sender_password:
            logger.error("Email credentials not configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.config.sender_email
            msg["To"] = ", ".join(recipients)
            
            # Add plain text
            msg.attach(MIMEText(body, "plain", "utf-8"))
            
            # Add HTML if provided
            if html_body:
                msg.attach(MIMEText(html_body, "html", "utf-8"))
            
            # Send email
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.sender_email, self.config.sender_password)
                server.sendmail(
                    self.config.sender_email,
                    recipients,
                    msg.as_string()
                )
            
            logger.info(f"Email sent to {len(recipients)} recipients")
            return True
            
        except smtplib.SMTPAuthenticationError:
            logger.error(
                "SMTP authentication failed. "
                "For Gmail, use an App Password: https://myaccount.google.com/apppasswords"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def send_signal_report(
        self,
        report_text: str,
        scan_date: str,
        buy_count: int,
        sell_count: int,
    ) -> bool:
        """
        Send a trading signal report.
        
        Args:
            report_text: Formatted report text
            scan_date: Date of scan
            buy_count: Number of buy signals
            sell_count: Number of sell signals
            
        Returns:
            True if sent successfully
        """
        # Create subject with emoji
        if buy_count > 3:
            emoji = "🔥"
        elif buy_count > 0:
            emoji = "📈"
        else:
            emoji = "📊"
        
        subject = f"{emoji} BIST100 Günlük Sinyal Raporu - {scan_date} ({buy_count} AL, {sell_count} SAT)"
        
        # Create HTML version
        html_body = self._create_html_report(report_text, buy_count, sell_count)
        
        return self.send(
            subject=subject,
            body=report_text,
            html_body=html_body,
        )
    
    def _create_html_report(
        self,
        text_report: str,
        buy_count: int,
        sell_count: int,
    ) -> str:
        """Create HTML version of report."""
        # Convert text to HTML-safe
        html_content = text_report.replace("\n", "<br>")
        html_content = html_content.replace("🟢", "<span style='color: green;'>🟢</span>")
        html_content = html_content.replace("🔴", "<span style='color: red;'>🔴</span>")
        html_content = html_content.replace("✅", "<span style='color: green;'>✅</span>")
        html_content = html_content.replace("❌", "<span style='color: red;'>❌</span>")
        html_content = html_content.replace("🔥", "<span style='color: orange;'>🔥</span>")
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: 'Consolas', 'Monaco', monospace;
                    background-color: #1a1a2e;
                    color: #eee;
                    padding: 20px;
                    line-height: 1.6;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    text-align: center;
                }}
                .summary {{
                    display: flex;
                    justify-content: space-around;
                    margin: 20px 0;
                }}
                .stat {{
                    text-align: center;
                    padding: 15px;
                    background: #16213e;
                    border-radius: 8px;
                    min-width: 100px;
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                }}
                .buy {{ color: #00ff88; }}
                .sell {{ color: #ff4444; }}
                .content {{
                    background: #16213e;
                    padding: 20px;
                    border-radius: 10px;
                    white-space: pre-wrap;
                }}
                .footer {{
                    margin-top: 20px;
                    text-align: center;
                    color: #888;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 BIST100 Trading Sinyalleri</h1>
            </div>
            
            <div class="summary">
                <div class="stat">
                    <div class="stat-value buy">{buy_count}</div>
                    <div>AL Sinyali</div>
                </div>
                <div class="stat">
                    <div class="stat-value sell">{sell_count}</div>
                    <div>SAT Sinyali</div>
                </div>
            </div>
            
            <div class="content">
                {html_content}
            </div>
            
            <div class="footer">
                Bu e-posta otomatik olarak BIST100 Trading System tarafından gönderilmiştir.<br>
                ⚠️ Yatırım tavsiyesi değildir.
            </div>
        </body>
        </html>
        """


def create_email_config_from_env() -> EmailConfig:
    """
    Create email config from environment variables.
    
    Expected environment variables:
    - BIST_SMTP_SERVER (default: smtp.gmail.com)
    - BIST_SMTP_PORT (default: 587)
    - BIST_EMAIL_SENDER
    - BIST_EMAIL_PASSWORD
    - BIST_EMAIL_RECIPIENTS (comma-separated)
    """
    import os
    
    recipients_str = os.getenv("BIST_EMAIL_RECIPIENTS", "")
    recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]
    
    return EmailConfig(
        smtp_server=os.getenv("BIST_SMTP_SERVER", "smtp.gmail.com"),
        smtp_port=int(os.getenv("BIST_SMTP_PORT", "587")),
        sender_email=os.getenv("BIST_EMAIL_SENDER", ""),
        sender_password=os.getenv("BIST_EMAIL_PASSWORD", ""),
        recipient_emails=recipients,
    )
