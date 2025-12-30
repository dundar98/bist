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
        """Create HTML version of report with accessible design."""
        from datetime import date
        
        # Parse the text report to extract signals
        lines = text_report.split('\n')
        
        buy_signals_data = []
        in_buy_section = False
        
        current_signal = None

        for line in lines:
            line = line.strip()
            if "TÜM AL SİNYALLERİ" in line or "EN GÜÇLÜ AL SİNYALLERİ" in line:
                in_buy_section = True
                continue
            elif "SAT SİNYALLERİ" in line or "DİKKAT" in line or line.startswith("="):
                if current_signal:
                    buy_signals_data.append(current_signal)
                    current_signal = None
                in_buy_section = False
                continue
            
            if in_buy_section:
                if '|' in line and (line.startswith('🔥') or line.startswith('✅') or line.startswith('⚠️')):
                    if current_signal:
                        buy_signals_data.append(current_signal)
                    
                    # Row 1: symbol | Sinyal | Olasılık | Fiyat
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 4:
                        symbol = parts[0].replace('🔥', '').replace('✅', '').replace('⚠️', '').strip()
                        prob = parts[2].replace('Olasılık:', '').strip()
                        price = parts[3].replace('Fiyat:', '').strip()
                        
                        current_signal = {
                            "symbol": symbol,
                            "prob": prob,
                            "price": price,
                            "target": "-",
                            "horizon": "-",
                            "history": ""
                        }
                elif '🎯 Hedef:' in line and current_signal:
                    # Row 2: Target | Horizon | History
                    parts = [p.strip() for p in line.split('|')]
                    for p in parts:
                        if 'Hedef:' in p:
                            current_signal["target"] = p.split('Hedef:')[1].strip()
                        elif 'Vade:' in p:
                            current_signal["horizon"] = p.split('Vade:')[1].strip()
                        elif '📜' in p:
                            current_signal["history"] = p.split('📜')[1].strip()
        
        if current_signal:
            buy_signals_data.append(current_signal)

        # Build buy signals HTML
        buy_signals_html = ""
        for sig in buy_signals_data:
            buy_signals_html += f'''
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #e0e0e0; font-weight: bold; color: #1a73e8;">{sig['symbol']}</td>
                <td style="padding: 10px; border-bottom: 1px solid #e0e0e0; color: #0d652d; font-weight: bold;">{sig['prob']}</td>
                <td style="padding: 10px; border-bottom: 1px solid #e0e0e0;">{sig['price']}</td>
                <td style="padding: 10px; border-bottom: 1px solid #e0e0e0; color: #c2185b; font-weight: bold;">{sig['target']}</td>
                <td style="padding: 10px; border-bottom: 1px solid #e0e0e0;">{sig['horizon']}</td>
                <td style="padding: 10px; border-bottom: 1px solid #e0e0e0; font-size: 11px; color: #666;">{sig['history']}</td>
            </tr>'''

        # Build buy signals table
        buy_table = ""
        if buy_signals_html:
            buy_table = f'''
            <div style="margin: 20px 0;">
                <h2 style="color: #0d652d; font-size: 18px; margin-bottom: 15px; border-left: 4px solid #0d652d; padding-left: 10px;">
                    ✅ AL SİNYALLERİ (DETAYLI)
                </h2>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; font-size: 13px;">
                        <thead>
                            <tr style="background: #f1f8e9;">
                                <th style="padding: 10px; text-align: left; font-weight: 600; color: #333;">Hisse</th>
                                <th style="padding: 10px; text-align: left; font-weight: 600; color: #333;">Olasılık</th>
                                <th style="padding: 10px; text-align: left; font-weight: 600; color: #333;">Fiyat</th>
                                <th style="padding: 10px; text-align: left; font-weight: 600; color: #333;">Hedef</th>
                                <th style="padding: 10px; text-align: left; font-weight: 600; color: #333;">Vade</th>
                                <th style="padding: 10px; text-align: left; font-weight: 600; color: #333;">Geçmiş</th>
                            </tr>
                        </thead>
                        <tbody>
                            {buy_signals_html}
                        </tbody>
                    </table>
                </div>
            </div>'''
        
        # Simple sell list if needed (omitted for brevity as user prioritized BUY)
        sell_table = ""
        
        return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="color-scheme" content="light">
    <meta name="supported-color-schemes" content="light">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f5f5f5; color: #333;">
    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%); padding: 25px; border-radius: 12px 12px 0 0; text-align: center;">
            <h1 style="margin: 0; color: #fff; font-size: 24px; font-weight: 600;">
                📊 BIST100 Günlük Sinyal Raporu
            </h1>
            <p style="margin: 10px 0 0 0; color: rgba(255,255,255,0.9); font-size: 14px;">
                {date.today().strftime('%d %B %Y')}
            </p>
        </div>
        
        <!-- Summary Cards -->
        <div style="background: #fff; padding: 20px; border-bottom: 1px solid #e0e0e0;">
            <table style="width: 100%;" cellpadding="0" cellspacing="0">
                <tr>
                    <td style="width: 50%; text-align: center; padding: 15px;">
                        <div style="background: #e8f5e9; border-radius: 8px; padding: 20px;">
                            <div style="font-size: 36px; font-weight: bold; color: #0d652d;">{buy_count}</div>
                            <div style="font-size: 14px; color: #333; margin-top: 5px;">AL Sinyali</div>
                        </div>
                    </td>
                    <td style="width: 50%; text-align: center; padding: 15px;">
                        <div style="background: #ffebee; border-radius: 8px; padding: 20px;">
                            <div style="font-size: 36px; font-weight: bold; color: #d32f2f;">{sell_count}</div>
                            <div style="font-size: 14px; color: #333; margin-top: 5px;">SAT Sinyali</div>
                        </div>
                    </td>
                </tr>
            </table>
        </div>
        
        <!-- Signal Tables -->
        <div style="background: #fff; padding: 20px;">
            {buy_table}
            {sell_table}
            
            {f'<p style="color: #666; font-size: 14px; text-align: center; margin-top: 20px;">📈 Bugün {buy_count + sell_count} aktif sinyal tespit edildi.</p>' if buy_count + sell_count > 0 else '<p style="color: #666; font-size: 14px; text-align: center; margin-top: 20px;">ℹ️ Bugün güçlü sinyal tespit edilmedi.</p>'}
        </div>
        
        <!-- Footer -->
        <div style="background: #fafafa; padding: 20px; border-radius: 0 0 12px 12px; text-align: center; border-top: 1px solid #e0e0e0;">
            <p style="margin: 0 0 10px 0; color: #666; font-size: 12px;">
                ⚠️ <strong>UYARI:</strong> Bu sinyaller yatırım tavsiyesi değildir.
            </p>
            <p style="margin: 0; color: #999; font-size: 11px;">
                BIST100 Deep Learning Trading System tarafından otomatik olarak oluşturulmuştur.
            </p>
        </div>
    </div>
</body>
</html>'''


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
