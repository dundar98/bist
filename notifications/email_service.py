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
        
        # Find buy signals section
        buy_signals_html = ""
        sell_signals_html = ""
        in_buy_section = False
        in_sell_section = False
        
        for line in lines:
            if "EN GÜÇLÜ AL SİNYALLERİ" in line:
                in_buy_section = True
                in_sell_section = False
                continue
            elif "SAT SİNYALLERİ" in line:
                in_buy_section = False
                in_sell_section = True
                continue
            elif "hisse taranamadı" in line or "DİKKAT" in line:
                in_buy_section = False
                in_sell_section = False
                continue
            
            if in_buy_section and ('THYAO' in line or 'GARAN' in line or 'AKBNK' in line or 'Olasılık' in line):
                if '|' in line:
                    # Parse: 🔥 THYAO  | Olasılık: 78.5% | Fiyat: 245.30 TL | RSI: 42
                    parts = line.split('|')
                    if len(parts) >= 3:
                        symbol = parts[0].replace('🔥', '').replace('✅', '').strip()
                        prob = parts[1].replace('Olasılık:', '').strip()
                        price = parts[2].replace('Fiyat:', '').strip()
                        rsi = parts[3].replace('RSI:', '').strip() if len(parts) > 3 else '-'
                        
                        buy_signals_html += f'''
                        <tr>
                            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0; font-weight: bold; color: #1a73e8;">{symbol}</td>
                            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0; color: #0d652d; font-weight: bold;">{prob}</td>
                            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">{price}</td>
                            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">{rsi}</td>
                        </tr>'''
            
            if in_sell_section and '|' in line:
                parts = line.split('|')
                if len(parts) >= 2:
                    symbol = parts[0].replace('❌', '').strip()
                    prob = parts[1].replace('Olasılık:', '').strip()
                    price = parts[2].replace('Fiyat:', '').strip() if len(parts) > 2 else '-'
                    
                    sell_signals_html += f'''
                    <tr>
                        <td style="padding: 12px; border-bottom: 1px solid #e0e0e0; font-weight: bold; color: #d32f2f;">{symbol}</td>
                        <td style="padding: 12px; border-bottom: 1px solid #e0e0e0; color: #d32f2f;">{prob}</td>
                        <td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">{price}</td>
                    </tr>'''
        
        # Build buy signals table
        buy_table = ""
        if buy_signals_html:
            buy_table = f'''
            <div style="margin: 20px 0;">
                <h2 style="color: #0d652d; font-size: 18px; margin-bottom: 15px; border-left: 4px solid #0d652d; padding-left: 10px;">
                    ✅ AL SİNYALLERİ
                </h2>
                <table style="width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <thead>
                        <tr style="background: #e8f5e9;">
                            <th style="padding: 12px; text-align: left; font-weight: 600; color: #333;">Hisse</th>
                            <th style="padding: 12px; text-align: left; font-weight: 600; color: #333;">Olasılık</th>
                            <th style="padding: 12px; text-align: left; font-weight: 600; color: #333;">Fiyat</th>
                            <th style="padding: 12px; text-align: left; font-weight: 600; color: #333;">RSI</th>
                        </tr>
                    </thead>
                    <tbody>
                        {buy_signals_html}
                    </tbody>
                </table>
            </div>'''
        
        # Build sell signals table
        sell_table = ""
        if sell_signals_html:
            sell_table = f'''
            <div style="margin: 20px 0;">
                <h2 style="color: #d32f2f; font-size: 18px; margin-bottom: 15px; border-left: 4px solid #d32f2f; padding-left: 10px;">
                    ❌ SAT SİNYALLERİ
                </h2>
                <table style="width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <thead>
                        <tr style="background: #ffebee;">
                            <th style="padding: 12px; text-align: left; font-weight: 600; color: #333;">Hisse</th>
                            <th style="padding: 12px; text-align: left; font-weight: 600; color: #333;">Olasılık</th>
                            <th style="padding: 12px; text-align: left; font-weight: 600; color: #333;">Fiyat</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sell_signals_html}
                    </tbody>
                </table>
            </div>'''
        
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
