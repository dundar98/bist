#!/usr/bin/env python3
"""
E-posta Test Scripti

Bu script e-posta ayarlarınızın doğru çalışıp çalışmadığını test eder.

Gmail için App Password oluşturma:
1. https://myaccount.google.com/apppasswords adresine gidin
2. "Select app" → "Mail" seçin
3. "Select device" → "Other" → "BIST Trading" yazın
4. "Generate" tıklayın
5. 16 karakterlik şifreyi kopyalayın (boşlukları silin)

Kullanım:
    python scripts/test_email.py --sender your@gmail.com --password xxxx --recipient your@email.com
"""

import argparse
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


def send_test_email(
    sender_email: str,
    sender_password: str,
    recipient_email: str,
    smtp_server: str = "smtp.gmail.com",
    smtp_port: int = 587,
) -> bool:
    """Test email connection and send a test message."""
    
    print("\n" + "=" * 50)
    print("📧 E-POSTA BAGLANTI TESTİ")
    print("=" * 50)
    
    # Test message
    subject = f"🧪 BIST100 Trading System - Test E-postası"
    body = f"""
Merhaba!

Bu bir test e-postasıdır. E-posta sisteminiz başarıyla çalışıyor! ✅

📅 Tarih: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
📤 Gönderen: {sender_email}
📥 Alıcı: {recipient_email}
🖥️ SMTP: {smtp_server}:{smtp_port}

Artık günlük sinyal raporları bu adrese gönderilecek.

İyi yatırımlar! 📈

---
BIST100 Deep Learning Trading System
"""

    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="font-family: Arial, sans-serif; background: #1a1a2e; color: #fff; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background: #16213e; border-radius: 10px; padding: 30px;">
            <h1 style="color: #00ff88; text-align: center;">✅ Test Başarılı!</h1>
            <p style="font-size: 16px; line-height: 1.6;">
                E-posta sisteminiz düzgün çalışıyor. Artık günlük BIST100 sinyal raporları
                bu adrese gönderilecek.
            </p>
            <div style="background: #0f3460; padding: 15px; border-radius: 8px; margin: 20px 0;">
                <p><strong>📅 Tarih:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>📤 Gönderen:</strong> {sender_email}</p>
                <p><strong>📥 Alıcı:</strong> {recipient_email}</p>
            </div>
            <p style="text-align: center; color: #888; font-size: 12px;">
                BIST100 Deep Learning Trading System
            </p>
        </div>
    </body>
    </html>
    """
    
    try:
        print(f"\n1️⃣ SMTP sunucusuna bağlanılıyor: {smtp_server}:{smtp_port}")
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.set_debuglevel(0)  # Set to 1 for verbose output
        
        print("2️⃣ TLS şifrelemesi başlatılıyor...")
        server.starttls()
        
        print(f"3️⃣ Giriş yapılıyor: {sender_email}")
        server.login(sender_email, sender_password)
        print("   ✅ Giriş başarılı!")
        
        print(f"4️⃣ Test e-postası gönderiliyor: {recipient_email}")
        
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = recipient_email
        
        msg.attach(MIMEText(body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))
        
        server.sendmail(sender_email, recipient_email, msg.as_string())
        
        print("   ✅ E-posta gönderildi!")
        
        server.quit()
        
        print("\n" + "=" * 50)
        print("✅ BAŞARILI! E-posta ayarlarınız doğru çalışıyor.")
        print("=" * 50)
        print(f"\n📬 Gelen kutunuzu kontrol edin: {recipient_email}")
        print("   (Spam klasörünü de kontrol etmeyi unutmayın)")
        
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        print("\n" + "=" * 50)
        print("❌ GİRİŞ HATASI!")
        print("=" * 50)
        print("\nOlası sebepler:")
        print("1. Yanlış e-posta veya şifre")
        print("2. Gmail için normal şifre yerine App Password gerekli")
        print("\n📌 Gmail App Password oluşturma:")
        print("   1. https://myaccount.google.com/apppasswords adresine gidin")
        print("   2. 2 Faktörlü Doğrulama açık olmalı")
        print("   3. 'Select app' → 'Mail' seçin")
        print("   4. 16 karakterlik şifreyi kopyalayın")
        print(f"\nHata detayı: {e}")
        return False
        
    except smtplib.SMTPException as e:
        print("\n" + "=" * 50)
        print("❌ SMTP HATASI!")
        print("=" * 50)
        print(f"\nHata: {e}")
        return False
        
    except Exception as e:
        print("\n" + "=" * 50)
        print("❌ BEKLENMEYEN HATA!")
        print("=" * 50)
        print(f"\nHata: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="BIST100 E-posta Test Scripti",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnek kullanım:
    python scripts/test_email.py --sender your@gmail.com --password xxxx --recipient your@email.com

Gmail App Password oluşturma:
    https://myaccount.google.com/apppasswords
        """
    )
    
    parser.add_argument(
        "--sender",
        required=True,
        help="Gönderen e-posta adresi (Gmail)"
    )
    
    parser.add_argument(
        "--password",
        required=True,
        help="Gmail App Password (16 karakter, boşluksuz)"
    )
    
    parser.add_argument(
        "--recipient",
        required=True,
        help="Alıcı e-posta adresi"
    )
    
    parser.add_argument(
        "--smtp-server",
        default="smtp.gmail.com",
        help="SMTP sunucusu (varsayılan: smtp.gmail.com)"
    )
    
    parser.add_argument(
        "--smtp-port",
        type=int,
        default=587,
        help="SMTP portu (varsayılan: 587)"
    )
    
    args = parser.parse_args()
    
    success = send_test_email(
        sender_email=args.sender,
        sender_password=args.password,
        recipient_email=args.recipient,
        smtp_server=args.smtp_server,
        smtp_port=args.smtp_port,
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
