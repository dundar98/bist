@echo off
REM BIST100 Günlük Sinyal Tarama - Windows Batch Dosyası
REM 
REM Bu dosyayı Windows Task Scheduler ile zamanlayabilirsiniz:
REM 1. Task Scheduler açın (taskschd.msc)
REM 2. "Create Basic Task" seçin
REM 3. İsim: "BIST100 Günlük Tarama"
REM 4. Trigger: Daily, 09:00 (hafta içi)
REM 5. Action: Start a program
REM 6. Program: Bu .bat dosyasının tam yolu
REM
REM E-posta için aşağıdaki değişkenleri doldurun:

SET BIST_EMAIL_SENDER=your.email@gmail.com
SET BIST_EMAIL_PASSWORD=your_app_password
SET BIST_EMAIL_RECIPIENTS=recipient@example.com

REM Python yolunu ayarlayın (gerekirse)
SET PYTHON_PATH=python

REM Proje dizinine git
cd /d "c:\Users\dunda\OneDrive\Masaüstü\kod\bist"

REM Taramayı çalıştır
%PYTHON_PATH% scripts/run_daily_scan.py --email --data-source yfinance

REM Hata varsa bekle
IF %ERRORLEVEL% NEQ 0 (
    echo Hata oluştu! Çıkış kodu: %ERRORLEVEL%
    pause
)
