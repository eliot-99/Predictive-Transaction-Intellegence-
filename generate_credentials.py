#!/usr/bin/env python3
"""
Generate all required credentials for FraudGuard BFSI deployment
Run this script before deploying to Render
"""

import secrets
import string
from cryptography.fernet import Fernet

def generate_secret_key(length=64):
    """Generate a secure Flask SECRET_KEY"""
    alphabet = string.ascii_letters + string.digits + string.punctuation
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def generate_fernet_key():
    """Generate a Fernet encryption key"""
    return Fernet.generate_key().decode()

def main():
    print("=" * 80)
    print("FraudGuard BFSI - Credential Generator")
    print("=" * 80)
    print()
    
    print("🔐 GENERATED CREDENTIALS")
    print("-" * 80)
    print()
    
    # Generate Fernet key
    fernet_key = generate_fernet_key()
    print("1. FERNET_KEY (for encrypting user data):")
    print(f"   {fernet_key}")
    print()
    
    # Generate Flask secret key
    secret_key = generate_secret_key()
    print("2. SECRET_KEY (for Flask sessions):")
    print(f"   {secret_key}")
    print()
    
    print("-" * 80)
    print()
    print("⚠️  IMPORTANT SECURITY NOTES:")
    print()
    print("1. SAVE THESE CREDENTIALS SECURELY!")
    print("   - Store in a password manager (1Password, LastPass, Bitwarden)")
    print("   - Never commit to Git")
    print("   - Never share publicly")
    print()
    print("2. FERNET_KEY is CRITICAL:")
    print("   - Used to encrypt user emails, names, and institutions")
    print("   - If lost, all encrypted data becomes unrecoverable")
    print("   - Backup in multiple secure locations")
    print()
    print("3. For Render deployment:")
    print("   - Add these as environment variables in Render Dashboard")
    print("   - Go to: Service → Environment → Add Environment Variable")
    print()
    print("-" * 80)
    print()
    print("📋 ADDITIONAL CREDENTIALS NEEDED:")
    print()
    print("3. MONGODB_URI:")
    print("   - Sign up at: https://www.mongodb.com/cloud/atlas")
    print("   - Create free M0 cluster")
    print("   - Get connection string")
    print("   - Format: mongodb+srv://user:pass@cluster.mongodb.net/")
    print()
    print("4. SMTP Credentials (Gmail):")
    print("   - Enable 2FA: https://myaccount.google.com/security")
    print("   - Generate App Password: https://myaccount.google.com/apppasswords")
    print("   - SMTP_SERVER: smtp.gmail.com")
    print("   - SMTP_PORT: 587")
    print("   - SMTP_USERNAME: your-email@gmail.com")
    print("   - SMTP_PASSWORD: (16-character app password)")
    print("   - SMTP_FROM_EMAIL: your-email@gmail.com")
    print()
    print("5. GEMINI_API_KEY:")
    print("   - Get key at: https://makersuite.google.com/app/apikey")
    print("   - Free tier available")
    print()
    print("=" * 80)
    print()
    print("✅ Credentials generated successfully!")
    print()
    print("Next steps:")
    print("1. Save these credentials securely")
    print("2. Set up MongoDB Atlas")
    print("3. Get SMTP credentials")
    print("4. Get Gemini API key")
    print("5. Follow RENDER_DEPLOYMENT_GUIDE.md")
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
