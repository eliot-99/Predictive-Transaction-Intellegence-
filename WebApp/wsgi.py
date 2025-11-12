"""
WSGI entry point for Vercel deployment
"""
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Set production environment for Vercel
os.environ.setdefault('FLASK_ENV', 'production')

try:
    from app import app
    # Vercel expects the WSGI application object
    application = app
    print("Flask app loaded successfully")
except Exception as e:
    print(f"Error loading Flask app: {e}")
    raise

if __name__ == "__main__":
    app.run()
