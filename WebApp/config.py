"""
Configuration file for FraudGuard BFSI Platform
"""
import os
from datetime import timedelta


def parse_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def parse_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY')
    # Database configuration
    MONGODB_URI = os.environ.get('MONGODB_URI')
    MONGODB_DB_NAME = os.environ.get('MONGODB_DB_NAME') 
    DATA_ENCRYPTION_KEY = os.environ.get('DATA_ENCRYPTION_KEY')
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = parse_int(os.environ.get('MAIL_PORT'), 587)
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_USE_TLS = parse_bool(os.environ.get('MAIL_USE_TLS'), True)
    MAIL_USE_SSL = parse_bool(os.environ.get('MAIL_USE_SSL'), False)
    MAIL_FROM_ADDRESS = os.environ.get('MAIL_FROM_ADDRESS')
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # External API Configuration
    FRAUD_API_URL = os.environ.get('FRAUD_API_URL')
    FRAUD_API_TIMEOUT = 10
    
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    MONGODB_DB_NAME = os.environ.get('MONGODB_DB_NAME') or 'fraudguard_test'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}