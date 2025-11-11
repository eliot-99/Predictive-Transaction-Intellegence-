"""
FraudGuard BFSI - Banking Fraud Detection Web Platform
Main Flask Application (MongoDB + Gemini)
"""
import os
import csv
import json
import math
import hashlib
import secrets
import smtplib
import ssl
from datetime import datetime, timedelta
from email.message import EmailMessage
from io import StringIO, BytesIO
from functools import wraps
from types import SimpleNamespace

from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv
from google import genai
import requests
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify, send_file, g
)
from werkzeug.security import check_password_hash, generate_password_hash
import logging
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, PyMongoError, ServerSelectionTimeoutError
from pymongo.uri_parser import parse_uri
from bson.objectid import ObjectId

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

app = Flask(__name__)
app.config.from_object('config.DevelopmentConfig')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def resolve_mongo_timeout():
    raw_value = os.getenv('MONGODB_CONNECT_TIMEOUT_MS')
    if not raw_value:
        return 5000
    try:
        parsed = int(raw_value)
        if parsed <= 0:
            raise ValueError
        return parsed
    except ValueError:
        logger.warning('Invalid MONGODB_CONNECT_TIMEOUT_MS value %s, using default 5000', raw_value)
        return 5000


def setup_mongo():
    timeout_ms = resolve_mongo_timeout()
    uri = app.config['MONGODB_URI']
    try:
        parsed = parse_uri(uri)
        nodelist = parsed.get('nodelist') or []
        if nodelist:
            hosts = ','.join(f"{host}:{port}" if port else host for host, port in nodelist)
        elif parsed.get('is_srv'):
            hosts = parsed.get('database') or 'mongodb+srv'
        else:
            hosts = 'mongodb'
    except Exception:
        hosts = 'unresolved'
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms, connectTimeoutMS=timeout_ms)
        client.admin.command('ping')
    except ServerSelectionTimeoutError as error:
        logger.critical('MongoDB connection failed for hosts=%s: %s', hosts, error)
        raise
    except PyMongoError as error:
        logger.critical('MongoDB configuration error for hosts=%s: %s', hosts, error)
        raise
    db_name = app.config['MONGODB_DB_NAME']
    logger.info('MongoDB connected hosts=%s db=%s timeout_ms=%s', hosts or 'default', db_name, timeout_ms)
    db = client[db_name]
    users = db['users']
    transactions = db['transactions']
    password_resets = db['password_resets']
    try:
        users.create_index('email_hash', unique=True, partialFilterExpression={'email_hash': {'$exists': True}})
        transactions.create_index([('user_id', ASCENDING), ('created_at', DESCENDING)])
        password_resets.create_index('email_hash')
        password_resets.create_index('expires_at', expireAfterSeconds=0)
        logger.info('MongoDB indexes ready')
    except PyMongoError as index_error:
        logger.warning('MongoDB index creation issue: %s', index_error)
    return client, db, users, transactions, password_resets


mongo_client, mongo_db, users_collection, transactions_collection, password_reset_collection = setup_mongo()

_gemini_client = None

def get_gemini_client():
    global _gemini_client
    if _gemini_client:
        return _gemini_client
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError('GEMINI_API_KEY is not configured')
    _gemini_client = genai.Client(api_key=api_key)
    model_name = os.getenv('GEMINI_MODEL') or 'gemini-2.5-flash'
    logger.info('Gemini client ready model=%s', model_name)
    return _gemini_client


def get_cipher():
    key = app.config.get('DATA_ENCRYPTION_KEY')
    if not key:
        raise RuntimeError('DATA_ENCRYPTION_KEY is not configured')
    if isinstance(key, str):
        key = key.encode()
    return Fernet(key)


def encrypt_value(value):
    if value is None:
        return ''
    text = str(value).strip()
    if not text:
        return ''
    cipher = get_cipher()
    return cipher.encrypt(text.encode()).decode()


def decrypt_value(token):
    if not token:
        return ''
    cipher = get_cipher()
    if isinstance(token, bytes):
        token = token.decode()
    try:
        return cipher.decrypt(token.encode()).decode()
    except InvalidToken:
        logger.warning(f'Invalid Fernet token - key mismatch or corrupted data')
        return token
    except Exception as error:
        logger.error(f'Unable to decrypt value: {error}')
        return ''


def normalize_email(value):
    return (value or '').strip().lower()


def compute_email_hash(email):
    return hashlib.sha256(email.encode()).hexdigest()


def upgrade_user_doc_if_needed(doc, normalized_email):
    if not doc:
        return doc
    update_payload = {}
    if normalized_email and not doc.get('email_hash'):
        update_payload['email_hash'] = compute_email_hash(normalized_email)
    if normalized_email and not doc.get('email_encrypted'):
        update_payload['email_encrypted'] = encrypt_value(normalized_email)
    if not doc.get('full_name_encrypted') and doc.get('name'):
        update_payload['full_name_encrypted'] = encrypt_value(doc.get('name'))
    if not doc.get('institution_name_encrypted') and doc.get('bank_id'):
        update_payload['institution_name_encrypted'] = encrypt_value(doc.get('bank_id'))
    if update_payload:
        users_collection.update_one({'_id': doc['_id']}, {'$set': update_payload})
        doc.update(update_payload)
    return doc


def get_user_by_email(normalized_email):
    if not normalized_email:
        return None
    email_hash = compute_email_hash(normalized_email)
    user_doc = users_collection.find_one({'email_hash': email_hash})
    if user_doc:
        return user_doc
    legacy_doc = users_collection.find_one({'email': normalized_email})
    if legacy_doc:
        return upgrade_user_doc_if_needed(legacy_doc, normalized_email)
    return None


def get_decrypted_user_email(doc):
    if not doc:
        return ''
    encrypted = doc.get('email_encrypted')
    if encrypted:
        return decrypt_value(encrypted)
    return doc.get('email', '')


def get_decrypted_full_name(doc):
    if not doc:
        return ''
    encrypted = doc.get('full_name_encrypted')
    if encrypted:
        return decrypt_value(encrypted)
    return doc.get('name', '')


def get_decrypted_institution_name(doc):
    if not doc:
        return ''
    encrypted = doc.get('institution_name_encrypted')
    if encrypted:
        return decrypt_value(encrypted)
    return doc.get('bank_id', '')


def generate_reset_otp():
    return f"{secrets.randbelow(10000):04d}"


def send_password_reset_email(recipient_email, otp):
    server = app.config.get('MAIL_SERVER')
    username = app.config.get('MAIL_USERNAME')
    password = app.config.get('MAIL_PASSWORD')
    if not server or not username or not password:
        raise RuntimeError('Mail server configuration is incomplete')
    port = app.config.get('MAIL_PORT') or 587
    use_tls = app.config.get('MAIL_USE_TLS', True)
    use_ssl = app.config.get('MAIL_USE_SSL', False)
    sender = app.config.get('MAIL_FROM_ADDRESS') or username
    message = EmailMessage()
    message['Subject'] = 'FraudGuard BFSI Password Reset Code'
    message['From'] = sender
    message['To'] = recipient_email
    message.set_content(f'Your one-time password is {otp}. It expires in 10 minutes.')
    context = ssl.create_default_context()
    if use_ssl:
        with smtplib.SMTP_SSL(server, port, context=context) as smtp:
            smtp.login(username, password)
            smtp.send_message(message)
        return
    with smtplib.SMTP(server, port) as smtp:
        if use_tls:
            smtp.starttls(context=context)
        smtp.login(username, password)
        smtp.send_message(message)


# ==================== HELPERS ====================
def to_object_id(value):
    try:
        return ObjectId(value)
    except Exception:
        return None


def hydrate_user(doc):
    if not doc:
        return None
    full_name = get_decrypted_full_name(doc)
    institution_name = get_decrypted_institution_name(doc)
    email = get_decrypted_user_email(doc)
    payload = {
        'id': str(doc['_id']),
        'name': full_name or institution_name or '',
        'full_name': full_name or '',
        'institution_name': institution_name or '',
        'email': email,
        'created_at': doc.get('created_at')
    }
    return SimpleNamespace(**payload)


def hydrate_transaction(doc):
    if not doc:
        return None
    created_at = doc.get('created_at') or datetime.utcnow()
    return SimpleNamespace(
        id=str(doc['_id']),
        transaction_id=doc.get('transaction_id', ''),
        amount=float(doc.get('amount', 0.0)),
        location=doc.get('location', ''),
        fraud_probability=float(doc.get('fraud_probability', 0.0)),
        risk_score=float(doc.get('risk_score', 0.0)),
        alert_triggered=bool(doc.get('alert_triggered', False)),
        alert_reasons=doc.get('alert_reasons') or '',
        prediction=int(doc.get('prediction', 0)),
        created_at=created_at
    )


def ensure_user_context():
    session_user_id = session.get('user_id')
    if not session_user_id:
        g.user = None
        g.user_doc = None
        return
    object_id = to_object_id(session_user_id)
    if not object_id:
        session.clear()
        g.user = None
        g.user_doc = None
        return
    user_doc = users_collection.find_one({'_id': object_id})
    if not user_doc:
        session.clear()
        g.user = None
        g.user_doc = None
        return
    g.user_doc = user_doc
    g.user = hydrate_user(user_doc)


def get_current_user_object_id():
    if not getattr(g, 'user', None):
        return None
    return to_object_id(g.user.id)


def paginate_transactions(user_object_id, page, per_page):
    query = {'user_id': user_object_id}
    skip = max(page - 1, 0) * per_page
    cursor = transactions_collection.find(query).sort('created_at', DESCENDING).skip(skip).limit(per_page)
    items = [hydrate_transaction(doc) for doc in cursor]
    total = transactions_collection.count_documents(query)
    total_pages = max(1, math.ceil(total / per_page)) if total else 1
    return items, total, total_pages


def map_risk_type(label):
    normalized = (label or '').strip().upper()
    if normalized in ('HIGH', 'HIGH RISK'):
        return 'danger'
    if normalized in ('MEDIUM', 'MEDIUM RISK'):
        return 'warning'
    if normalized in ('LOW', 'LOW RISK', 'SAFE'):
        return 'success'
    return 'info'


def interpret_gemini_output(raw_text):
    message_text = raw_text.strip()
    risk_label = None
    try:
        parsed = json.loads(message_text)
        message_text = parsed.get('bot_message') or parsed.get('message') or parsed.get('response') or message_text
        risk_label = parsed.get('risk') or parsed.get('risk_level') or parsed.get('classification')
    except json.JSONDecodeError:
        lines = [line.strip() for line in message_text.splitlines() if line.strip()]
        for line in reversed(lines):
            lowered = line.lower()
            if 'risk classification' in lowered or 'risk level' in lowered:
                risk_label = line.split(':', 1)[1].strip() if ':' in line else line.split()[-1]
                lines.remove(line)
                message_text = '\n'.join(lines).strip()
                break
    message_text = message_text.strip()
    risk_type = map_risk_type(risk_label)
    if risk_label:
        formatted_label = risk_label.title()
    else:
        formatted_label = 'Info'
    final_message = message_text if message_text else raw_text.strip()
    if risk_label:
        final_message = f"{final_message}\n\nRisk classification: {formatted_label}"
    return final_message, risk_type


def keyword_fallback_response(user_message):
    message_lower = user_message.lower()
    response = {
        'bot_message': 'Thank you for your question. How can FraudGuard help protect your bank today?',
        'type': 'info',
        'model': 'keyword-fallback'
    }
    if any(word in message_lower for word in ['risk', 'safe', 'fraud', 'dangerous', 'suspicious']):
        if any(word in message_lower for word in ['million', 'large', 'high', 'big']):
            response['bot_message'] = 'HIGH RISK ALERT: Large transactions may trigger fraud alerts, especially if combined with night hours or foreign locations. Always verify with your bank!'
            response['type'] = 'danger'
        else:
            response['bot_message'] = 'Risk assessment: Your transaction appears safe based on normal patterns. Monitor for unusual activity.'
            response['type'] = 'success'
    elif any(word in message_lower for word in ['russia', 'turkey', 'usa', 'china', 'uae', 'foreign', 'international']):
        response['bot_message'] = 'Foreign transactions may trigger alerts if combined with high amounts or unusual hours. Always verify location and amount before confirming.'
        response['type'] = 'warning'
    elif any(word in message_lower for word in ['night', 'midnight', 'early', 'morning', '2am', '3am', '4am']):
        response['bot_message'] = 'Night transactions (12 AM - 5 AM) are flagged as potentially risky. Combine with other factors for full risk assessment.'
        response['type'] = 'warning'
    elif any(word in message_lower for word in ['feature', 'model', 'accuracy', 'auc', 'performance']):
        response['bot_message'] = 'FraudGuard uses XGBoost & LightGBM models with 99.1% AUC. Real-time risk scoring combines ML predictions with rule-based fraud signatures.'
        response['type'] = 'info'
    elif any(word in message_lower for word in ['help', 'how', 'what', 'guide', 'support']):
        response['bot_message'] = 'Use FraudGuard to: (1) Predict fraud risk for transactions, (2) View transaction history, (3) Get real-time alerts, (4) Export reports. Visit /about for more info!'
        response['type'] = 'info'
    return response


def call_gemini_api(user_message):
    client = get_gemini_client()
    model_name = os.getenv('GEMINI_MODEL') or 'gemini-2.5-flash'
    guidance = (
        "You are FraudGuard AI, a banking fraud prevention assistant. Provide precise, actionable guidance for banking fraud prevention, "
        "assess the risk level as HIGH, MEDIUM, or LOW, and recommend next steps. Keep responses concise (max 3 sentences). "
        "End the response with a line that states 'Risk classification: <LEVEL>'."
    )
    prompt = f"{guidance}\n\nUser question: {user_message.strip()}"
    try:
        result = client.models.generate_content(model=model_name, contents=prompt)
    except Exception as error:
        logger.error('Gemini request failed: %s', error)
        raise
    texts = []
    primary = getattr(result, 'text', None)
    if primary:
        texts.append(primary)
    candidates = getattr(result, 'candidates', None) or []
    for candidate in candidates:
        content = getattr(candidate, 'content', None)
        parts = getattr(content, 'parts', None) if content else None
        if parts:
            for part in parts:
                value = getattr(part, 'text', None)
                if value:
                    texts.append(value)
    merged = '\n'.join(segment.strip() for segment in texts if segment and segment.strip()).strip()
    if not merged:
        raise RuntimeError('Gemini response contained no text')
    return merged


# ==================== AUTHENTICATION HELPERS ====================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if g.user is None:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# ==================== CONTEXT PROCESSORS ====================
@app.before_request
def before_request():
    ensure_user_context()


@app.context_processor
def inject_user():
    return {'current_user': getattr(g, 'user', None)}


# ==================== ROUTES ====================
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        institution_name = request.form.get('institution_name', '').strip()
        full_name = request.form.get('full_name', '').strip()
        email = normalize_email(request.form.get('email', ''))
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not all([institution_name, full_name, email, password, confirm_password]):
            flash('All fields are required.', 'danger')
            return redirect(url_for('signup'))
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return redirect(url_for('signup'))
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('signup'))

        email_hash = compute_email_hash(email)
        if users_collection.find_one({'email_hash': email_hash}):
            flash('Email already registered.', 'warning')
            return redirect(url_for('login'))

        try:
            user_doc = {
                'full_name_encrypted': encrypt_value(full_name),
                'institution_name_encrypted': encrypt_value(institution_name),
                'email_hash': email_hash,
                'email_encrypted': encrypt_value(email),
                'password_hash': generate_password_hash(password),
                'created_at': datetime.utcnow()
            }
            users_collection.insert_one(user_doc)
            flash('Signup successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except DuplicateKeyError as dup_error:
            logger.warning(f'Registration conflict: {dup_error}')
            flash('Account already exists. Please log in.', 'warning')
            return redirect(url_for('login'))
        except PyMongoError as db_error:
            logger.error(f'Signup error: {db_error}')
            flash('An error occurred during signup. Please try again.', 'danger')
            return redirect(url_for('signup'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = normalize_email(request.form.get('email', ''))
        password = request.form.get('password', '')

        if not email or not password:
            flash('Email and password required.', 'danger')
            return redirect(url_for('login'))

        user_doc = get_user_by_email(email)
        if not user_doc or not check_password_hash(user_doc.get('password_hash', ''), password):
            flash('Invalid email or password.', 'danger')
            return redirect(url_for('login'))

        session['user_id'] = str(user_doc['_id'])
        session.permanent = True
        app.permanent_session_lifetime = timedelta(days=7)

        full_name = get_decrypted_full_name(user_doc)
        flash(f"Welcome back, {full_name or 'User'}!", 'success')
        return redirect(url_for('dashboard'))

    return render_template('login.html')


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        step = request.form.get('step', 'request')
        if step == 'request':
            email = normalize_email(request.form.get('email', ''))
            if not email:
                flash('Email is required.', 'danger')
                return render_template('forgot_password.html', stage='request', email=email)
            user_doc = get_user_by_email(email)
            if not user_doc:
                flash('No account found for this email.', 'danger')
                return render_template('forgot_password.html', stage='request', email=email)
            otp = generate_reset_otp()
            otp_hash = generate_password_hash(otp)
            expires_at = datetime.utcnow() + timedelta(minutes=10)
            email_hash = compute_email_hash(email)
            password_reset_collection.delete_many({'email_hash': email_hash})
            password_reset_collection.insert_one({
                'email_hash': email_hash,
                'otp_hash': otp_hash,
                'expires_at': expires_at,
                'created_at': datetime.utcnow()
            })
            recipient = get_decrypted_user_email(user_doc) or email
            if recipient.startswith('gAAAAA'):
                logger.error(f'Decryption failed for user email. Fernet key mismatch.')
                flash('System configuration error. Please contact support.', 'danger')
                return render_template('forgot_password.html', stage='request', email=email)
            try:
                send_password_reset_email(recipient, otp)
            except Exception as error:
                logger.error(f'Password reset email error: {error}')
                flash('Unable to send OTP. Please try again later.', 'danger')
                return render_template('forgot_password.html', stage='request', email=email)
            session['reset_email_hash'] = email_hash
            session['reset_email_value'] = recipient
            flash('A verification code has been sent to your email.', 'info')
            return render_template('forgot_password.html', stage='verify', email=recipient)
        if step == 'verify':
            email_hash = session.get('reset_email_hash')
            cached_email = session.get('reset_email_value', '')
            if not email_hash:
                flash('Reset session expired. Please request a new code.', 'warning')
                return redirect(url_for('forgot_password'))
            otp_input = request.form.get('otp', '').strip()
            password_value = request.form.get('password', '')
            confirm_password = request.form.get('confirm_password', '')
            if not otp_input or not password_value or not confirm_password:
                flash('All fields are required.', 'danger')
                return render_template('forgot_password.html', stage='verify', email=cached_email)
            if password_value != confirm_password:
                flash('Passwords do not match.', 'danger')
                return render_template('forgot_password.html', stage='verify', email=cached_email)
            if len(password_value) < 6:
                flash('Password must be at least 6 characters.', 'danger')
                return render_template('forgot_password.html', stage='verify', email=cached_email)
            reset_doc = password_reset_collection.find_one({'email_hash': email_hash})
            if not reset_doc:
                flash('Invalid or expired reset request.', 'danger')
                return redirect(url_for('forgot_password'))
            if reset_doc.get('expires_at') and reset_doc['expires_at'] < datetime.utcnow():
                password_reset_collection.delete_many({'email_hash': email_hash})
                flash('OTP expired. Please request a new code.', 'danger')
                return redirect(url_for('forgot_password'))
            if not check_password_hash(reset_doc.get('otp_hash', ''), otp_input):
                flash('Invalid OTP.', 'danger')
                return render_template('forgot_password.html', stage='verify', email=cached_email)
            users_collection.update_one(
                {'email_hash': email_hash},
                {'$set': {'password_hash': generate_password_hash(password_value)}}
            )
            password_reset_collection.delete_many({'email_hash': email_hash})
            session.pop('reset_email_hash', None)
            session.pop('reset_email_value', None)
            flash('Password updated successfully. You can now log in.', 'success')
            return redirect(url_for('login'))
    stored_email = session.get('reset_email_value', '')
    return render_template('forgot_password.html', stage='request', email=stored_email)


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    user_object_id = get_current_user_object_id()
    if not user_object_id:
        flash('Session expired. Please log in again.', 'warning')
        return redirect(url_for('logout'))

    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow_start = today_start + timedelta(days=1)

    today_query = {
        'user_id': user_object_id,
        'created_at': {'$gte': today_start, '$lt': tomorrow_start}
    }
    today_transactions = list(transactions_collection.find(today_query))
    today_count = len(today_transactions)
    fraud_count = sum(1 for tx in today_transactions if tx.get('alert_triggered'))
    fraud_rate = (fraud_count / today_count * 100) if today_count else 0

    high_risk = transactions_collection.count_documents({'user_id': user_object_id, 'alert_triggered': True})

    avg_risk = 0
    try:
        aggregation = transactions_collection.aggregate([
            {'$match': {'user_id': user_object_id}},
            {'$group': {'_id': None, 'avg_risk': {'$avg': '$risk_score'}}}
        ])
        avg_result = next(aggregation, None)
        if avg_result and avg_result.get('avg_risk') is not None:
            avg_risk = float(avg_result['avg_risk'])
    except PyMongoError as agg_error:
        logger.error(f'Average risk aggregation error: {agg_error}')

    recent_cursor = transactions_collection.find({
        'user_id': user_object_id,
        'alert_triggered': True
    }).sort('created_at', DESCENDING).limit(5)
    recent_alerts = [hydrate_transaction(doc) for doc in recent_cursor]

    from random import randint
    chart_data = {
        'labels': [f'{i}:00' for i in range(24)],
        'data': [round(randint(10, 95) / 100, 2) for _ in range(24)]
    }

    return render_template(
        'dashboard.html',
        today_count=today_count,
        fraud_rate=round(fraud_rate, 1),
        high_risk=high_risk,
        avg_risk=round(avg_risk, 2),
        recent_alerts=recent_alerts,
        chart_data=chart_data
    )


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            selected_model = request.form.get('selected_model', 'ensemble')
            if not selected_model:
                flash('Please select a prediction model.', 'warning')
                return render_template('predict.html', show_result=False)

            def safe_int(val, default=0):
                try:
                    return int(val) if val else default
                except (ValueError, TypeError):
                    return default

            def safe_float(val, default=0.0):
                try:
                    return float(val) if val else default
                except (ValueError, TypeError):
                    return default

            payload = {
                'User_ID': safe_int(request.form.get('User_ID'), 1),
                'Transaction_Amount': safe_float(request.form.get('Transaction_Amount'), 0),
                'Transaction_Location': request.form.get('Transaction_Location', 'Tashkent') or 'Tashkent',
                'Merchant_ID': safe_int(request.form.get('Merchant_ID'), 1),
                'Device_ID': safe_int(request.form.get('Device_ID'), 1),
                'Card_Type': request.form.get('Card_Type', 'Credit') or 'Credit',
                'Transaction_Currency': request.form.get('Transaction_Currency', 'UZS') or 'UZS',
                'Transaction_Status': request.form.get('Transaction_Status', 'Completed') or 'Completed',
                'Previous_Transaction_Count': safe_int(request.form.get('Previous_Transaction_Count'), 5),
                'Distance_Between_Transactions_km': safe_float(request.form.get('Distance_Between_Transactions_km'), 0),
                'Time_Since_Last_Transaction_min': safe_int(request.form.get('Time_Since_Last_Transaction_min'), 60),
                'Authentication_Method': request.form.get('Authentication_Method', 'PIN') or 'PIN',
                'Transaction_Velocity': safe_int(request.form.get('Transaction_Velocity'), 1),
                'Transaction_Category': request.form.get('Transaction_Category', 'Shopping') or 'Shopping',
                'Transaction_Hour': safe_int(request.form.get('Transaction_Hour'), 12),
                'Transaction_Day': safe_int(request.form.get('Transaction_Day'), 15),
                'Transaction_Month': safe_int(request.form.get('Transaction_Month'), 6),
                'Transaction_Weekday': safe_int(request.form.get('Transaction_Weekday'), 2),
                'Log_Transaction_Amount': safe_float(request.form.get('Log_Transaction_Amount'), 0),
                'Velocity_Distance_Interact': safe_float(request.form.get('Velocity_Distance_Interact'), 0),
                'Amount_Velocity_Interact': safe_float(request.form.get('Amount_Velocity_Interact'), 0),
                'Time_Distance_Interact': safe_float(request.form.get('Time_Distance_Interact'), 0),
                'Hour_sin': safe_float(request.form.get('Hour_sin'), 0),
                'Hour_cos': safe_float(request.form.get('Hour_cos'), 1),
                'Weekday_sin': safe_float(request.form.get('Weekday_sin'), 0),
                'Weekday_cos': safe_float(request.form.get('Weekday_cos'), 1)
            }

            base_api_url = app.config['FRAUD_API_URL'].replace('/detect', '')
            if selected_model == 'ensemble':
                api_url = f"{base_api_url}/ensemble"
                logger.info('Using Ensemble voting with all models')
            else:
                api_url = f"{base_api_url}/detect?model={selected_model}"
                logger.info(f'Using single model: {selected_model}')

            logger.info(f'Calling fraud detection API: {api_url}')
            response = requests.post(
                api_url,
                json=payload,
                timeout=app.config['FRAUD_API_TIMEOUT']
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f'API Response: {result}')

            user_object_id = get_current_user_object_id()
            if not user_object_id:
                flash('Session expired. Please log in again.', 'warning')
                return redirect(url_for('logout'))

            transaction_doc = {
                'user_id': user_object_id,
                'transaction_id': str(result.get('Transaction_ID', int(datetime.utcnow().timestamp() * 1000))),
                'amount': payload['Transaction_Amount'],
                'location': payload['Transaction_Location'],
                'fraud_probability': float(result.get('Fraud_Probability', 0)),
                'risk_score': float(result.get('Final_Risk_Score', 0)),
                'alert_triggered': bool(result.get('alert_triggered', False)),
                'alert_reasons': ', '.join(result.get('alert_reasons', [])),
                'prediction': int(result.get('isFraud_pred', 0)),
                'model_used': selected_model,
                'created_at': datetime.utcnow()
            }
            transactions_collection.insert_one(transaction_doc)

            risk_level = 'SAFE'
            risk_class = 'success'
            if transaction_doc['alert_triggered']:
                risk_level = 'HIGH'
                risk_class = 'danger'
            elif transaction_doc['risk_score'] > 0.4:
                risk_level = 'MEDIUM'
                risk_class = 'warning'

            flash('Prediction completed successfully!', 'success')
            return render_template(
                'predict.html',
                result=result,
                payload=payload,
                risk_level=risk_level,
                risk_class=risk_class,
                selected_model=selected_model,
                show_result=True,
                now=datetime.utcnow()
            )
        except requests.exceptions.RequestException as api_error:
            logger.error(f'API call failed: {api_error}')
            flash(f'API Error: Unable to reach fraud detection service. {str(api_error)}', 'danger')
            return render_template('predict.html', show_result=False)
        except Exception as error:
            logger.error(f'Prediction error: {error}')
            flash(f'Error: {str(error)}', 'danger')
            return render_template('predict.html', show_result=False)

    return render_template('predict.html', show_result=False)


@app.route('/history')
@login_required
def history():
    page = request.args.get('page', 1, type=int)
    per_page = 10

    user_object_id = get_current_user_object_id()
    if not user_object_id:
        flash('Session expired. Please log in again.', 'warning')
        return redirect(url_for('logout'))

    transactions, total, total_pages = paginate_transactions(user_object_id, page, per_page)

    return render_template(
        'history.html',
        transactions=transactions,
        page=page,
        total_pages=total_pages,
        total_transactions=total,
        max=max,
        min=min
    )


@app.route('/api/history')
@login_required
def api_history():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)

    user_object_id = get_current_user_object_id()
    if not user_object_id:
        return jsonify({'error': 'Session expired'}), 401

    transactions, total, total_pages = paginate_transactions(user_object_id, page, per_page)

    data = []
    for tx in transactions:
        data.append({
            'id': tx.id,
            'transaction_id': tx.transaction_id,
            'amount': tx.amount,
            'location': tx.location,
            'fraud_probability': tx.fraud_probability,
            'risk_score': tx.risk_score,
            'alert_triggered': tx.alert_triggered,
            'alert_reasons': tx.alert_reasons,
            'prediction': tx.prediction,
            'created_at': tx.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })

    return jsonify({
        'data': data,
        'page': page,
        'per_page': per_page,
        'total': total,
        'pages': total_pages
    })


@app.route('/export-csv')
@login_required
def export_csv():
    try:
        user_object_id = get_current_user_object_id()
        if not user_object_id:
            flash('Session expired. Please log in again.', 'warning')
            return redirect(url_for('logout'))

        cursor = transactions_collection.find({'user_id': user_object_id}).sort('created_at', DESCENDING)
        transactions = [hydrate_transaction(doc) for doc in cursor]

        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['Transaction ID', 'Date', 'Amount (UZS)', 'Location', 'Fraud Probability', 'Risk Score', 'Alert', 'Reasons'])

        for tx in transactions:
            writer.writerow([
                tx.transaction_id,
                tx.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                f"{tx.amount:,.2f}",
                tx.location,
                f"{tx.fraud_probability:.1%}",
                f"{tx.risk_score:.2f}",
                'YES' if tx.alert_triggered else 'NO',
                tx.alert_reasons or 'N/A'
            ])

        output.seek(0)
        bytes_output = BytesIO(output.getvalue().encode('utf-8'))
        bytes_output.seek(0)

        return send_file(
            bytes_output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"fraudguard_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    except Exception as error:
        import traceback
        logger.error(f'CSV export error: {error}\n{traceback.format_exc()}')
        flash('Error exporting CSV.', 'danger')
        return redirect(url_for('history'))


@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    try:
        user_message = request.json.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        try:
            gemini_raw = call_gemini_api(user_message)
            bot_message, response_type = interpret_gemini_output(gemini_raw)
            return jsonify({
                'bot_message': bot_message,
                'type': response_type,
                'model': 'gemini-1.5-flash'
            })
        except Exception as gemini_error:
            logger.error(f'Gemini API error: {gemini_error}')

        fallback = keyword_fallback_response(user_message)
        return jsonify(fallback)
    except Exception as error:
        import traceback
        logger.error(f'Chatbot error: {error}')
        logger.error(traceback.format_exc())
        return jsonify({'error': str(error), 'bot_message': 'An error occurred. Please try again.'}), 500


@app.route('/chatbot')
@login_required
def chatbot():
    return render_template('chatbot.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/404')
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


@app.route('/500')
@app.errorhandler(500)
def server_error(error):
    return render_template('500.html'), 500


# ==================== CLI COMMANDS ====================
@app.cli.command()
def init_db():
    try:
        users_collection.create_index('email_hash', unique=True, partialFilterExpression={'email_hash': {'$exists': True}})
        transactions_collection.create_index([('user_id', ASCENDING), ('created_at', DESCENDING)])
        password_reset_collection.create_index('email_hash')
        password_reset_collection.create_index('expires_at', expireAfterSeconds=0)
        print('MongoDB indexes ensured.')
    except PyMongoError as error:
        print(f'Index creation failed: {error}')


@app.cli.command()
def seed_demo():
    try:
        demo_email = 'demo@fraudguard.com'
        normalized_email = normalize_email(demo_email)
        email_hash = compute_email_hash(normalized_email)
        if not users_collection.find_one({'email_hash': email_hash}):
            users_collection.insert_one({
                'full_name_encrypted': encrypt_value('Demo User'),
                'institution_name_encrypted': encrypt_value('Demo Bank'),
                'email_hash': email_hash,
                'email_encrypted': encrypt_value(normalized_email),
                'password_hash': generate_password_hash('demo123'),
                'created_at': datetime.utcnow()
            })
            print('Demo user created.')
        else:
            print('Demo user already exists.')
    except PyMongoError as error:
        print(f'Demo user creation failed: {error}')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
