# FraudGuard BFSI - Web Application

A production-ready banking fraud detection web platform built with Flask, Bootstrap 5, and Chart.js.

## Features

- ✅ **Real-time Fraud Detection** - Integrates with FastAPI fraud detection API
- ✅ **User Authentication** - Secure login/signup with password hashing
- ✅ **Transaction History** - Track all predictions with detailed analytics
- ✅ **Risk Scoring** - Visual risk assessment (0.0 - 1.0 scale)
- ✅ **AI Chatbot** - Intelligent fraud assistant powered by rule-based logic
- ✅ **Responsive Design** - Mobile-first Bootstrap 5 UI
- ✅ **Data Export** - CSV export for compliance and analysis
- ✅ **Dashboard Analytics** - Real-time charts and statistics with Chart.js
- ✅ **Enterprise Security** - Password hashing, SQL injection prevention, XSS protection

## Project Structure

```
WebApp/
├── app.py                          # Flask application with all routes
├── config.py                       # Configuration management
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── models/
│   └── user.db                     # SQLite database (auto-created)
├── static/
│   ├── css/
│   │   └── style.css              # Custom banking-grade CSS
│   ├── js/
│   │   └── script.js              # Client-side JavaScript utilities
│   └── img/
│       └── (favicon, logo, etc)
└── templates/
    ├── base.html                   # Base template with navigation
    ├── login.html                  # Login page
    ├── signup.html                 # Registration page
    ├── dashboard.html              # Main dashboard with stats & charts
    ├── predict.html                # 25-field fraud prediction form
    ├── history.html                # Transaction history with pagination
    ├── chatbot.html                # AI fraud assistant chat interface
    ├── about.html                  # About & features page
    ├── 404.html                    # Not found error page
    └── 500.html                    # Server error page
```

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/fraudguard-bfsi.git
cd fraudguard-bfsi/WebApp
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 5. Initialize Database
```bash
flask db init
```

Or with CLI command:
```bash
python -c "from app import app, db; app.app_context().push(); db.create_all()"
```

### 6. Seed Demo Data (Optional)
```bash
flask seed-demo
```

Demo credentials:
- Email: `demo@fraudguard.com`
- Password: `demo123`

## Running Locally

### Development Server
```bash
python app.py
```

Server runs on `http://localhost:5000`

### With Gunicorn (Production-like)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Configuration

### Environment Variables

Create `.env` file in WebApp directory:

```env
# Flask
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
DEBUG=False

# External API
FRAUD_API_URL=https://fraud-detection-api.onrender.com/detect
FRAUD_API_TIMEOUT=10

# Database
SQLALCHEMY_DATABASE_URI=sqlite:///models/user.db

# Server
PORT=5000
```

### Database Models

#### User Model
```python
- id: Primary key
- name: Bank name (unique per bank)
- email: Admin email (unique)
- bank_id: Unique bank identifier
- password_hash: Hashed password (Werkzeug)
- created_at: Account creation timestamp
- transactions: Relationship to Transaction model
```

#### Transaction Model
```python
- id: Primary key
- user_id: Foreign key to User
- transaction_id: Unique transaction ID from API
- amount: Transaction amount
- location: Geographic location
- fraud_probability: ML model probability (0.0-1.0)
- risk_score: Final risk score with boosters
- alert_triggered: Boolean flag for high-risk transactions
- alert_reasons: Comma-separated reason list
- prediction: ML model prediction (0 or 1)
- created_at: Prediction timestamp
```

## API Integration

### Fraud Detection API Endpoint

POST `/detect` (External API)

**Request:**
```json
{
  "User_ID": 1,
  "Transaction_Amount": 1000000,
  "Transaction_Location": "Tashkent",
  "Merchant_ID": 1,
  "Device_ID": 1,
  "Card_Type": "Credit",
  "Transaction_Currency": "UZS",
  "Transaction_Status": "Completed",
  "Previous_Transaction_Count": 5,
  "Distance_Between_Transactions_km": 0,
  "Time_Since_Last_Transaction_min": 60,
  "Authentication_Method": "PIN",
  "Transaction_Velocity": 1,
  "Transaction_Category": "Shopping",
  "Transaction_Hour": 12,
  "Transaction_Day": 15,
  "Transaction_Month": 6,
  "Transaction_Weekday": 2,
  "Log_Transaction_Amount": 13.8,
  "Velocity_Distance_Interact": 0,
  "Amount_Velocity_Interact": 0,
  "Time_Distance_Interact": 0,
  "Hour_sin": 0.0,
  "Hour_cos": 1.0,
  "Weekday_sin": 0.0,
  "Weekday_cos": 1.0
}
```

**Response:**
```json
{
  "Transaction_ID": 1761672755316,
  "User_ID": 1,
  "Fraud_Probability": 0.5,
  "Final_Risk_Score": 0.95,
  "isFraud_pred": 0,
  "alert_triggered": true,
  "alert_reasons": ["High Amount", "Night Transaction", "Foreign Location"],
  "timestamp": "2025-10-28T23:02:35.316039"
}
```

## Routes

### Public Routes
- `GET /` - Redirect to dashboard or login
- `GET /login` - Login page
- `POST /login` - Process login
- `GET /signup` - Registration page
- `POST /signup` - Process registration
- `GET /about` - About page with features

### Protected Routes (require authentication)
- `GET /dashboard` - Main dashboard with statistics
- `GET /predict` - Fraud prediction form
- `POST /predict` - Submit prediction and call API
- `GET /history` - Transaction history
- `GET /export-csv` - Export history as CSV
- `GET /chatbot` - AI chatbot interface
- `POST /api/chat` - Chatbot API endpoint
- `GET /api/history` - JSON API for history

### Error Routes
- `GET /404` - Page not found
- `GET /500` - Server error

## Features Explained

### Dashboard
- **Statistics Cards**: Today's transactions, fraud rate, high-risk alerts, average risk score
- **Real-Time Chart**: 24-hour fraud probability trend visualization
- **Recent Alerts Table**: Last 5 flagged transactions
- **Quick Stats**: Security score, model accuracy, API status

### Fraud Prediction Form (25 Fields)
1. **Basic Info**: User ID, Amount, Location, Merchant ID, Device ID
2. **Card Details**: Card Type, Currency, Status
3. **Behavioral**: Previous transaction count, distance, time since last tx
4. **Authentication**: Authentication method
5. **Velocity**: Transaction velocity per hour
6. **Category**: Transaction category
7. **Temporal**: Hour, day, month, weekday
8. **Calculated Features**: Sine/cosine transforms, interaction terms

### Transaction History
- Paginated table (10 per page) with filtering
- Risk level badges (Safe, Medium, High)
- Alert reason display
- CSV export functionality
- Transaction detail modal

### AI Chatbot
- Multi-turn conversation interface
- Smart responses based on keywords
- Quick question suggestions
- Real-time fraud guidance

## Color Scheme (Banking Professional)

- **Primary**: `#003087` (Navy Blue)
- **Secondary**: `#00A3E0` (Sky Blue)
- **Success**: `#28a745` (Green)
- **Danger**: `#dc3545` (Red)
- **Warning**: `#ffc107` (Yellow)
- **Background**: `#f8f9fa` (Light Gray)

## Security Features

✅ **Authentication**
- Session-based authentication with Flask
- Password hashing using Werkzeug
- Automatic session timeout after 7 days

✅ **Data Protection**
- HTTPS/SSL encryption (Render)
- SQL injection prevention (SQLAlchemy ORM)
- XSS protection via Jinja2 auto-escaping
- CSRF protection via Flask-WTF

✅ **API Security**
- Request timeout (10 seconds)
- Error handling and logging
- API health checks
- Rate limiting ready (can be added with Flask-Limiter)

## Performance Optimization

- **Lazy Loading**: Charts loaded on demand
- **Pagination**: Transaction history paginated (10 per page)
- **Caching**: Static assets cached by browser
- **Compression**: CSS/JS minification ready
- **Database**: Indexed queries on common filters
- **API**: Async-ready with error handling

## Deployment

### Render Deployment

1. **Connect GitHub**
   ```bash
   git push origin main
   ```

2. **Set Environment Variables**
   - `SECRET_KEY`: Use secure random string
   - `FRAUD_API_URL`: Render API endpoint
   - `FLASK_ENV`: production

3. **Auto-Deploy Configuration**
   - Edit `render.yaml` in root
   - Push changes to trigger deployment
   - View logs: `render logs fraudguard-webapp`

### Local Production Build
```bash
# Install production dependencies
pip install -r requirements.txt

# Build database
python -c "from app import app, db; app.app_context().push(); db.create_all()"

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'flask'"
```bash
pip install -r requirements.txt
```

### "Database locked" errors
```bash
# Delete existing database and reinitialize
rm models/user.db
python -c "from app import app, db; app.app_context().push(); db.create_all()"
```

### API connection timeout
- Check `FRAUD_API_URL` environment variable
- Verify API service is running
- Check network connectivity
- View Flask logs for details

### Port already in use
```bash
# Change port in app.py or use:
python app.py --port 5001
```

## Testing

### Manual Testing
1. Create account at `/signup`
2. Login at `/login` with credentials
3. View dashboard at `/dashboard`
4. Test prediction with sample data at `/predict`
5. Export CSV from `/history`
6. Chat with AI at `/chatbot`

### Browser DevTools
- Check Console for JavaScript errors
- Monitor Network tab for API calls
- Validate HTML with inspector

### API Testing
```bash
# Test API health
curl http://localhost:5000/api/health

# Test chat endpoint
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Is 75M from Russia safe?"}'
```

## Future Enhancements

- [ ] Two-factor authentication (2FA)
- [ ] Rate limiting per user
- [ ] Advanced analytics dashboard
- [ ] Webhook notifications
- [ ] Mobile app (React Native/Flutter)
- [ ] Dark mode toggle
- [ ] Batch prediction upload
- [ ] Role-based access control (RBAC)
- [ ] Audit logging
- [ ] Machine learning model versioning

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Support

- 📧 Email: contact@fraudguard.bfsi
- 🐛 Issues: GitHub Issues
- 💬 Discussions: GitHub Discussions
- 📖 Documentation: See `/about` page

---

**Built with** ❤️ **for Banking & Financial Services**