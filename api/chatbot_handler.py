import logging
import random
import re
from typing import Optional

logger = logging.getLogger(__name__)


class ChatbotHandler:
    """Handler for rule-based chatbot with fraud detection knowledge"""

    def __init__(self):
        self.loaded = False
        self.fraud_responses = {
            'signs': [
                "Common signs of credit card fraud include: unusual transaction locations, high-value purchases outside your spending pattern, multiple small transactions testing card validity, and transactions after card has been reported lost.",
                "Watch for these fraud indicators: transactions in foreign countries you haven't visited, purchases of high-value items you wouldn't normally buy, and multiple failed transaction attempts.",
                "Fraud red flags include: sudden changes in spending patterns, transactions during unusual hours, and purchases from merchants you don't typically use."
            ],
            'prevention': [
                "To prevent fraud: monitor your accounts regularly, use strong unique passwords, enable two-factor authentication, and report suspicious activity immediately to your bank.",
                "Protect yourself by: never sharing card details over email, using virtual cards for online purchases, and regularly checking your credit reports for unauthorized activity.",
                "Best practices include: setting up transaction alerts, using chip-enabled cards, and avoiding public Wi-Fi for financial transactions."
            ],
            'suspicious': [
                "If you suspect fraud: contact your bank immediately, change all passwords, monitor your accounts closely, and report the incident to authorities if significant amounts are involved.",
                "For suspected fraud: freeze your credit, dispute unauthorized charges within 60 days, and work with your bank to secure your accounts.",
                "When fraud is detected: change all security credentials, enable additional monitoring, and consider placing a fraud alert on your credit reports."
            ],
            'safe': [
                "Your transaction appears normal. Continue monitoring your accounts and report any unusual activity immediately.",
                "This looks like legitimate activity. Remember to regularly review your statements and set up transaction alerts for peace of mind.",
                "No red flags detected. Stay vigilant and contact your bank if you notice any unexpected changes in your account."
            ],
            'risk_score': [
                "A risk score is a numerical value (0.0 to 1.0) that indicates the probability of a transaction being fraudulent. Higher scores suggest greater risk. Our system combines multiple ML models with behavioral analysis to calculate comprehensive risk assessments.",
                "Risk scores range from 0.0 (very safe) to 1.0 (high risk). They're calculated using machine learning models that analyze transaction patterns, user behavior, location data, and historical fraud patterns. Scores above 0.7 typically trigger alerts.",
                "Our risk scoring system evaluates transactions using ensemble methods combining XGBoost, LightGBM, Random Forest, and Neural Networks. The final score incorporates behavioral factors like transaction velocity, location changes, and amount anomalies."
            ],
            'calculation': [
                "Risk scores are calculated using machine learning ensemble methods that analyze 25+ transaction features including amount, location, time, user behavior patterns, and historical data. The system uses multiple algorithms (XGBoost, LightGBM, Random Forest, Neural Networks) for comprehensive evaluation.",
                "The calculation process involves: 1) Feature engineering from transaction data, 2) Individual model predictions, 3) Ensemble probability averaging, 4) Behavioral risk boosters (location changes, velocity alerts, etc.), 5) Final risk score between 0.0-1.0.",
                "Our fraud detection algorithm processes transaction data through multiple layers: preprocessing (42 features), individual ML model predictions, ensemble probability calculation, and behavioral analysis boosts. The result is a comprehensive risk assessment."
            ],
            'models': [
                "We use an ensemble of 5 machine learning models: XGBoost, LightGBM, Random Forest, Logistic Regression, and Neural Networks. Each model analyzes different aspects of transaction patterns to provide comprehensive fraud detection.",
                "The system employs gradient boosting (XGBoost, LightGBM), tree-based (Random Forest), linear (Logistic Regression), and deep learning (Neural Network) models. Ensemble methods combine these predictions for higher accuracy.",
                "Our ML pipeline includes: XGBoost for complex pattern recognition, LightGBM for speed and accuracy, Random Forest for robust predictions, Logistic Regression for interpretability, and Neural Networks for deep pattern analysis."
            ]
        }

    def load_model(self) -> bool:
        """Initialize the chatbot (no model loading needed for rule-based)"""
        try:
            if self.loaded:
                logger.info("Chatbot already initialized")
                return True

            logger.info("Initializing rule-based chatbot...")
            self.loaded = True
            logger.info("Chatbot initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize chatbot: {e}", exc_info=True)
            self.loaded = False
            return False
    
    def generate_response(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """Generate a rule-based chatbot response"""
        try:
            if not self.loaded:
                logger.error("Chatbot not initialized")
                return "Chatbot is not available. Please try again later."

            logger.info(f"Processing message: {user_message[:50]}...")

            # Convert to lowercase for matching
            message_lower = user_message.lower()

            # Check for specific technical questions first (higher priority)
            if 'risk score' in message_lower or 'risk scoring' in message_lower:
                response = random.choice(self.fraud_responses['risk_score'])
            elif ('calculat' in message_lower or 'how is it' in message_lower or 'algorithm' in message_lower or 'process' in message_lower) and ('risk' in message_lower or 'score' in message_lower or 'fraud' in message_lower):
                response = random.choice(self.fraud_responses['calculation'])
            elif 'model' in message_lower or 'machine learning' in message_lower or 'ml' in message_lower or 'xgboost' in message_lower or 'lightgbm' in message_lower or 'neural' in message_lower or 'ensemble' in message_lower:
                response = random.choice(self.fraud_responses['models'])

            # Check for general fraud-related keywords
            elif any(word in message_lower for word in ['sign', 'symptom', 'indicator', 'red flag']) and not any(word in message_lower for word in ['prevent', 'protect']):
                response = random.choice(self.fraud_responses['signs'])
            elif any(word in message_lower for word in ['prevent', 'protect', 'avoid', 'secure']):
                response = random.choice(self.fraud_responses['prevention'])
            elif any(word in message_lower for word in ['suspicious', 'stolen', 'hack', 'breach']) or ('fraud' in message_lower and any(word in message_lower for word in ['detect', 'think', 'suspect'])):
                response = random.choice(self.fraud_responses['suspicious'])
            elif any(word in message_lower for word in ['normal', 'safe', 'okay', 'legitimate']) and not any(word in message_lower for word in ['prevent', 'protect']):
                response = random.choice(self.fraud_responses['safe'])
            else:
                # Default helpful response
                response = "I'm FraudGuard, your AI assistant for banking fraud detection. I can help you with questions about fraud signs, prevention tips, risk scores, how calculations work, and our ML models. What would you like to know?"

            logger.info(f"Generated response: {response[:100]}...")
            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return "An error occurred while generating a response. Please try again."
    
    def determine_response_type(self, response: str) -> str:
        """Determine the response type based on content"""
        response_lower = response.lower()
        if any(word in response_lower for word in ['danger', 'alert', 'risk', 'warning', 'high']):
            return 'danger' if 'high risk' in response_lower else 'warning'
        elif any(word in response_lower for word in ['safe', 'secure', 'okay', 'normal', 'fine']):
            return 'success'
        else:
            return 'info'


chatbot_handler = ChatbotHandler()
