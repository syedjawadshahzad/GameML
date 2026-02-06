"""
🎮 Choose the Model - AI in Finance Simulator
A Streamlit game for learning ML method selection in finance
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# Page configuration
st.set_page_config(
    page_title="🎮 Choose the Model - AI Finance Simulator",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize data directory
DATA_DIR = Path("game_data")
DATA_DIR.mkdir(exist_ok=True)

# File paths
TEAMS_FILE = DATA_DIR / "teams.json"
GAME_STATE_FILE = DATA_DIR / "game_state.json"
SUBMISSIONS_FILE = DATA_DIR / "submissions.json"

# Admin credentials
ADMIN_PASSWORD = "admin123"

# ============== URL DETECTION ==============

def get_base_url() -> str:
    """
    Best-effort base URL detection.
    - Works on Streamlit Cloud and most reverse proxies.
    - Falls back to localhost in local dev.
    """
    try:
        from streamlit.web.server.websocket_headers import _get_websocket_headers
        headers = _get_websocket_headers() or {}

        host = headers.get("Host")
        proto = headers.get("X-Forwarded-Proto", "http")  # streamlit cloud/proxies often set this
        if host:
            return f"{proto}://{host}"
    except Exception:
        pass

    return "http://localhost:8501"

# ============== DATA PERSISTENCE ==============

def load_json(filepath, default=None):
    """Load JSON file or return default"""
    if default is None:
        default = {}
    try:
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
    except:
        pass
    return default

def save_json(filepath, data):
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def get_teams():
    """Get all teams"""
    return load_json(TEAMS_FILE, {})

def save_teams(teams):
    """Save teams"""
    save_json(TEAMS_FILE, teams)

def get_game_state():
    """Get current game state"""
    default_state = {
        "current_round": 0,
        "round_active": False,
        "current_scenario": None,
        "round_start_time": None,
        "round_end_time": None,
        "rounds_completed": [],
        "teams_setup": False,
        "num_teams": 0,
        "total_rounds": 5,
        "rounds_per_team": 5,
        "team_rounds": {},
        "min_round_seconds": 180
    }
    return load_json(GAME_STATE_FILE, default_state)

def save_game_state(state):
    """Save game state"""
    save_json(GAME_STATE_FILE, state)

def get_submissions():
    """Get all submissions"""
    return load_json(SUBMISSIONS_FILE, {})

def save_submissions(submissions):
    """Save submissions"""
    save_json(SUBMISSIONS_FILE, submissions)

def reset_game():
    """Reset all game data"""
    save_teams({})
    save_game_state({
        "current_round": 0,
        "round_active": False,
        "current_scenario": None,
        "round_start_time": None,
        "round_end_time": None,
        "rounds_completed": [],
        "teams_setup": False,
        "num_teams": 0,
        "total_rounds": 5,
        "rounds_per_team": 5,
        "team_rounds": {},
        "min_round_seconds": 180
    })
    save_submissions({})

# ============== GAME DATA ==============

SCENARIOS = {
    # ============ CLASSIFICATION SCENARIOS ============
    "credit_risk": {
        "name": "💳 Credit Risk Assessment",
        "question": "Will the borrower default on their loan?",
        "type": "classification",
        "description": "A bank needs to decide whether to approve loan applications. You must predict if borrowers will default.",
        "inputs": ["Income", "Credit Score", "Debt-to-Income Ratio", "Employment Years"],
        "target": "Default (Yes/No)",
        "correct_methods": ["logistic", "tree"],
        "best_method": "logistic",
        "simulation": "credit_risk"
    },
    "fraud_detection": {
        "name": "🚨 Fraud Detection",
        "question": "Is this transaction fraudulent?",
        "type": "classification",
        "description": "Detect fraudulent credit card transactions in real-time. False positives annoy customers, false negatives cost money.",
        "inputs": ["Transaction Amount", "Time of Day", "Location Mismatch", "Merchant Category"],
        "target": "Fraud (Yes/No)",
        "correct_methods": ["logistic", "tree"],
        "best_method": "tree",
        "simulation": "fraud_detection"
    },
    "churn_prediction": {
        "name": "📱 Customer Churn Prediction",
        "question": "Will this customer cancel their subscription?",
        "type": "classification",
        "description": "A telecom company wants to identify customers likely to leave. Retaining a customer is 5x cheaper than acquiring a new one.",
        "inputs": ["Monthly Charges", "Tenure (months)", "Support Tickets", "Contract Type"],
        "target": "Churn (Yes/No)",
        "correct_methods": ["logistic", "tree"],
        "best_method": "logistic",
        "simulation": "churn_prediction"
    },
    "loan_approval": {
        "name": "🏦 Loan Approval Decision",
        "question": "Should this loan application be approved?",
        "type": "classification",
        "description": "A microfinance institution must decide whether to approve small business loans. Wrong approvals lead to losses; wrong rejections mean missed revenue.",
        "inputs": ["Business Revenue", "Years in Operation", "Owner Credit History", "Loan Amount Requested"],
        "target": "Approve (Yes/No)",
        "correct_methods": ["logistic", "tree"],
        "best_method": "tree",
        "simulation": "loan_approval"
    },
    "insurance_claim": {
        "name": "🛡️ Insurance Claim Fraud",
        "question": "Is this insurance claim fraudulent?",
        "type": "classification",
        "description": "An insurance company processes thousands of claims daily. Fraudulent claims cost billions per year, but flagging legitimate claims damages customer trust.",
        "inputs": ["Claim Amount", "Policy Age", "Previous Claims Count", "Incident Severity"],
        "target": "Fraudulent (Yes/No)",
        "correct_methods": ["logistic", "tree"],
        "best_method": "tree",
        "simulation": "insurance_claim"
    },
    "email_spam": {
        "name": "📧 Spam Email Detection",
        "question": "Is this email spam or legitimate?",
        "type": "classification",
        "description": "A financial firm needs to filter phishing emails from legitimate client communications. Missing a phishing email can lead to data breaches.",
        "inputs": ["Word Count", "Link Count", "Sender Reputation Score", "Urgency Keywords"],
        "target": "Spam (Yes/No)",
        "correct_methods": ["logistic", "tree"],
        "best_method": "logistic",
        "simulation": "email_spam"
    },
    "market_direction": {
        "name": "📈 Market Direction Forecast",
        "question": "Will the stock market go up or down tomorrow?",
        "type": "classification",
        "description": "A hedge fund wants to predict whether the market index will rise or fall the next trading day to adjust portfolio positions.",
        "inputs": ["Today's Return (%)", "Trading Volume", "Volatility Index (VIX)", "Sector Momentum"],
        "target": "Direction (Up/Down)",
        "correct_methods": ["logistic", "tree"],
        "best_method": "logistic",
        "simulation": "market_direction"
    },
    # ============ REGRESSION SCENARIOS ============
    "house_price": {
        "name": "🏠 House Price Prediction",
        "question": "How much is this house worth?",
        "type": "regression",
        "description": "Predict the market value of houses for mortgage lending decisions.",
        "inputs": ["Square Footage", "Bedrooms", "Location Score", "Age of Property"],
        "target": "Price ($)",
        "correct_methods": ["linear"],
        "best_method": "linear",
        "simulation": "house_price"
    },
    "revenue_forecast": {
        "name": "💰 Revenue Forecasting",
        "question": "What will next quarter's revenue be?",
        "type": "regression",
        "description": "A retail company needs to forecast quarterly revenue to plan inventory and staffing. Overestimating wastes resources; underestimating loses sales.",
        "inputs": ["Marketing Spend", "Store Count", "Season Index", "Economic Indicator"],
        "target": "Revenue ($)",
        "correct_methods": ["linear"],
        "best_method": "linear",
        "simulation": "revenue_forecast"
    },
    "salary_prediction": {
        "name": "💼 Salary Prediction",
        "question": "What salary should we offer this candidate?",
        "type": "regression",
        "description": "An HR department needs to determine competitive salary offers based on candidate profiles. Offering too little loses talent; too much hurts the budget.",
        "inputs": ["Years of Experience", "Education Level", "Skill Score", "Industry Demand Index"],
        "target": "Salary ($)",
        "correct_methods": ["linear"],
        "best_method": "linear",
        "simulation": "salary_prediction"
    },
    "insurance_premium": {
        "name": "🏥 Insurance Premium Pricing",
        "question": "How much should the insurance premium be?",
        "type": "regression",
        "description": "An insurance company must set premiums that cover expected claims while remaining competitive. Pricing too high loses customers; too low leads to losses.",
        "inputs": ["Age", "BMI", "Smoker Status", "Number of Dependents"],
        "target": "Annual Premium ($)",
        "correct_methods": ["linear"],
        "best_method": "linear",
        "simulation": "insurance_premium"
    },
    "portfolio_return": {
        "name": "📊 Portfolio Return Estimation",
        "question": "What will the expected annual return of this portfolio be?",
        "type": "regression",
        "description": "A wealth management firm needs to estimate expected returns for client portfolios to set realistic expectations and rebalance allocations.",
        "inputs": ["Equity Allocation (%)", "Bond Duration", "Risk Factor (Beta)", "Dividend Yield"],
        "target": "Expected Return (%)",
        "correct_methods": ["linear"],
        "best_method": "linear",
        "simulation": "portfolio_return"
    },
    "energy_cost": {
        "name": "⚡ Energy Cost Prediction",
        "question": "What will the monthly energy cost be for this building?",
        "type": "regression",
        "description": "A property management company needs to estimate energy costs for budgeting. Accurate predictions help negotiate better utility contracts.",
        "inputs": ["Building Size (sqft)", "Occupancy Rate", "Average Temperature", "Equipment Age"],
        "target": "Monthly Cost ($)",
        "correct_methods": ["linear"],
        "best_method": "linear",
        "simulation": "energy_cost"
    },
    "customer_lifetime_value": {
        "name": "🎯 Customer Lifetime Value",
        "question": "How much revenue will this customer generate over their lifetime?",
        "type": "regression",
        "description": "An e-commerce company wants to predict how much each customer will spend over their entire relationship to optimize acquisition and retention budgets.",
        "inputs": ["First Purchase Amount", "Visit Frequency", "Product Categories Browsed", "Referral Source Score"],
        "target": "Lifetime Value ($)",
        "correct_methods": ["linear"],
        "best_method": "linear",
        "simulation": "customer_lifetime_value"
    }
}

METHODS = {
    "linear": {
        "name": "📈 Linear Regression",
        "type": "regression",
        "description": "Predicts continuous numerical values. Best for 'how much?' questions.",
        "icon": "📈"
    },
    "logistic": {
        "name": "📊 Logistic Regression",
        "type": "classification",
        "description": "Predicts probabilities for yes/no outcomes. Best for 'will it happen?' questions.",
        "icon": "📊"
    },
    "tree": {
        "name": "🌳 Decision Tree",
        "type": "classification",
        "description": "Makes decisions through a series of yes/no questions. Good for complex classification.",
        "icon": "🌳"
    }
}

TEAM_NAMES = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon",
    "Zeta", "Eta", "Theta", "Iota", "Kappa"
]

# ============== SCORING ==============

def calculate_score(scenario_id, method_id, parameters, reflection_score=0):
    """Calculate score for a submission"""
    scenario = SCENARIOS[scenario_id]
    method = METHODS[method_id]

    score = 0
    feedback = []

    # Method selection score (0-50 points)
    if method_id in scenario["correct_methods"]:
        if method_id == scenario["best_method"]:
            score += 50
            feedback.append("✅ Excellent! You chose the optimal method for this problem.")
        else:
            score += 35
            feedback.append("✓ Good choice! This method works, but there might be a better option.")
    else:
        score += 10
        if scenario["type"] == "classification" and method["type"] == "regression":
            feedback.append("❌ Method mismatch! You used regression for a classification problem.")
        elif scenario["type"] == "regression" and method["type"] == "classification":
            feedback.append("❌ Method mismatch! You used classification for a regression problem.")
        else:
            feedback.append("❌ This method isn't ideal for this problem type.")

    # Parameter tuning score (0-30 points)
    param_score = min(30, parameters.get("param_score", 15))
    score += param_score

    # Reflection score (0-20 points)
    score += min(20, reflection_score)

    return score, feedback

# ============== SIMULATION ENGINES ==============

def simulate_credit_risk(method_id, threshold, income, credit_score, dti):
    """Simulate credit risk model"""
    np.random.seed(42)
    n_applicants = 100

    # Generate synthetic applicants
    incomes = np.random.normal(income, 20000, n_applicants)
    scores = np.random.normal(credit_score, 50, n_applicants)
    dtis = np.random.normal(dti, 10, n_applicants)

    # True default probability (hidden from students)
    true_default_prob = 0.3 - (scores - 600) / 1000 + (dtis - 30) / 200
    true_default_prob = np.clip(true_default_prob, 0.05, 0.95)
    true_defaults = np.random.binomial(1, true_default_prob)

    if method_id == "linear":
        # Linear regression gives weird predictions
        predictions = 0.5 - (scores - 650) / 500 + (dtis - 35) / 100
        # Some predictions go outside 0-1 range
        approved = predictions < threshold
        warning = "⚠️ Warning: Linear regression produced probabilities outside 0-100%!"
    elif method_id == "logistic":
        # Logistic regression gives proper probabilities
        z = -2 + (700 - scores) / 100 + (dtis - 30) / 20
        predictions = 1 / (1 + np.exp(-z))
        approved = predictions < threshold
        warning = None
    else:  # tree
        predictions = np.where(
            (scores > 680) & (dtis < 35), 0.1,
            np.where((scores > 620) | (dtis < 40), 0.35, 0.7)
        )
        approved = predictions < threshold
        warning = None

    # Calculate metrics
    approval_rate = approved.mean() * 100
    defaults_among_approved = true_defaults[approved].mean() * 100 if approved.sum() > 0 else 0

    # Profit calculation (simplified)
    revenue_per_good_loan = 5000
    loss_per_default = 20000
    good_loans = approved.sum() - (true_defaults & approved).sum()
    bad_loans = (true_defaults & approved).sum()
    profit = good_loans * revenue_per_good_loan - bad_loans * loss_per_default

    return {
        "approval_rate": approval_rate,
        "default_rate": defaults_among_approved,
        "profit": profit,
        "predictions": predictions,
        "warning": warning,
        "param_score": 15 + int((50 - abs(defaults_among_approved - 10)) / 3)
    }

def simulate_fraud_detection(method_id, sensitivity, amount_threshold, time_weight):
    """Simulate fraud detection model"""
    np.random.seed(42)
    n_transactions = 1000

    # Generate synthetic transactions
    amounts = np.random.exponential(200, n_transactions)
    hours = np.random.randint(0, 24, n_transactions)
    location_mismatch = np.random.binomial(1, 0.1, n_transactions)

    # True fraud (hidden)
    fraud_prob = 0.02 + (amounts > 500) * 0.1 + location_mismatch * 0.15 + ((hours < 5) | (hours > 22)) * 0.05
    true_fraud = np.random.binomial(1, fraud_prob)

    if method_id == "linear":
        # Linear regression for classification - bad idea
        predictions = 0.01 + (amounts / 5000) + location_mismatch * 0.3
        flagged = predictions > sensitivity
        warning = "⚠️ Linear regression isn't designed for classification tasks!"
    elif method_id == "logistic":
        z = -4 + (amounts / 300) + location_mismatch * 2 + ((hours < 5) | (hours > 22)) * time_weight
        predictions = 1 / (1 + np.exp(-z))
        flagged = predictions > sensitivity
        warning = None
    else:  # tree
        flagged = (amounts > amount_threshold) | (location_mismatch == 1) | ((hours < 5) | (hours > 22))
        predictions = flagged.astype(float)
        warning = None

    # Metrics
    true_positives = (flagged & true_fraud).sum()
    false_positives = (flagged & ~true_fraud).sum()
    false_negatives = (~flagged & true_fraud).sum()
    true_negatives = (~flagged & ~true_fraud).sum()

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # Business impact
    cost_per_fraud_missed = 1000
    cost_per_false_alarm = 50
    total_cost = false_negatives * cost_per_fraud_missed + false_positives * cost_per_false_alarm

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "precision": precision * 100,
        "recall": recall * 100,
        "total_cost": total_cost,
        "warning": warning,
        "param_score": 15 + int(min(precision, recall) * 20)
    }

def simulate_house_price(method_id, sqft_weight, bedroom_weight, location_weight):
    """Simulate house price prediction"""
    np.random.seed(42)
    n_houses = 100

    # Generate synthetic houses
    sqft = np.random.normal(2000, 500, n_houses)
    bedrooms = np.random.randint(2, 6, n_houses)
    location_score = np.random.uniform(1, 10, n_houses)
    age = np.random.randint(0, 50, n_houses)

    # True prices (hidden)
    true_prices = 100000 + sqft * 150 + bedrooms * 20000 + location_score * 30000 - age * 1000
    true_prices += np.random.normal(0, 20000, n_houses)

    if method_id == "linear":
        # Linear regression - correct choice
        predictions = 50000 + sqft * sqft_weight + bedrooms * bedroom_weight + location_score * location_weight
        warning = None
    elif method_id == "logistic":
        # Logistic regression for continuous values - wrong
        predictions = np.full(n_houses, true_prices.mean())  # Just predicts average
        warning = "⚠️ Logistic regression outputs probabilities, not continuous values!"
    else:  # tree
        # Decision tree for regression - suboptimal
        predictions = np.where(
            sqft > 2200, 450000,
            np.where(sqft > 1800, 350000, 250000)
        )
        warning = "⚠️ Decision trees create step-wise predictions, not smooth estimates."

    # Metrics
    mae = np.abs(predictions - true_prices).mean()
    mape = (np.abs(predictions - true_prices) / true_prices).mean() * 100

    return {
        "mae": mae,
        "mape": mape,
        "predictions": predictions[:10].tolist(),
        "actuals": true_prices[:10].tolist(),
        "warning": warning,
        "param_score": max(5, 30 - int(mape / 2))
    }

# --- New Classification Simulations ---

def simulate_churn_prediction(method_id, threshold, tenure_weight, charges_weight):
    """Simulate customer churn prediction"""
    np.random.seed(43)
    n = 200
    monthly_charges = np.random.uniform(20, 100, n)
    tenure = np.random.randint(1, 72, n)
    tickets = np.random.poisson(2, n)
    contract = np.random.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2])  # month-to-month, 1yr, 2yr
    true_churn_prob = 0.4 - tenure / 200 + (monthly_charges - 50) / 300 + tickets * 0.05 - contract * 0.15
    true_churn_prob = np.clip(true_churn_prob, 0.05, 0.9)
    true_churn = np.random.binomial(1, true_churn_prob)

    if method_id == "linear":
        predictions = 0.3 - tenure * tenure_weight / 10000 + monthly_charges * charges_weight / 10000
        flagged = predictions > threshold
        warning = "⚠️ Linear regression outputs can go outside 0-1 range for classification!"
    elif method_id == "logistic":
        z = -1 + (monthly_charges - 50) / 30 - tenure * tenure_weight / 500 + tickets * 0.3
        predictions = 1 / (1 + np.exp(-z))
        flagged = predictions > threshold
        warning = None
    else:
        flagged = ((tenure < 12) & (monthly_charges > 60)) | (tickets > 4)
        predictions = flagged.astype(float)
        warning = None

    tp = (flagged & true_churn).sum()
    fp = (flagged & ~true_churn.astype(bool)).sum()
    fn = (~flagged & true_churn.astype(bool)).sum()
    tn = (~flagged & ~true_churn.astype(bool)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    retention_cost = 50
    churn_cost = 500
    total_cost = fn * churn_cost + fp * retention_cost

    return {
        "true_positives": int(tp), "false_positives": int(fp),
        "false_negatives": int(fn), "true_negatives": int(tn),
        "precision": precision * 100, "recall": recall * 100,
        "total_cost": total_cost, "warning": warning,
        "param_score": 15 + int(min(precision, recall) * 20)
    }

def simulate_loan_approval(method_id, threshold, revenue_weight, history_weight):
    """Simulate loan approval"""
    np.random.seed(44)
    n = 150
    revenue = np.random.lognormal(10, 1, n)
    years_op = np.random.randint(0, 20, n)
    credit_hist = np.random.uniform(300, 850, n)
    loan_amt = np.random.lognormal(9, 0.8, n)
    ratio = loan_amt / revenue
    true_approve_prob = 0.3 + (credit_hist - 500) / 1000 + years_op / 50 - ratio * 0.3
    true_approve_prob = np.clip(true_approve_prob, 0.1, 0.9)
    true_good = np.random.binomial(1, true_approve_prob)

    if method_id == "linear":
        predictions = revenue * revenue_weight / 1e7 + credit_hist * history_weight / 10000 - ratio * 0.2
        approved = predictions > threshold
        warning = "⚠️ Linear regression isn't designed for approval decisions!"
    elif method_id == "logistic":
        z = -2 + credit_hist / 200 + years_op / 10 - ratio * 2
        predictions = 1 / (1 + np.exp(-z))
        approved = predictions > threshold
        warning = None
    else:
        approved = (credit_hist > 650) & (years_op > 2) & (ratio < 1.5)
        predictions = approved.astype(float)
        warning = None

    good_approved = (approved & true_good).sum()
    bad_approved = (approved & ~true_good.astype(bool)).sum()
    approval_rate = approved.mean() * 100
    default_rate = bad_approved / approved.sum() * 100 if approved.sum() > 0 else 0
    profit = good_approved * 3000 - bad_approved * 15000

    return {
        "approval_rate": approval_rate, "default_rate": default_rate,
        "profit": profit, "predictions": predictions,
        "warning": warning, "param_score": 15 + int((50 - abs(default_rate - 8)) / 3)
    }

def simulate_insurance_claim(method_id, threshold, amount_weight, history_weight):
    """Simulate insurance claim fraud detection"""
    np.random.seed(45)
    n = 500
    claim_amt = np.random.exponential(5000, n)
    policy_age = np.random.randint(0, 20, n)
    prev_claims = np.random.poisson(1.5, n)
    severity = np.random.choice([1, 2, 3], n, p=[0.5, 0.35, 0.15])
    fraud_prob = 0.05 + (claim_amt > 10000) * 0.15 + (prev_claims > 3) * 0.1 + (policy_age < 1) * 0.1
    true_fraud = np.random.binomial(1, np.clip(fraud_prob, 0.01, 0.8))

    if method_id == "linear":
        predictions = claim_amt * amount_weight / 100000 + prev_claims * history_weight / 20
        flagged = predictions > threshold
        warning = "⚠️ Linear regression produces values outside 0-1 for fraud classification!"
    elif method_id == "logistic":
        z = -3 + claim_amt / 8000 + prev_claims * 0.5 - policy_age * 0.1
        predictions = 1 / (1 + np.exp(-z))
        flagged = predictions > threshold
        warning = None
    else:
        flagged = (claim_amt > 10000 * (1 - amount_weight / 10)) | (prev_claims > 3) | (policy_age < 1)
        predictions = flagged.astype(float)
        warning = None

    tp = (flagged & true_fraud).sum()
    fp = (flagged & ~true_fraud.astype(bool)).sum()
    fn = (~flagged & true_fraud.astype(bool)).sum()
    tn = (~flagged & ~true_fraud.astype(bool)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    total_cost = fn * 5000 + fp * 200

    return {
        "true_positives": int(tp), "false_positives": int(fp),
        "false_negatives": int(fn), "true_negatives": int(tn),
        "precision": precision * 100, "recall": recall * 100,
        "total_cost": total_cost, "warning": warning,
        "param_score": 15 + int(min(precision, recall) * 20)
    }

def simulate_email_spam(method_id, threshold, link_weight, urgency_weight):
    """Simulate spam email detection"""
    np.random.seed(46)
    n = 300
    word_count = np.random.randint(10, 500, n)
    link_count = np.random.poisson(2, n)
    sender_rep = np.random.uniform(0, 10, n)
    urgency = np.random.uniform(0, 1, n)
    spam_prob = 0.1 + (link_count > 5) * 0.3 + (sender_rep < 3) * 0.2 + (urgency > 0.7) * 0.15
    true_spam = np.random.binomial(1, np.clip(spam_prob, 0.02, 0.9))

    if method_id == "linear":
        predictions = link_count * link_weight / 30 + urgency * urgency_weight / 5 - sender_rep / 20
        flagged = predictions > threshold
        warning = "⚠️ Linear regression isn't suited for spam classification!"
    elif method_id == "logistic":
        z = -2 + link_count * 0.3 - sender_rep * 0.3 + urgency * urgency_weight
        predictions = 1 / (1 + np.exp(-z))
        flagged = predictions > threshold
        warning = None
    else:
        flagged = (link_count > 5) | (sender_rep < 2) | (urgency > 0.8)
        predictions = flagged.astype(float)
        warning = None

    tp = (flagged & true_spam).sum()
    fp = (flagged & ~true_spam.astype(bool)).sum()
    fn = (~flagged & true_spam.astype(bool)).sum()
    tn = (~flagged & ~true_spam.astype(bool)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    total_cost = fn * 2000 + fp * 100

    return {
        "true_positives": int(tp), "false_positives": int(fp),
        "false_negatives": int(fn), "true_negatives": int(tn),
        "precision": precision * 100, "recall": recall * 100,
        "total_cost": total_cost, "warning": warning,
        "param_score": 15 + int(min(precision, recall) * 20)
    }

def simulate_market_direction(method_id, threshold, volume_weight, vix_weight):
    """Simulate market direction prediction"""
    np.random.seed(47)
    n = 250
    today_return = np.random.normal(0, 1.5, n)
    volume = np.random.lognormal(15, 0.5, n)
    vix = np.random.uniform(10, 40, n)
    momentum = np.random.normal(0, 1, n)
    up_prob = 0.52 + today_return * 0.02 - (vix - 20) * 0.005 + momentum * 0.03
    true_up = np.random.binomial(1, np.clip(up_prob, 0.2, 0.8))

    if method_id == "linear":
        predictions = 0.5 + today_return * 0.05 - vix * vix_weight / 1000
        flagged = predictions > threshold
        warning = "⚠️ Linear regression can't properly model up/down classification!"
    elif method_id == "logistic":
        z = 0.1 + today_return * 0.15 - (vix - 20) * vix_weight / 50 + momentum * 0.2
        predictions = 1 / (1 + np.exp(-z))
        flagged = predictions > threshold
        warning = None
    else:
        flagged = (today_return > 0) & (vix < 25) | (momentum > 0.5)
        predictions = flagged.astype(float)
        warning = None

    tp = (flagged & true_up).sum()
    fp = (flagged & ~true_up.astype(bool)).sum()
    fn = (~flagged & true_up.astype(bool)).sum()
    tn = (~flagged & ~true_up.astype(bool)).sum()
    accuracy = (tp + tn) / n * 100
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        "true_positives": int(tp), "false_positives": int(fp),
        "false_negatives": int(fn), "true_negatives": int(tn),
        "precision": precision * 100, "recall": recall * 100,
        "accuracy": accuracy, "total_cost": int(fp * 1000 + fn * 800),
        "warning": warning, "param_score": 15 + int(min(precision, recall) * 20)
    }

# --- New Regression Simulations ---

def simulate_revenue_forecast(method_id, marketing_weight, store_weight, season_weight):
    """Simulate revenue forecasting"""
    np.random.seed(48)
    n = 80
    marketing = np.random.uniform(50000, 500000, n)
    stores = np.random.randint(10, 100, n)
    season = np.random.uniform(0.5, 1.5, n)
    econ = np.random.normal(100, 10, n)
    true_revenue = 1e6 + marketing * 3 + stores * 50000 + season * 200000 + econ * 5000
    true_revenue += np.random.normal(0, 100000, n)

    if method_id == "linear":
        predictions = 500000 + marketing * marketing_weight + stores * store_weight + season * season_weight
        warning = None
    elif method_id == "logistic":
        predictions = np.full(n, true_revenue.mean())
        warning = "⚠️ Logistic regression predicts categories, not revenue amounts!"
    else:
        predictions = np.where(marketing > 300000, 8e6, np.where(marketing > 150000, 5e6, 3e6))
        warning = "⚠️ Decision trees create step-wise predictions, not smooth revenue estimates."

    mae = np.abs(predictions - true_revenue).mean()
    mape = (np.abs(predictions - true_revenue) / true_revenue).mean() * 100

    return {
        "mae": mae, "mape": mape,
        "predictions": predictions[:10].tolist(), "actuals": true_revenue[:10].tolist(),
        "warning": warning, "param_score": max(5, 30 - int(mape / 2))
    }

def simulate_salary_prediction(method_id, exp_weight, edu_weight, skill_weight):
    """Simulate salary prediction"""
    np.random.seed(49)
    n = 100
    experience = np.random.uniform(0, 30, n)
    education = np.random.choice([1, 2, 3, 4], n, p=[0.1, 0.3, 0.4, 0.2])
    skill = np.random.uniform(1, 10, n)
    demand = np.random.uniform(0.5, 2.0, n)
    true_salary = 30000 + experience * 3000 + education * 15000 + skill * 5000 + demand * 10000
    true_salary += np.random.normal(0, 8000, n)

    if method_id == "linear":
        predictions = 25000 + experience * exp_weight + education * edu_weight + skill * skill_weight
        warning = None
    elif method_id == "logistic":
        predictions = np.full(n, true_salary.mean())
        warning = "⚠️ Logistic regression outputs probabilities, not salary amounts!"
    else:
        predictions = np.where(experience > 15, 120000, np.where(experience > 7, 80000, 50000))
        warning = "⚠️ Decision trees create salary brackets, not precise predictions."

    mae = np.abs(predictions - true_salary).mean()
    mape = (np.abs(predictions - true_salary) / true_salary).mean() * 100

    return {
        "mae": mae, "mape": mape,
        "predictions": predictions[:10].tolist(), "actuals": true_salary[:10].tolist(),
        "warning": warning, "param_score": max(5, 30 - int(mape / 2))
    }

def simulate_insurance_premium(method_id, age_weight, bmi_weight, smoker_weight):
    """Simulate insurance premium pricing"""
    np.random.seed(50)
    n = 120
    age = np.random.randint(18, 70, n)
    bmi = np.random.normal(28, 5, n)
    smoker = np.random.binomial(1, 0.2, n)
    dependents = np.random.randint(0, 5, n)
    true_premium = 2000 + age * 100 + bmi * 50 + smoker * 8000 + dependents * 500
    true_premium += np.random.normal(0, 1500, n)

    if method_id == "linear":
        predictions = 1000 + age * age_weight + bmi * bmi_weight + smoker * smoker_weight
        warning = None
    elif method_id == "logistic":
        predictions = np.full(n, true_premium.mean())
        warning = "⚠️ Logistic regression can't predict premium amounts!"
    else:
        predictions = np.where(smoker == 1, 15000, np.where(age > 50, 8000, 5000))
        warning = "⚠️ Decision trees create premium tiers, not individualized pricing."

    mae = np.abs(predictions - true_premium).mean()
    mape = (np.abs(predictions - true_premium) / true_premium).mean() * 100

    return {
        "mae": mae, "mape": mape,
        "predictions": predictions[:10].tolist(), "actuals": true_premium[:10].tolist(),
        "warning": warning, "param_score": max(5, 30 - int(mape / 2))
    }

def simulate_portfolio_return(method_id, equity_weight, bond_weight, risk_weight):
    """Simulate portfolio return estimation"""
    np.random.seed(51)
    n = 60
    equity_alloc = np.random.uniform(10, 90, n)
    bond_dur = np.random.uniform(1, 10, n)
    beta = np.random.uniform(0.5, 2.0, n)
    div_yield = np.random.uniform(0, 6, n)
    true_return = 2 + equity_alloc * 0.08 + div_yield * 0.5 - bond_dur * 0.1 + beta * 1.5
    true_return += np.random.normal(0, 1.5, n)

    if method_id == "linear":
        predictions = 1 + equity_alloc * equity_weight / 1000 + div_yield * 0.4 + beta * risk_weight / 10
        warning = None
    elif method_id == "logistic":
        predictions = np.full(n, true_return.mean())
        warning = "⚠️ Logistic regression outputs probabilities, not return percentages!"
    else:
        predictions = np.where(equity_alloc > 60, 10, np.where(equity_alloc > 30, 7, 4))
        warning = "⚠️ Decision trees create return buckets, not smooth estimates."

    mae = np.abs(predictions - true_return).mean()
    mape = (np.abs(predictions - true_return) / np.abs(true_return)).mean() * 100

    return {
        "mae": mae, "mape": mape,
        "predictions": predictions[:10].tolist(), "actuals": true_return[:10].tolist(),
        "warning": warning, "param_score": max(5, 30 - int(mape / 2))
    }

def simulate_energy_cost(method_id, size_weight, occupancy_weight, temp_weight):
    """Simulate energy cost prediction"""
    np.random.seed(52)
    n = 100
    bldg_size = np.random.uniform(5000, 50000, n)
    occupancy = np.random.uniform(0.3, 1.0, n)
    avg_temp = np.random.normal(65, 15, n)
    equip_age = np.random.randint(0, 25, n)
    true_cost = 500 + bldg_size * 0.1 + occupancy * 2000 + np.abs(avg_temp - 68) * 30 + equip_age * 50
    true_cost += np.random.normal(0, 500, n)

    if method_id == "linear":
        predictions = 300 + bldg_size * size_weight / 1000 + occupancy * occupancy_weight + np.abs(avg_temp - 68) * temp_weight
        warning = None
    elif method_id == "logistic":
        predictions = np.full(n, true_cost.mean())
        warning = "⚠️ Logistic regression can't predict continuous energy costs!"
    else:
        predictions = np.where(bldg_size > 30000, 5000, np.where(bldg_size > 15000, 3000, 1500))
        warning = "⚠️ Decision trees create cost brackets, not precise estimates."

    mae = np.abs(predictions - true_cost).mean()
    mape = (np.abs(predictions - true_cost) / true_cost).mean() * 100

    return {
        "mae": mae, "mape": mape,
        "predictions": predictions[:10].tolist(), "actuals": true_cost[:10].tolist(),
        "warning": warning, "param_score": max(5, 30 - int(mape / 2))
    }

def simulate_customer_lifetime_value(method_id, purchase_weight, freq_weight, category_weight):
    """Simulate customer lifetime value prediction"""
    np.random.seed(53)
    n = 100
    first_purchase = np.random.lognormal(3.5, 0.8, n)
    visit_freq = np.random.uniform(1, 30, n)
    categories = np.random.randint(1, 10, n)
    referral = np.random.uniform(1, 5, n)
    true_clv = first_purchase * 5 + visit_freq * 100 + categories * 200 + referral * 300
    true_clv += np.random.normal(0, 500, n)
    true_clv = np.maximum(true_clv, 50)

    if method_id == "linear":
        predictions = first_purchase * purchase_weight / 10 + visit_freq * freq_weight + categories * category_weight
        warning = None
    elif method_id == "logistic":
        predictions = np.full(n, true_clv.mean())
        warning = "⚠️ Logistic regression can't predict lifetime value amounts!"
    else:
        predictions = np.where(visit_freq > 20, 5000, np.where(visit_freq > 10, 3000, 1000))
        warning = "⚠️ Decision trees create value tiers, not individual estimates."

    mae = np.abs(predictions - true_clv).mean()
    mape = (np.abs(predictions - true_clv) / true_clv).mean() * 100

    return {
        "mae": mae, "mape": mape,
        "predictions": predictions[:10].tolist(), "actuals": true_clv[:10].tolist(),
        "warning": warning, "param_score": max(5, 30 - int(mape / 2))
    }

# ============== GENERIC SIMULATION UI DISPATCHER ==============

def show_classification_simulation(scenario_id, method_id):
    """Generic classification simulation UI"""
    sim_funcs = {
        "credit_risk": ("credit_risk", show_credit_risk_simulation),
        "fraud_detection": ("fraud_detection", show_fraud_simulation),
    }

    if scenario_id in sim_funcs:
        return sim_funcs[scenario_id][1](method_id)

    # Generic classification UI for new scenarios
    st.subheader("🎛️ Adjust Parameters")
    scenario = SCENARIOS[scenario_id]

    if scenario_id == "churn_prediction":
        col1, col2 = st.columns(2)
        with col1:
            threshold = st.slider("Churn Threshold", 0.0, 1.0, 0.4, help="Flag customers above this probability")
            tenure_weight = st.slider("Tenure Weight", 1, 20, 10)
        with col2:
            charges_weight = st.slider("Charges Weight", 1, 20, 10)
        results = simulate_churn_prediction(method_id, threshold, tenure_weight, charges_weight)

    elif scenario_id == "loan_approval":
        col1, col2 = st.columns(2)
        with col1:
            threshold = st.slider("Approval Threshold", 0.0, 1.0, 0.5)
            revenue_weight = st.slider("Revenue Weight", 1, 20, 10)
        with col2:
            history_weight = st.slider("Credit History Weight", 1, 20, 10)
        results = simulate_loan_approval(method_id, threshold, revenue_weight, history_weight)

    elif scenario_id == "insurance_claim":
        col1, col2 = st.columns(2)
        with col1:
            threshold = st.slider("Fraud Threshold", 0.0, 1.0, 0.4)
            amount_weight = st.slider("Claim Amount Weight", 1, 10, 5)
        with col2:
            history_weight = st.slider("Claims History Weight", 1, 10, 5)
        results = simulate_insurance_claim(method_id, threshold, amount_weight, history_weight)

    elif scenario_id == "email_spam":
        col1, col2 = st.columns(2)
        with col1:
            threshold = st.slider("Spam Threshold", 0.0, 1.0, 0.5)
            link_weight = st.slider("Link Count Weight", 1, 10, 5)
        with col2:
            urgency_weight = st.slider("Urgency Weight", 0.5, 3.0, 1.5)
        results = simulate_email_spam(method_id, threshold, link_weight, urgency_weight)

    elif scenario_id == "market_direction":
        col1, col2 = st.columns(2)
        with col1:
            threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5)
            volume_weight = st.slider("Volume Weight", 1, 10, 5)
        with col2:
            vix_weight = st.slider("VIX Weight", 1, 10, 5)
        results = simulate_market_direction(method_id, threshold, volume_weight, vix_weight)
    else:
        return {"param_score": 15, "warning": None}

    # Display results
    st.subheader("📊 Results")
    if results.get("warning"):
        st.error(results["warning"])

    # For loan-approval-style results (approval_rate/default_rate/profit)
    if "approval_rate" in results:
        c1, c2, c3 = st.columns(3)
        c1.metric("Approval Rate", f"{results['approval_rate']:.1f}%")
        c2.metric("Default Rate", f"{results['default_rate']:.1f}%")
        c3.metric("Profit/Loss", f"${results['profit']:,.0f}")
        if hasattr(results.get("predictions", None), '__len__') and len(results["predictions"]) > 10:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=results["predictions"], name="Predicted Probabilities"))
            fig.update_layout(title="Distribution of Predictions")
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Confusion matrix style
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True Positives", results['true_positives'])
        c2.metric("False Positives", results['false_positives'])
        c3.metric("False Negatives", results['false_negatives'])
        c4.metric("Total Cost", f"${results['total_cost']:,.0f}")

        col_p, col_r = st.columns(2)
        col_p.metric("Precision", f"{results['precision']:.1f}%")
        col_r.metric("Recall", f"{results['recall']:.1f}%")

        fig = go.Figure(data=go.Heatmap(
            z=[[results['true_negatives'], results['false_positives']],
               [results['false_negatives'], results['true_positives']]],
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            colorscale='Blues',
            text=[[results['true_negatives'], results['false_positives']],
                  [results['false_negatives'], results['true_positives']]],
            texttemplate="%{text}", textfont={"size": 20}
        ))
        fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

    return results

def show_regression_simulation(scenario_id, method_id):
    """Generic regression simulation UI"""
    if scenario_id == "house_price":
        return show_house_price_simulation(method_id)

    st.subheader("🎛️ Adjust Parameters")
    scenario = SCENARIOS[scenario_id]

    if scenario_id == "revenue_forecast":
        c1, c2, c3 = st.columns(3)
        with c1:
            p1 = st.slider("Marketing Spend Weight", 1.0, 8.0, 3.0)
        with c2:
            p2 = st.slider("Store Count Weight ($)", 20000, 80000, 50000)
        with c3:
            p3 = st.slider("Season Weight ($)", 50000, 400000, 200000)
        results = simulate_revenue_forecast(method_id, p1, p2, p3)

    elif scenario_id == "salary_prediction":
        c1, c2, c3 = st.columns(3)
        with c1:
            p1 = st.slider("Experience Weight ($)", 1000, 6000, 3000)
        with c2:
            p2 = st.slider("Education Weight ($)", 5000, 30000, 15000)
        with c3:
            p3 = st.slider("Skill Weight ($)", 2000, 10000, 5000)
        results = simulate_salary_prediction(method_id, p1, p2, p3)

    elif scenario_id == "insurance_premium":
        c1, c2, c3 = st.columns(3)
        with c1:
            p1 = st.slider("Age Weight ($)", 30, 200, 100)
        with c2:
            p2 = st.slider("BMI Weight ($)", 10, 100, 50)
        with c3:
            p3 = st.slider("Smoker Penalty ($)", 2000, 15000, 8000)
        results = simulate_insurance_premium(method_id, p1, p2, p3)

    elif scenario_id == "portfolio_return":
        c1, c2, c3 = st.columns(3)
        with c1:
            p1 = st.slider("Equity Weight", 20, 120, 80)
        with c2:
            p2 = st.slider("Bond Weight", 1, 20, 5)
        with c3:
            p3 = st.slider("Risk Factor Weight", 5, 30, 15)
        results = simulate_portfolio_return(method_id, p1, p2, p3)

    elif scenario_id == "energy_cost":
        c1, c2, c3 = st.columns(3)
        with c1:
            p1 = st.slider("Building Size Weight", 50, 200, 100)
        with c2:
            p2 = st.slider("Occupancy Weight ($)", 500, 4000, 2000)
        with c3:
            p3 = st.slider("Temperature Weight ($)", 10, 60, 30)
        results = simulate_energy_cost(method_id, p1, p2, p3)

    elif scenario_id == "customer_lifetime_value":
        c1, c2, c3 = st.columns(3)
        with c1:
            p1 = st.slider("First Purchase Weight", 10, 80, 50)
        with c2:
            p2 = st.slider("Visit Frequency Weight ($)", 30, 200, 100)
        with c3:
            p3 = st.slider("Category Weight ($)", 50, 400, 200)
        results = simulate_customer_lifetime_value(method_id, p1, p2, p3)
    else:
        return {"param_score": 15, "warning": None, "mae": 0, "mape": 0, "predictions": [], "actuals": []}

    # Display results
    st.subheader("📊 Results")
    if results.get("warning"):
        st.error(results["warning"])

    c1, c2 = st.columns(2)
    target_label = scenario.get("target", "Value")
    c1.metric("Mean Absolute Error", f"${results['mae']:,.0f}" if "$" in target_label else f"{results['mae']:.2f}")
    c2.metric("Mean % Error", f"{results['mape']:.1f}%")

    if results.get("predictions") and results.get("actuals"):
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Predicted', x=list(range(1, 11)), y=results['predictions']))
        fig.add_trace(go.Bar(name='Actual', x=list(range(1, 11)), y=results['actuals']))
        fig.update_layout(title="Predicted vs Actual (Sample of 10)", barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    return results

# ============== ADMIN INTERFACE ==============

def show_admin_login():
    """Show admin login form"""
    st.title("🔐 Admin Login")

    password = st.text_input("Enter Admin Password", type="password")
    if st.button("Login", type="primary"):
        if password == ADMIN_PASSWORD:
            st.session_state.admin_logged_in = True
            st.rerun()
        else:
            st.error("Invalid password")

def show_admin_dashboard():
    """Show admin control panel"""
    game_state = get_game_state()

    st.title("🎮 Admin Dashboard")

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")

        if st.button("🚪 Logout"):
            st.session_state.admin_logged_in = False
            st.rerun()

        st.divider()

        if st.button("🔄 Refresh"):
            st.rerun()

        st.divider()

        if st.button("🗑️ Reset Game", type="secondary"):
            reset_game()
            st.success("Game reset!")
            st.rerun()

    # Check if teams are set up
    if not game_state.get("teams_setup"):
        show_team_setup()
    else:
        show_game_control()

def assign_random_rounds(num_teams, total_rounds, rounds_per_team):
    """Assign random scenarios to each team from the pool"""
    import random
    all_scenario_ids = list(SCENARIOS.keys())
    classification_ids = [s for s in all_scenario_ids if SCENARIOS[s]["type"] == "classification"]
    regression_ids = [s for s in all_scenario_ids if SCENARIOS[s]["type"] == "regression"]

    team_rounds = {}
    for i in range(num_teams):
        team_id = f"team_{i+1}"
        # Ensure a mix: at least half classification, half regression (roughly)
        n_class = max(1, rounds_per_team // 2)
        n_reg = rounds_per_team - n_class

        # Pick random scenarios, allow repeats only if pool is smaller than needed
        picked_class = random.sample(classification_ids, min(n_class, len(classification_ids)))
        while len(picked_class) < n_class:
            picked_class.append(random.choice(classification_ids))

        picked_reg = random.sample(regression_ids, min(n_reg, len(regression_ids)))
        while len(picked_reg) < n_reg:
            picked_reg.append(random.choice(regression_ids))

        combined = picked_class + picked_reg
        random.shuffle(combined)
        team_rounds[team_id] = combined

    return team_rounds

def show_team_setup():
    """Initial team setup - select number of teams"""
    st.header("👥 Setup Teams")

    st.info("Select the number of teams and configure rounds. Each team gets a random set of scenarios from the pool of 15 rounds.")

    num_teams = st.slider("Number of Teams", 1, 10, 5)

    st.divider()
    st.subheader("🎯 Round Configuration")

    total_rounds = st.slider(
        "Total Rounds per Game",
        min_value=3, max_value=15, value=5,
        help="How many rounds to play in total. Admin advances rounds manually."
    )

    rounds_per_team = st.slider(
        "Scenarios per Team (randomly assigned)",
        min_value=3, max_value=15, value=min(total_rounds, 5),
        help="Each team gets this many random scenarios from the pool of 15. They'll cycle through them across rounds."
    )

    if rounds_per_team > total_rounds:
        st.warning("Scenarios per team shouldn't exceed total rounds. Adjusting.")
        rounds_per_team = total_rounds

    min_round_minutes = st.slider(
        "Minimum Round Duration (minutes)",
        min_value=1, max_value=10, value=3,
        help="Each round stays visible for at least this long, even after submission."
    )

    # Preview team names
    st.divider()
    st.subheader("Team Preview")
    cols = st.columns(5)
    for i in range(num_teams):
        with cols[i % 5]:
            st.write(f"🏷️ Team {TEAM_NAMES[i]}")

    if st.button("✅ Create Teams & Assign Rounds", type="primary", use_container_width=True):
        # Create teams
        teams = {}
        for i in range(num_teams):
            team_id = f"team_{i+1}"
            teams[team_id] = {
                "name": f"Team {TEAM_NAMES[i]}",
                "created_at": datetime.now().isoformat(),
                "members": [],
                "total_score": 0
            }
        save_teams(teams)

        # Assign random rounds to each team
        team_rounds = assign_random_rounds(num_teams, total_rounds, rounds_per_team)

        # Update game state
        game_state = get_game_state()
        game_state["teams_setup"] = True
        game_state["num_teams"] = num_teams
        game_state["total_rounds"] = total_rounds
        game_state["rounds_per_team"] = rounds_per_team
        game_state["team_rounds"] = team_rounds
        game_state["min_round_seconds"] = min_round_minutes * 60
        save_game_state(game_state)

        st.success(f"Created {num_teams} teams with {rounds_per_team} random scenarios each!")
        st.rerun()

def show_game_control():
    """Main game control after teams are set up"""

    tab1, tab2, tab3, tab4 = st.tabs(["🔗 Team Links", "🎯 Round Control", "📊 Live Scores", "📋 Submissions"])

    with tab1:
        show_team_links()

    with tab2:
        show_round_control()

    with tab3:
        show_live_scores()

    with tab4:
        show_all_submissions()

def show_team_links():
    """Display all team join links"""
    st.header("🔗 Team Join Links")

    st.info("Share these links with your teams. Students just click and enter their name - no code required!")

    teams = get_teams()

    # Auto-detect base URL
    # Check for Streamlit Cloud environment
    try:
        # Get the current URL from session state or use default
        if "base_url" not in st.session_state:
            st.session_state.base_url = "http://localhost:8501"
    except:
        pass

    # URL input for customization
    st.subheader("Base URL")
    base_url = get_base_url()
    st.caption(f"Detected app URL: {base_url}")

    # If you still want an override:
    base_url = st.text_input("Base URL (override if needed)", value=base_url)

    st.session_state.base_url = base_url

    st.divider()

    # Display all team links in a nice format
    st.subheader("📋 All Team Links")

    # Create a table of links
    link_data = []
    for team_id, team in teams.items():
        full_link = f"{base_url}?join={team_id}"
        member_count = len(team.get('members', []))
        link_data.append({
            "Team": team["name"],
            "Members": member_count,
            "Join Link": full_link
        })

    # Display as copyable text blocks
    for team_id, team in teams.items():
        full_link = f"{base_url}?join={team_id}"
        member_count = len(team.get('members', []))
        status = "🟢" if member_count > 0 else "⚪"

        col1, col2, col3 = st.columns([2, 1, 4])
        with col1:
            st.write(f"{status} **{team['name']}**")
        with col2:
            st.write(f"{member_count} joined")
        with col3:
            st.code(full_link, language=None)

    st.divider()

    # Copy all links at once
    st.subheader("📝 Copy All Links")
    all_links = "\n".join([f"{teams[tid]['name']}: {base_url}?join={tid}" for tid in teams])
    st.text_area("All Links (copy this)", all_links, height=200)

    # Team status
    st.divider()
    st.subheader("👥 Team Status")

    for team_id, team in teams.items():
        with st.expander(f"🏷️ {team['name']} ({len(team.get('members', []))} members)"):
            if team.get('members'):
                for member in team['members']:
                    st.write(f"  • {member}")
            else:
                st.write("  No members yet")
            st.metric("Score", team.get('total_score', 0))

def show_round_control():
    """Control game rounds"""
    st.header("🎯 Round Control")

    game_state = get_game_state()
    teams = get_teams()
    submissions = get_submissions()
    total_rounds = game_state.get("total_rounds", 5)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Status")
        st.metric("Round", f"{game_state['current_round']} / {total_rounds}")
        st.metric("Status", "🟢 Active" if game_state["round_active"] else "🔴 Inactive")
        st.metric("Teams", len(teams))
        st.metric("Min Round Duration", f"{game_state.get('min_round_seconds', 180) // 60} min")

        if game_state["round_active"]:
            # Show timer
            start_time = datetime.fromisoformat(game_state["round_start_time"])
            elapsed = (datetime.now() - start_time).total_seconds()
            min_secs = game_state.get("min_round_seconds", 180)
            remaining = max(0, min_secs - elapsed)
            if remaining > 0:
                st.warning(f"⏱️ Min time remaining: {int(remaining // 60)}m {int(remaining % 60)}s")
            else:
                st.success("✅ Minimum time elapsed - can end round")

            # Show submission status
            current_round = game_state["current_round"]
            submitted_teams = sum(1 for s in submissions.values() if s.get("round") == current_round)
            st.metric("Submissions", f"{submitted_teams} / {len(teams)}")

    with col2:
        st.subheader("Round Actions")

        if not game_state["round_active"]:
            if game_state["current_round"] >= total_rounds:
                st.success("🏁 All rounds completed! Check Live Scores for final results.")
            else:
                next_round = game_state["current_round"] + 1
                st.info(f"Next: Round {next_round} of {total_rounds}")

                # Show what each team will get
                st.caption("Each team gets their own random scenario for this round:")
                team_rounds = game_state.get("team_rounds", {})
                for tid, team in teams.items():
                    assigned = team_rounds.get(tid, [])
                    round_idx = next_round - 1
                    if round_idx < len(assigned):
                        scenario_name = SCENARIOS.get(assigned[round_idx], {}).get("name", "Unknown")
                        st.write(f"  {team['name']}: {scenario_name}")
                    else:
                        st.write(f"  {team['name']}: (no more scenarios)")

                if st.button("🚀 Start Round", type="primary", use_container_width=True):
                    game_state["current_round"] = next_round
                    game_state["round_active"] = True
                    game_state["current_scenario"] = "mixed"  # each team gets their own
                    game_state["round_start_time"] = datetime.now().isoformat()
                    save_game_state(game_state)
                    st.success(f"Round {next_round} started!")
                    st.rerun()
        else:
            st.warning("Round is active - waiting for submissions")

            # Check minimum time
            start_time = datetime.fromisoformat(game_state["round_start_time"])
            elapsed = (datetime.now() - start_time).total_seconds()
            min_secs = game_state.get("min_round_seconds", 180)
            can_end = elapsed >= min_secs

            if not can_end:
                st.info(f"Cannot end round yet. {int((min_secs - elapsed) // 60)}m {int((min_secs - elapsed) % 60)}s remaining.")

            if st.button("🛑 End Round", type="secondary", use_container_width=True, disabled=not can_end):
                game_state["round_active"] = False
                game_state["round_end_time"] = datetime.now().isoformat()
                game_state["rounds_completed"].append({
                    "round": game_state["current_round"],
                    "scenario": "mixed",
                    "ended_at": datetime.now().isoformat()
                })
                save_game_state(game_state)
                st.success("Round ended!")
                st.rerun()

    # Team Round Assignments
    st.divider()
    st.subheader("🎲 Team Scenario Assignments")
    team_rounds = game_state.get("team_rounds", {})
    for tid, team in teams.items():
        assigned = team_rounds.get(tid, [])
        with st.expander(f"{team['name']} - {len(assigned)} scenarios"):
            for i, sid in enumerate(assigned):
                scenario_info = SCENARIOS.get(sid, {})
                status = "✅" if i < game_state["current_round"] else ("🔵" if i == game_state["current_round"] - 1 and game_state["round_active"] else "⬜")
                st.write(f"  {status} Round {i+1}: {scenario_info.get('name', sid)} ({scenario_info.get('type', '?')})")

    # Round history
    st.divider()
    st.subheader("📜 Round History")
    if game_state.get("rounds_completed"):
        for r in reversed(game_state["rounds_completed"]):
            st.write(f"Round {r['round']} completed at {r['ended_at'][:19]}")
    else:
        st.info("No completed rounds yet")

def show_live_scores():
    st.header("📊 Live Leaderboard")

    teams = get_teams()
    submissions = get_submissions()
    game_state = get_game_state()

    # Build team scores with round-by-round breakdown
    team_scores = []
    team_round_scores = {}  # {team_name: {round: score}}
    for team_id, team in teams.items():
        total_score = 0.0
        rounds_played = 0
        best_score = 0.0
        round_scores = {}

        for sub in submissions.values():
            if sub.get("team_id") == team_id:
                score_val = sub.get("score", 0)
                try:
                    score_val = float(score_val)
                except Exception:
                    score_val = 0.0
                total_score += score_val
                rounds_played += 1
                best_score = max(best_score, score_val)
                round_scores[sub.get("round", 0)] = score_val

        team_name = team.get("name", "Unknown")
        team_scores.append({
            "Team": team_name,
            "Score": total_score,
            "Rounds": rounds_played,
            "Avg": round(total_score / rounds_played, 1) if rounds_played > 0 else 0.0,
            "Best": best_score
        })
        team_round_scores[team_name] = round_scores

    if not team_scores:
        st.info("No scores yet")
        return

    df = pd.DataFrame(team_scores)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)

    # Podium display for top 3
    if len(df) >= 1:
        st.subheader("🏆 Top Teams")
        podium_cols = st.columns(min(3, len(df)))
        medals = ["🥇", "🥈", "🥉"]
        for i in range(min(3, len(df))):
            with podium_cols[i]:
                row = df.iloc[i]
                st.markdown(f"### {medals[i]} {row['Team']}")
                st.metric("Total Score", f"{row['Score']:.0f}")
                st.caption(f"Avg: {row['Avg']} | Best: {row['Best']:.0f} | Rounds: {row['Rounds']}")

    st.divider()

    # Score bar chart
    st.subheader("📊 Score Comparison")
    fig = go.Figure()

    # Sort for chart
    chart_df = df.sort_values("Score", ascending=True)

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

    fig.add_trace(go.Bar(
        y=chart_df["Team"],
        x=chart_df["Score"],
        orientation='h',
        marker_color=colors[:len(chart_df)],
        text=chart_df["Score"].apply(lambda x: f"{x:.0f}"),
        textposition='auto',
        textfont=dict(size=16, color='white')
    ))

    fig.update_layout(
        height=max(300, len(df) * 60),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Total Score",
        yaxis_title="",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Round-by-round breakdown
    total_rounds = game_state.get("current_round", 0)
    if total_rounds > 0 and any(team_round_scores.values()):
        st.divider()
        st.subheader("📈 Round-by-Round Performance")

        fig2 = go.Figure()
        for team_name, round_scores in team_round_scores.items():
            if round_scores:
                rounds = sorted(round_scores.keys())
                scores = [round_scores[r] for r in rounds]
                fig2.add_trace(go.Scatter(
                    x=[f"R{r}" for r in rounds],
                    y=scores,
                    mode='lines+markers',
                    name=team_name,
                    line=dict(width=3),
                    marker=dict(size=10)
                ))

        fig2.update_layout(
            height=400,
            yaxis_title="Score",
            xaxis_title="Round",
            yaxis=dict(range=[0, 105]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Full leaderboard table
    st.divider()
    st.subheader("📋 Full Leaderboard")
    display_df = df.copy()
    display_df.index = range(1, len(display_df) + 1)
    display_df.index.name = "Rank"

    max_score = float(df["Score"].max()) if len(df) else 0.0
    if not np.isfinite(max_score):
        max_score = 0.0
    max_value = max(100.0, max_score)

    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score",
                min_value=0.0,
                max_value=max_value
            )
        }
    )

def show_all_submissions():
    """Show all team submissions"""
    st.header("📋 All Submissions")

    submissions = get_submissions()
    teams = get_teams()

    if submissions:
        sub_list = []
        for sub_key, sub in submissions.items():
            team_name = teams.get(sub.get("team_id"), {}).get("name", "Unknown")
            scenario_name = SCENARIOS.get(sub.get("scenario"), {}).get("name", "Unknown")
            method_name = METHODS.get(sub.get("method"), {}).get("name", "Unknown")

            sub_list.append({
                "Team": team_name,
                "Round": sub.get("round", 0),
                "Scenario": scenario_name,
                "Method": method_name,
                "Score": sub.get("score", 0),
                "Time": sub.get("submitted_at", "")[:19]
            })

        df = pd.DataFrame(sub_list)
        df = df.sort_values(["Round", "Score"], ascending=[False, False])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No submissions yet")

# ============== PLAYER INTERFACE ==============

def show_team_join(team_id):
    """Show team join screen"""
    teams = get_teams()

    if team_id not in teams:
        st.error("❌ Invalid team link!")
        st.info("Please ask your instructor for a valid team link.")
        return

    team = teams[team_id]

    st.title(f"🎮 Join: {team['name']}")

    st.markdown("""
    ### Welcome to Choose the Model!
    Enter your name below to join your team.
    """)

    player_name = st.text_input("Your Name", placeholder="Enter your name...")

    if st.button("🚀 Join Team", type="primary", use_container_width=True):
        if player_name and len(player_name.strip()) > 0:
            player_name = player_name.strip()
            if player_name not in team.get("members", []):
                teams[team_id]["members"] = team.get("members", []) + [player_name]
                save_teams(teams)

            st.session_state.team_id = team_id
            st.session_state.player_name = player_name
            st.session_state.joined = True
            st.rerun()
        else:
            st.warning("Please enter your name")

    # Show current team members
    if team.get("members"):
        st.divider()
        st.subheader("Already on this team:")
        for member in team["members"]:
            st.write(f"• {member}")

def get_team_scenario_for_round(team_id, game_state):
    """Get the scenario assigned to a specific team for the current round"""
    team_rounds = game_state.get("team_rounds", {})
    assigned = team_rounds.get(team_id, [])
    round_idx = game_state["current_round"] - 1
    if round_idx < len(assigned):
        return assigned[round_idx]
    # Fallback: use first scenario in pool
    return list(SCENARIOS.keys())[0]

def get_round_time_remaining(game_state):
    """Get seconds remaining for the minimum round duration. Returns 0 if time elapsed."""
    start_str = game_state.get("round_start_time")
    if not start_str:
        return 0
    start_time = datetime.fromisoformat(start_str)
    elapsed = (datetime.now() - start_time).total_seconds()
    min_secs = game_state.get("min_round_seconds", 180)
    return max(0, min_secs - elapsed)

def show_player_game():
    """Main player game interface"""
    team_id = st.session_state.get("team_id")
    player_name = st.session_state.get("player_name")
    teams = get_teams()
    game_state = get_game_state()

    team = teams.get(team_id, {})

    # Header
    st.title("🎮 Choose the Model")
    st.caption(f"Team: **{team.get('name', 'Unknown')}** | Player: **{player_name}**")

    # Check if round is active
    if not game_state.get("round_active"):
        show_waiting_screen(team, game_state)
        return

    # Show round timer
    remaining = get_round_time_remaining(game_state)
    if remaining > 0:
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        st.info(f"⏱️ Round time remaining: **{mins}:{secs:02d}** - Take your time to analyze the problem!")

    # Get this team's scenario for the current round
    team_scenario_id = get_team_scenario_for_round(team_id, game_state)

    # Check if already submitted this round
    submissions = get_submissions()
    submission_key = f"{team_id}_round_{game_state['current_round']}"

    if submission_key in submissions:
        show_submission_results(submissions[submission_key], game_state)
        return

    # Show game phases
    if "game_phase" not in st.session_state:
        st.session_state.game_phase = "scenario"

    # Store team's scenario in session
    st.session_state.team_scenario_id = team_scenario_id

    if st.session_state.game_phase == "scenario":
        show_scenario_screen(game_state, team_scenario_id)
    elif st.session_state.game_phase == "method":
        show_method_selection()
    elif st.session_state.game_phase == "simulation":
        show_simulation_screen(game_state, team_scenario_id)
    elif st.session_state.game_phase == "reflection":
        show_reflection_screen(game_state, team_scenario_id)

def show_waiting_screen(team, game_state):
    """Show waiting screen between rounds"""
    total_rounds = game_state.get("total_rounds", 5)
    current_round = game_state.get("current_round", 0)

    if current_round >= total_rounds:
        st.header("🏁 Game Over!")
        st.balloons()
    else:
        st.header("⏳ Waiting for Next Round")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Team")
        st.metric("Team Name", team.get("name", "Unknown"))
        st.metric("Team Score", team.get("total_score", 0))

        if team.get("members"):
            st.write("**Team Members:**")
            for member in team["members"]:
                st.write(f"  • {member}")

    with col2:
        st.subheader("Game Progress")
        st.metric("Rounds Completed", f"{current_round} / {total_rounds}")

        # Progress bar
        progress = current_round / total_rounds if total_rounds > 0 else 0
        st.progress(min(1.0, progress))

        if current_round >= total_rounds:
            st.success("🏆 All rounds complete! Final scores are in.")
        else:
            st.info("🎯 Waiting for instructor to start the next round...")

    if st.button("🔄 Check for Updates", use_container_width=True):
        st.rerun()

def show_scenario_screen(game_state, team_scenario_id=None):
    """Display the current scenario"""
    scenario_id = team_scenario_id or game_state["current_scenario"]
    scenario = SCENARIOS[scenario_id]

    st.header(f"Round {game_state['current_round']}: {scenario['name']}")

    st.markdown(f"""
    ### 📋 The Problem

    > {scenario['description']}

    **Key Question:** {scenario['question']}

    **Problem Type:** {"Classification (Yes/No)" if scenario['type'] == 'classification' else "Regression (How Much?)"}
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📥 Input Variables")
        for inp in scenario["inputs"]:
            st.write(f"• {inp}")

    with col2:
        st.subheader("📤 Target Variable")
        st.write(f"**{scenario['target']}**")

    st.divider()

    if st.button("▶️ Choose My Method", type="primary", use_container_width=True):
        st.session_state.game_phase = "method"
        st.rerun()

def show_method_selection():
    """Let player choose ML method"""
    st.header("🔬 Choose Your Method")

    st.warning("⚠️ Choose carefully! Your choice will be **locked** once submitted.")

    cols = st.columns(3)

    for idx, (method_id, method) in enumerate(METHODS.items()):
        with cols[idx]:
            st.markdown(f"""
            ### {method['icon']} {method['name'].split(' ', 1)[1]}

            **Type:** {method['type'].title()}

            {method['description']}
            """)

            if st.button(f"Select", key=f"select_{method_id}", use_container_width=True):
                st.session_state.selected_method = method_id
                st.session_state.game_phase = "simulation"
                st.rerun()

def show_simulation_screen(game_state, team_scenario_id=None):
    """Interactive model simulation"""
    scenario_id = team_scenario_id or game_state["current_scenario"]
    scenario = SCENARIOS[scenario_id]
    method_id = st.session_state.selected_method
    method = METHODS[method_id]

    st.header(f"🧪 Testing: {method['name']}")
    st.caption(f"Scenario: {scenario['name']}")

    # Check method-problem match
    if scenario["type"] != method["type"]:
        st.error(f"⚠️ You're using a **{method['type']}** method for a **{scenario['type']}** problem!")

    st.divider()

    # Route to the right simulation UI
    if scenario["type"] == "classification":
        results = show_classification_simulation(scenario_id, method_id)
    else:
        results = show_regression_simulation(scenario_id, method_id)

    # Store results
    st.session_state.simulation_results = results

    st.divider()

    # Method is locked - no going back
    st.warning(f"🔒 Method locked: **{method['name']}** - You cannot change your selection.")

    if st.button("▶️ Submit & Reflect", type="primary", use_container_width=True):
        st.session_state.game_phase = "reflection"
        st.rerun()

def show_credit_risk_simulation(method_id):
    """Credit risk interactive simulation"""
    st.subheader("🎛️ Adjust Parameters")

    col1, col2 = st.columns(2)

    with col1:
        threshold = st.slider(
            "Approval Threshold",
            0.0, 1.0, 0.3,
            help="Approve loans where default probability is below this threshold"
        )
        income = st.slider("Average Income ($)", 30000, 150000, 75000)

    with col2:
        credit_score = st.slider("Average Credit Score", 500, 800, 680)
        dti = st.slider("Avg Debt-to-Income Ratio (%)", 10, 60, 35)

    results = simulate_credit_risk(method_id, threshold, income, credit_score, dti)

    st.subheader("📊 Results")

    if results["warning"]:
        st.error(results["warning"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Approval Rate", f"{results['approval_rate']:.1f}%")
    col2.metric("Default Rate", f"{results['default_rate']:.1f}%")
    col3.metric("Profit/Loss", f"${results['profit']:,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=results["predictions"], name="Predicted Default Prob"))
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
    fig.update_layout(title="Distribution of Predicted Default Probabilities")
    st.plotly_chart(fig, use_container_width=True)

    return results

def show_fraud_simulation(method_id):
    """Fraud detection interactive simulation"""
    st.subheader("🎛️ Adjust Parameters")

    col1, col2 = st.columns(2)

    with col1:
        sensitivity = st.slider(
            "Detection Sensitivity",
            0.0, 1.0, 0.5,
            help="Higher = flag more transactions"
        )
        amount_threshold = st.slider("Amount Threshold ($)", 100, 1000, 500)

    with col2:
        time_weight = st.slider("Time-of-Day Weight", 0.0, 3.0, 1.0)

    results = simulate_fraud_detection(method_id, sensitivity, amount_threshold, time_weight)

    st.subheader("📊 Results")

    if results["warning"]:
        st.error(results["warning"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("True Positives", results['true_positives'])
    col2.metric("False Positives", results['false_positives'])
    col3.metric("False Negatives", results['false_negatives'])
    col4.metric("Total Cost", f"${results['total_cost']:,.0f}")

    fig = go.Figure(data=go.Heatmap(
        z=[[results['true_negatives'], results['false_positives']],
           [results['false_negatives'], results['true_positives']]],
        x=['Predicted Normal', 'Predicted Fraud'],
        y=['Actual Normal', 'Actual Fraud'],
        colorscale='Blues',
        text=[[results['true_negatives'], results['false_positives']],
              [results['false_negatives'], results['true_positives']]],
        texttemplate="%{text}",
        textfont={"size": 20}
    ))
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

    return results

def show_house_price_simulation(method_id):
    """House price prediction simulation"""
    st.subheader("🎛️ Adjust Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        sqft_weight = st.slider("Sq Ft Weight ($)", 50, 300, 150)
    with col2:
        bedroom_weight = st.slider("Bedroom Weight ($)", 5000, 50000, 20000)
    with col3:
        location_weight = st.slider("Location Weight ($)", 10000, 50000, 30000)

    results = simulate_house_price(method_id, sqft_weight, bedroom_weight, location_weight)

    st.subheader("📊 Results")

    if results["warning"]:
        st.error(results["warning"])

    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error", f"${results['mae']:,.0f}")
    col2.metric("Mean % Error", f"{results['mape']:.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Predicted', x=list(range(1, 11)), y=results['predictions']))
    fig.add_trace(go.Bar(name='Actual', x=list(range(1, 11)), y=results['actuals']))
    fig.update_layout(title="Predicted vs Actual Prices (Sample of 10 Houses)", barmode='group')
    st.plotly_chart(fig, use_container_width=True)

    return results

def show_reflection_screen(game_state, team_scenario_id=None):
    """Reflection questions before final submission"""
    st.header("💭 Reflection")

    scenario_id = team_scenario_id or game_state["current_scenario"]
    scenario = SCENARIOS[scenario_id]
    method_id = st.session_state.selected_method
    method = METHODS[method_id]

    st.info(f"You chose **{method['name']}** for **{scenario['name']}**")

    q1 = st.radio(
        "1. Was your chosen method appropriate for this problem?",
        ["Yes, it was the best choice", "It worked but might not be optimal", "No, it was a mismatch"],
        index=None
    )

    q2 = st.radio(
        "2. What type of output does this problem require?",
        ["A probability (0-100%)", "A continuous number (how much)", "A category (which group)"],
        index=None
    )

    q3 = st.text_area(
        "3. Would a bank prefer this model? Why or why not?",
        placeholder="Consider accuracy, explainability, and business impact..."
    )

    # Calculate reflection score
    reflection_score = 0

    if q1:
        correct_answer_q1 = "Yes, it was the best choice" if method_id == scenario["best_method"] else \
                           "It worked but might not be optimal" if method_id in scenario["correct_methods"] else \
                           "No, it was a mismatch"
        if q1 == correct_answer_q1:
            reflection_score += 7

    if q2:
        correct_answer_q2 = "A probability (0-100%)" if scenario["type"] == "classification" else "A continuous number (how much)"
        if q2 == correct_answer_q2:
            reflection_score += 7

    if q3 and len(q3) > 20:
        reflection_score += 6

    st.session_state.reflection_score = reflection_score
    st.session_state.reflection_answers = {"q1": q1, "q2": q2, "q3": q3}

    st.divider()

    if st.button("🔒 Submit Final Answer", type="primary", use_container_width=True):
        submit_answer(game_state)

def submit_answer(game_state):
    """Lock in the team's answer"""
    team_id = st.session_state.team_id
    scenario_id = st.session_state.get("team_scenario_id") or game_state["current_scenario"]
    method_id = st.session_state.selected_method
    results = st.session_state.simulation_results
    reflection_score = st.session_state.get("reflection_score", 0)

    score, feedback = calculate_score(scenario_id, method_id, results, reflection_score)

    submissions = get_submissions()
    submission_key = f"{team_id}_round_{game_state['current_round']}"

    submissions[submission_key] = {
        "team_id": team_id,
        "round": game_state["current_round"],
        "scenario": scenario_id,
        "method": method_id,
        "score": score,
        "feedback": feedback,
        "reflection_answers": st.session_state.get("reflection_answers", {}),
        "submitted_at": datetime.now().isoformat(),
        "submitted_by": st.session_state.get("player_name", "Unknown")
    }

    save_submissions(submissions)

    teams = get_teams()
    if team_id in teams:
        teams[team_id]["total_score"] = teams[team_id].get("total_score", 0) + score
        save_teams(teams)

    st.session_state.game_phase = "scenario"
    st.rerun()

def show_submission_results(submission, game_state=None):
    """Show results after submission - stays visible for minimum round duration"""

    # Show timer if round is still within minimum duration
    if game_state:
        remaining = get_round_time_remaining(game_state)
        if remaining > 0:
            mins = int(remaining // 60)
            secs = int(remaining % 60)
            st.warning(f"⏱️ Round still active for **{mins}:{secs:02d}** - Review your results and learn from the feedback!")

    st.header("✅ Round Complete!")

    scenario = SCENARIOS.get(submission["scenario"], {})
    method = METHODS.get(submission["method"], {})

    st.success("Your answer has been locked in!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Choices")
        st.write(f"**Scenario:** {scenario.get('name', 'Unknown')}")
        st.write(f"**Method:** {method.get('name', 'Unknown')}")
        st.write(f"**Submitted by:** {submission.get('submitted_by', 'Unknown')}")

    with col2:
        st.subheader("Score")
        st.metric("Points Earned", f"{submission['score']} / 100")

        # Score breakdown visual
        score = submission.get("score", 0)
        if score >= 80:
            st.success("🌟 Excellent performance!")
        elif score >= 60:
            st.info("👍 Good job! Room for improvement.")
        elif score >= 40:
            st.warning("📚 Review the feedback below to improve.")
        else:
            st.error("💡 Study the best approach explanation below.")

    st.subheader("📝 Feedback")
    for fb in submission.get("feedback", []):
        st.write(fb)

    st.subheader("💡 The Best Approach")
    best_method = scenario.get("best_method", "")
    if best_method:
        best = METHODS.get(best_method, {})
        st.info(f"For {scenario.get('name', 'this problem')}, the optimal method is **{best.get('name', 'Unknown')}** because {scenario.get('type', 'this')} problems require {'probability outputs' if scenario.get('type') == 'classification' else 'continuous value predictions'}.")

    st.divider()
    st.info("⏳ Waiting for the next round to begin...")

    if st.button("🔄 Check for Next Round"):
        st.rerun()

# ============== MAIN APP ==============

def main():
    """Main application entry point"""

    query_params = st.query_params

    # Handle team join links
    if "join" in query_params:
        team_id = query_params["join"]
        if not st.session_state.get("joined"):
            show_team_join(team_id)
            return
        else:
            show_player_game()
            return

    # Handle admin access
    if "admin" in query_params:
        if not st.session_state.get("admin_logged_in"):
            show_admin_login()
        else:
            show_admin_dashboard()
        return

    # Default: Show home page
    show_home_page()

def show_home_page():
    """Show the landing page"""
    st.title("🎮 Choose the Model")
    st.subheader("AI in Finance Simulator")

    st.markdown("""
    Welcome to the Machine Learning Finance Game! Learn which ML methods work best for different financial problems.

    ### 🎯 How to Play

    1. **Join your team** using the link provided by your instructor
    2. **Read the scenario** - understand the financial problem
    3. **Choose a method** - Linear Regression, Logistic Regression, or Decision Tree
    4. **Test your model** - adjust parameters and see results
    5. **Reflect** - answer questions about your choice
    6. **Submit** - lock in your answer and earn points!

    ### 📊 Scoring
    - **Method Selection:** 0-50 points (choosing the right approach)
    - **Parameter Tuning:** 0-30 points (optimizing your model)
    - **Reflection:** 0-20 points (understanding why)

    ---

    ### 🔗 Quick Links
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**For Students:**")
        st.write("Ask your instructor for the team join link")

    with col2:
        st.markdown("**For Instructors:**")
        if st.button("🔐 Admin Dashboard"):
            st.query_params["admin"] = "true"
            st.rerun()

if __name__ == "__main__":
    main()
