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
        "num_teams": 0
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
        "num_teams": 0
    })
    save_submissions({})

# ============== GAME DATA ==============

SCENARIOS = {
    "credit_risk": {
        "name": "💳 Credit Risk Assessment",
        "question": "Will the borrower default on their loan?",
        "type": "classification",
        "description": "A bank needs to decide whether to approve loan applications. You must predict if borrowers will default.",
        "inputs": ["Income", "Credit Score", "Debt-to-Income Ratio", "Employment Years"],
        "target": "Default (Yes/No)",
        "correct_methods": ["logistic", "tree"],
        "best_method": "logistic"
    },
    "fraud_detection": {
        "name": "🚨 Fraud Detection",
        "question": "Is this transaction fraudulent?",
        "type": "classification",
        "description": "Detect fraudulent credit card transactions in real-time. False positives annoy customers, false negatives cost money.",
        "inputs": ["Transaction Amount", "Time of Day", "Location Mismatch", "Merchant Category"],
        "target": "Fraud (Yes/No)",
        "correct_methods": ["logistic", "tree"],
        "best_method": "tree"
    },
    "house_price": {
        "name": "🏠 House Price Prediction",
        "question": "How much is this house worth?",
        "type": "regression",
        "description": "Predict the market value of houses for mortgage lending decisions.",
        "inputs": ["Square Footage", "Bedrooms", "Location Score", "Age of Property"],
        "target": "Price ($)",
        "correct_methods": ["linear"],
        "best_method": "linear"
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

def show_team_setup():
    """Initial team setup - select number of teams"""
    st.header("👥 Setup Teams")

    st.info("Select the number of teams for this game session. Teams will be automatically created with join links.")

    num_teams = st.slider("Number of Teams", 1, 10, 5)

    # Preview team names
    st.subheader("Team Preview")
    cols = st.columns(5)
    for i in range(num_teams):
        with cols[i % 5]:
            st.write(f"🏷️ Team {TEAM_NAMES[i]}")

    if st.button("✅ Create Teams", type="primary", use_container_width=True):
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

        # Update game state
        game_state = get_game_state()
        game_state["teams_setup"] = True
        game_state["num_teams"] = num_teams
        save_game_state(game_state)

        st.success(f"Created {num_teams} teams!")
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

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Status")
        st.metric("Current Round", game_state["current_round"])
        st.metric("Status", "🟢 Active" if game_state["round_active"] else "🔴 Inactive")
        st.metric("Teams", len(teams))

        if game_state["current_scenario"]:
            scenario = SCENARIOS.get(game_state["current_scenario"], {})
            st.metric("Scenario", scenario.get("name", "Unknown"))

            # Show submission status
            if game_state["round_active"]:
                current_round = game_state["current_round"]
                submitted_teams = sum(1 for s in submissions.values() if s.get("round") == current_round)
                st.metric("Submissions", f"{submitted_teams} / {len(teams)}")

    with col2:
        st.subheader("Round Actions")

        if not game_state["round_active"]:
            scenario_choice = st.selectbox(
                "Select Scenario",
                options=list(SCENARIOS.keys()),
                format_func=lambda x: SCENARIOS[x]["name"]
            )

            if st.button("🚀 Start Round", type="primary", use_container_width=True):
                game_state["current_round"] += 1
                game_state["round_active"] = True
                game_state["current_scenario"] = scenario_choice
                game_state["round_start_time"] = datetime.now().isoformat()
                save_game_state(game_state)
                st.success(f"Round {game_state['current_round']} started!")
                st.rerun()
        else:
            st.warning("Round is active - waiting for submissions")

            if st.button("🛑 End Round", type="secondary", use_container_width=True):
                game_state["round_active"] = False
                game_state["round_end_time"] = datetime.now().isoformat()
                game_state["rounds_completed"].append({
                    "round": game_state["current_round"],
                    "scenario": game_state["current_scenario"],
                    "ended_at": datetime.now().isoformat()
                })
                save_game_state(game_state)
                st.success("Round ended!")
                st.rerun()

    # Round history
    st.divider()
    st.subheader("📜 Round History")
    if game_state.get("rounds_completed"):
        for r in reversed(game_state["rounds_completed"]):
            scenario_name = SCENARIOS.get(r["scenario"], {}).get("name", "Unknown")
            st.write(f"Round {r['round']}: {scenario_name}")
    else:
        st.info("No completed rounds yet")

def show_live_scores():
    st.header("📊 Live Leaderboard")

    teams = get_teams()
    submissions = get_submissions()

    team_scores = []
    for team_id, team in teams.items():
        total_score = 0.0
        rounds_played = 0

        for sub in submissions.values():
            if sub.get("team_id") == team_id:
                score_val = sub.get("score", 0)
                try:
                    score_val = float(score_val)
                except Exception:
                    score_val = 0.0
                total_score += score_val
                rounds_played += 1

        team_scores.append({
            "Team": team.get("name", "Unknown"),
            "Score": total_score,
            "Rounds": rounds_played,
            "Avg": round(total_score / rounds_played, 1) if rounds_played > 0 else 0.0
        })

    if not team_scores:
        st.info("No scores yet")
        return

    df = pd.DataFrame(team_scores)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"

    max_score = float(df["Score"].max()) if len(df) else 0.0
    if not np.isfinite(max_score):
        max_score = 0.0
    max_value = max(100.0, max_score)

    st.dataframe(
        df,
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

    # Check if already submitted this round
    submissions = get_submissions()
    submission_key = f"{team_id}_round_{game_state['current_round']}"

    if submission_key in submissions:
        show_submission_results(submissions[submission_key])
        return

    # Show game phases
    if "game_phase" not in st.session_state:
        st.session_state.game_phase = "scenario"

    if st.session_state.game_phase == "scenario":
        show_scenario_screen(game_state)
    elif st.session_state.game_phase == "method":
        show_method_selection()
    elif st.session_state.game_phase == "simulation":
        show_simulation_screen(game_state)
    elif st.session_state.game_phase == "reflection":
        show_reflection_screen(game_state)

def show_waiting_screen(team, game_state):
    """Show waiting screen between rounds"""
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
        st.subheader("Game Status")
        st.metric("Rounds Completed", game_state.get("current_round", 0))
        st.info("🎯 Waiting for instructor to start the next round...")

    if st.button("🔄 Check for Updates", use_container_width=True):
        st.rerun()

def show_scenario_screen(game_state):
    """Display the current scenario"""
    scenario_id = game_state["current_scenario"]
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

def show_simulation_screen(game_state):
    """Interactive model simulation"""
    scenario_id = game_state["current_scenario"]
    scenario = SCENARIOS[scenario_id]
    method_id = st.session_state.selected_method
    method = METHODS[method_id]

    st.header(f"🧪 Testing: {method['name']}")
    st.caption(f"Scenario: {scenario['name']}")

    # Check method-problem match
    if scenario["type"] != method["type"]:
        st.error(f"⚠️ You're using a **{method['type']}** method for a **{scenario['type']}** problem!")

    st.divider()

    # Different parameter controls based on scenario
    if scenario_id == "credit_risk":
        results = show_credit_risk_simulation(method_id)
    elif scenario_id == "fraud_detection":
        results = show_fraud_simulation(method_id)
    else:  # house_price
        results = show_house_price_simulation(method_id)

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

def show_reflection_screen(game_state):
    """Reflection questions before final submission"""
    st.header("💭 Reflection")

    scenario_id = game_state["current_scenario"]
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
    scenario_id = game_state["current_scenario"]
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

def show_submission_results(submission):
    """Show results after submission"""
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
