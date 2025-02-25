import streamlit as st
import hashlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------- USER AUTHENTICATION FUNCTIONS -----------------
def hash_password(password):
    """Return a SHA-256 hash of the password."""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    """Register a new user if the username does not exist."""
    if username in st.session_state.users:
        return False, "Username already exists."
    st.session_state.users[username] = hash_password(password)
    return True, "User registered successfully."

def login_user(username, password):
    """Verify user login credentials."""
    if username not in st.session_state.users:
        return False, "Username not found."
    if st.session_state.users[username] != hash_password(password):
        return False, "Incorrect password."
    return True, "Login successful."

def show_login_register():
    """Display login and registration options in the sidebar."""
    st.sidebar.title("User Authentication")
    auth_mode = st.sidebar.radio("Select Option", ["Login", "Register"])
    
    if auth_mode == "Login":
        st.sidebar.header("Login")
        login_username = st.sidebar.text_input("Username", key="login_username")
        login_password = st.sidebar.text_input("Password", type="password", key="login_password")
        if st.sidebar.button("Login"):
            success, message = login_user(login_username, login_password)
            if success:
                st.session_state.logged_in = True
                st.sidebar.success(message)
                st.session_state.current_page = "home"  # go to home page
                st.experimental_rerun()
            else:
                st.sidebar.error(message)
    else:
        st.sidebar.header("Register")
        reg_username = st.sidebar.text_input("Username", key="reg_username")
        reg_password = st.sidebar.text_input("Password", type="password", key="reg_password")
        if st.sidebar.button("Register"):
            success, message = register_user(reg_username, reg_password)
            if success:
                st.sidebar.success(message)
            else:
                st.sidebar.error(message)

# ----------------- PAGE NAVIGATION FUNCTIONS -----------------
def show_home_page():
    """Display the home page with navigation options."""
    st.title("Welcome to the Toxic Comment Analysis App")
    st.write("Please choose an option below:")
    
    if st.button("Toxic Comment Classification"):
        st.session_state.current_page = "classification"
        st.experimental_rerun()

def show_classification_page():
    """Display the toxic comment classification UI."""
    st.title("Toxic Comment Classification")
    st.write("Upload your CSV file (with columns 'comment' and 'toxic') below.")

    # Back button to return to home page
    if st.button("Back to Home"):
        st.session_state.current_page = "home"
        st.experimental_rerun()

    # ----------------- DATA UPLOAD -----------------
    uploaded_file = st.file_uploader("Browse CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(data.head())
    else:
        st.info("No file uploaded yet. Using a sample dataset for demonstration.")
        data = pd.DataFrame({
            'comment': [
                "I love this product, it is amazing!",
                "This is the worst experience I've ever had.",
                "Absolutely fantastic service.",
                "Horrible customer support, will never buy again.",
                "Great job, keep it up!",
                "Terrible, just terrible.",
                "Not bad, but could be improved.",
                "I hate this, awful quality.",
                "Best purchase ever!",
                "Disgusting, do not recommend."
            ],
            'toxic': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })

    # ----------------- MODEL SELECTION & PARAMETERS -----------------
    st.sidebar.header("Model Options")
    selected_models = st.sidebar.multiselect(
        "Select model(s) to train", ["Logistic Regression", "Naive Bayes"], default=["Logistic Regression"]
    )
    test_size = st.sidebar.slider("Test set size (%)", 10, 50, 20, step=5) / 100

    # ----------------- TRAINING FUNCTION -----------------
    @st.cache_data(show_spinner=False)
    def train_model(model_name, X_train, X_test, y_train, y_test):
        if model_name == "Logistic Regression":
            clf = LogisticRegression(max_iter=1000)
        elif model_name == "Naive Bayes":
            clf = MultinomialNB()
        else:
            raise ValueError("Unsupported model!")
        
        # Create a pipeline: TF-IDF vectorizer + classifier
        pipeline = make_pipeline(TfidfVectorizer(), clf)
        pipeline.fit(X_train, y_train)
        
        predictions = pipeline.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y_test, predictions)
        
        return pipeline, acc, report, conf_matrix

    # ----------------- TRAIN MODELS -----------------
    if st.sidebar.button("Train Models"):
        X = data["comment"]
        y = data["toxic"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        trained_models = {}
        metrics = {}
    
        for model_name in selected_models:
            with st.spinner(f"Training {model_name}..."):
                model_pipeline, acc, report, conf_matrix = train_model(
                    model_name, X_train, X_test, y_train, y_test
                )
                trained_models[model_name] = model_pipeline
                metrics[model_name] = {
                    "accuracy": acc,
                    "report": report,
                    "conf_matrix": conf_matrix,
                }
            st.success(f"{model_name} training completed!")
    
        # Display performance metrics for each model
        for model_name, vals in metrics.items():
            st.subheader(f"{model_name} Metrics")
            st.write(f"**Accuracy:** {vals['accuracy']:.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, trained_models[model_name].predict(X_test)))
            
            st.write("**Confusion Matrix:**")
            fig, ax = plt.subplots()
            sns.heatmap(vals["conf_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
    
        # Save the trained models in session state for later use in comment analysis
        st.session_state.trained_models = trained_models

    # ----------------- COMMENT ANALYSIS -----------------
    st.header("Analyze a Comment")
    user_comment = st.text_area("Enter a comment to analyze", "")
    
    if st.button("Analyze Comment"):
        if "trained_models" not in st.session_state:
            st.error("Please train a model first!")
        elif not user_comment:
            st.error("Please enter a comment.")
        else:
            st.write("### Predictions:")
            for model_name, pipeline in st.session_state.trained_models.items():
                prediction = pipeline.predict([user_comment])[0]
                label = "Toxic" if prediction == 1 else "Non-Toxic"
                
                # Get probability (if available)
                proba = None
                if model_name == "Logistic Regression":
                    if hasattr(pipeline.named_steps["logisticregression"], "predict_proba"):
                        proba = pipeline.predict_proba([user_comment])[0][1]
                elif model_name == "Naive Bayes":
                    if hasattr(pipeline.named_steps["multinomialnb"], "predict_proba"):
                        proba = pipeline.predict_proba([user_comment])[0][1]
                
                st.subheader(f"{model_name} Prediction")
                st.write(f"**Prediction:** {label}")
                if proba is not None:
                    st.write(f"**Probability of being toxic:** {proba:.2f}")

# ----------------- INITIALIZE SESSION STATE VARIABLES -----------------
if "users" not in st.session_state:
    st.session_state.users = {}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "login"  # default page is the login/register page

# ----------------- SIDEBAR: Logout Option -----------------
if st.session_state.logged_in:
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.current_page = "login"
        st.experimental_rerun()

# ----------------- APP MAIN LOGIC -----------------
if not st.session_state.logged_in:
    show_login_register()
    st.warning("Please login or register to access the application.")
    st.stop()
else:
    # Navigation based on the current page
    if st.session_state.current_page == "home":
        show_home_page()
    elif st.session_state.current_page == "classification":
        show_classification_page()
    else:
        st.write("Unknown page!")