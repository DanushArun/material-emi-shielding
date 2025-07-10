"""
Simple password authentication for Streamlit app.
"""

import streamlit as st
import hashlib
import os
from datetime import datetime, timedelta
import hmac

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password
    st.markdown("""
    <div style="
        max-width: 400px;
        margin: 100px auto;
        padding: 40px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    ">
        <h2 style="text-align: center; margin-bottom: 30px;">🔒 EMI Shielder Access</h2>
        <p style="text-align: center; color: #888; margin-bottom: 20px;">
            This application contains confidential research data.
            Please enter the access password to continue.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password",
            placeholder="Enter access password"
        )
        
        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("😕 Password incorrect. Please try again.")
            
    return False


def check_password_with_timeout(timeout_minutes=30):
    """
    Enhanced password check with session timeout.
    """
    # Check if session has timed out
    if "auth_time" in st.session_state:
        elapsed = datetime.now() - st.session_state["auth_time"]
        if elapsed > timedelta(minutes=timeout_minutes):
            # Session expired
            st.session_state["password_correct"] = False
            if "auth_time" in st.session_state:
                del st.session_state["auth_time"]
    
    # Regular password check
    is_authenticated = check_password()
    
    # Set authentication time if just authenticated
    if is_authenticated and "auth_time" not in st.session_state:
        st.session_state["auth_time"] = datetime.now()
    
    return is_authenticated


def hash_password(password: str) -> str:
    """
    Hash a password for storing.
    """
    return hashlib.sha256(password.encode()).hexdigest()


# Example multi-user authentication
def check_multi_user_password():
    """
    Multi-user authentication with different access levels.
    """
    def password_entered():
        """Check password against multiple users."""
        entered_password = st.session_state["password"]
        
        # Define users and their hashed passwords
        users = {
            "researcher": {
                "password_hash": hash_password("research123"),
                "access_level": "full"
            },
            "viewer": {
                "password_hash": hash_password("view123"),
                "access_level": "readonly"
            }
        }
        
        # Check each user
        for username, user_data in users.items():
            if user_data["password_hash"] == hash_password(entered_password):
                st.session_state["password_correct"] = True
                st.session_state["username"] = username
                st.session_state["access_level"] = user_data["access_level"]
                del st.session_state["password"]
                return
        
        st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    # Login form
    st.markdown("""
    <div style="
        max-width: 400px;
        margin: 100px auto;
        padding: 40px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    ">
        <h2 style="text-align: center; margin-bottom: 30px;">🔒 Secure Access</h2>
        <p style="text-align: center; color: #888; margin-bottom: 20px;">
            Please enter your access credentials.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.text_input(
            "Password",
            type="password",
            on_change=password_entered,
            key="password",
            placeholder="Enter password"
        )
        
        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("😕 Invalid credentials")
    
    return False