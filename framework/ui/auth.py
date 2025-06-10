import streamlit as st
import requests

def handle_authentication(action, email, password):
    """Handle both login and registration"""
    endpoint = "/rw/login" if action == "login" else "/rw/register"
    
    try:
        response = requests.post(
            f"https://www.wolfx0.com{endpoint}",
            headers={
                "Content-Type": "application/json",
                "Origin": "vscode-webview://"
            },
            json={"email": email, "password": password}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("token"):
                st.session_state.auth_token = data["token"]
                st.session_state.authenticated = True
                return True
        return False
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return False

def handle_logout():
    st.session_state.clear()
    st.experimental_rerun()

