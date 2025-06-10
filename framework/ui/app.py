import streamlit as st
from components import render_main_interface
from auth import handle_authentication
from session_manager import initialize_session_state
import os

def main():
    initialize_session_state()

    st.set_page_config(page_title="R&W AI Companion", layout="wide")

    inject_styles()

    if not st.session_state.get("authenticated"):
        render_auth_interface()
    else:
        render_main_interface()

def inject_styles():
    css_path = os.path.join("framework", "ui", "styles.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_auth_interface():
    st.title("R&W AI Companion")
    login_tab, register_tab = st.tabs(["Login", "Register"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if handle_authentication("login", email, password):
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    with register_tab:
        with st.form("register_form"):
            email = st.text_input("Email", key="reg_email")
            password = st.text_input("Password", type="password", key="reg_pwd")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_cpwd")
            if st.form_submit_button("Register"):
                if password != confirm_password:
                    st.error("Passwords do not match")
                elif handle_authentication("register", email, password):
                    st.success("Registration successful! Please login")
                    st.rerun()
                else:
                    st.error("Registration failed")

if __name__ == "__main__":
    main()

