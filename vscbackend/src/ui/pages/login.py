import streamlit as st
from st_cookies_manager import EncryptedCookieManager
import requests
import os

cookies = EncryptedCookieManager(
    password=os.environ.get("COOKIE_PASSWORD", "default-secret-key"),
    prefix="rw_auth/"
)
if not cookies.ready():
    st.stop()

BASE_URL = "http://localhost:5001"

def auth_form(is_login=False):
    with st.form(key='auth_form'):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login" if is_login else "Register")
        return {"email": email, "password": password} if submitted else None

def main():
    st.title("R&W Login/Register")
    auth_type = st.radio("Action:", ["Register", "Login"], horizontal=True)
    data = auth_form(is_login=(auth_type == "Login"))
    if data:
        endpoint = "/login" if auth_type == "Login" else "/register"
        try:
            response = requests.post(
                f"{BASE_URL}{endpoint}",
                json=data,
                timeout=5
            )
            if response.status_code in [200, 201]:
                cookies['token'] = response.json().get('token')
                cookies.save()
                st.success("Login successful! Redirecting...")
                st.rerun()
            else:
                try:
                    error_msg = response.json().get('error', 'Unknown error')
                except Exception:
                    error_msg = f"Non-JSON response: {response.text[:200]}..."
                st.error(f"Error: {error_msg}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {str(e)}")
    # If already logged in, redirect to prompt page
    if cookies.get('token'):
        st.switch_page("pages/prompt.py")

if __name__ == '__main__':
    main()

