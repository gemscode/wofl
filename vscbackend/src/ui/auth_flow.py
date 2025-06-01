# auth_flow.py
import streamlit as st
from st_cookies_manager import EncryptedCookieManager
import requests
import os

# Initialize encrypted cookies
cookies = EncryptedCookieManager(
    password=os.environ.get("COOKIE_PASSWORD", "default-secret-key"),
    prefix="rw_auth/",
)

if not cookies.ready():
    st.stop()

BASE_URL = "https://wolfx0.com/rw"  # Production API endpoint

def auth_form(is_login=False):
    with st.form(key='auth_form'):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login" if is_login else "Register")
        return {"email": email, "password": password} if submitted else None

def main():
    st.title("R&W Authentication Gateway")
    
    # Check existing session
    if cookies.get('token'):
        st.success("✅ Authenticated Session Detected")
        if st.button("Logout"):
            cookies['token'] = ""
            cookies.save()
            st.rerun()
        st.switch_page("pages/prompt.py")
        return

    # Auth workflow
    auth_type = st.radio("Action:", ["Register", "Login"], horizontal=True)
    data = auth_form(is_login=(auth_type == "Login"))
    
    if data:
        endpoint = "/login" if auth_type == "Login" else "/register"
        try:
            response = requests.post(
                f"{BASE_URL}{endpoint}",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code in [200, 201]:
                cookies['token'] = response.json().get('token')
                cookies.save()
                st.rerun()
            else:
                st.error(f"Authentication Failed: {response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {str(e)}")

if __name__ == '__main__':
    main()

