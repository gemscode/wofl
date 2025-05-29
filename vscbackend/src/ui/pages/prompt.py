# pages/prompt.py
import streamlit as st
from st_cookies_manager import EncryptedCookieManager
import requests
import os
import jwt
from datetime import datetime

cookies = EncryptedCookieManager(
    password=os.environ.get("COOKIE_PASSWORD", "default-secret-key"),
    prefix="rw_auth/",
)

if not cookies.ready():
    st.stop()

BASE_URL = "https://wolfx0.com/rw"  # Production API

def is_token_valid(token: str) -> bool:
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        exp = payload.get('exp')
        return exp and datetime.utcnow() < datetime.fromtimestamp(exp)
    except jwt.PyJWTError:
        return False

def main():
    st.title("R&W Prompt Interface")
    
    # Authentication check
    token = cookies.get('token')
    if not token or not is_token_valid(token):
        st.error("Session Expired - Please Reauthenticate")
        cookies['token'] = ""
        cookies.save()
        st.switch_page("pages/login.py")
        return

    # Logout control
    if st.button("🚪 Logout", type="primary"):
        cookies['token'] = ""
        cookies.save()
        st.rerun()

    # API interaction
    prompt = st.text_area("Enter Your Prompt:", height=150)
    if st.button("🚀 Submit"):
        try:
            response = requests.post(
                f"{BASE_URL}/prompt",
                json={"prompt": prompt},
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                st.write("### Response")
                st.write(response.json()['response'])
            else:
                st.error(f"API Error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {str(e)}")

if __name__ == '__main__':
    main()

