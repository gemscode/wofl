import streamlit as st
from st_cookies_manager import EncryptedCookieManager
import requests
import os

# Initialize cookies
cookies = EncryptedCookieManager(
    password=os.environ.get("COOKIE_PASSWORD", "default-secret-key"),  # Static secret from .env
    prefix="rw_auth/"
)

if not cookies.ready():
    st.stop()

BASE_URL = "http://localhost:5001"
MODELS = {
    "Llama3 70B": "llama3-70b-8192",
    "Llama3 8B": "llama3-8b-8192",
    "Mixtral": "mixtral-8x7b-32768"
}

def main():
    # Redirect to login if not authenticated
    if not cookies.get('token'):
        st.switch_page("pages/login.py")

    st.title("R&W Prompt Tester")
    
    # Logout button at top-right
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Logout", type="primary"):
            cookies['token'] = ""
            cookies.save()
            st.switch_page("login.py")

    # Prompt interface
    model = st.selectbox("Select Model", options=list(MODELS.keys()))
    prompt = st.text_area("Enter your prompt:", height=150)
    
    if st.button("Submit"):
        headers = {"Authorization": f"Bearer {cookies.get('token')}"}
        payload = {
            "prompt": prompt,
            "model": MODELS[model]
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/prompt",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                st.subheader("Response:")
                st.write(response.json()['response'])
            else:
                st.error(f"Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {str(e)}")

if __name__ == '__main__':
    main()

