import streamlit as st
import requests
from streamlit_ace import st_ace
from auth import handle_logout
from session_manager import handle_new_thread

def render_file_selector():
    files = get_agent_files()
    if not files:
        st.warning("No agent files found.")
        return
    selected = st.selectbox(
        "Select Agent File",
        options=files,
        format_func=lambda x: x.split("/")[-1],
        key="file_selector"
    )
    if selected != st.session_state.get("selected_file"):
        st.session_state.selected_file = selected
        st.rerun()

def get_agent_files():
    try:
        response = requests.get(
            "https://www.wolfx0.com/rw/files",
            headers={
                "Authorization": f"Bearer {st.session_state.get('auth_token', '')}",
                "Origin": "vscode-webview://"
            }
        )
        return response.json().get('files', [])
    except Exception as e:
        st.error(f"Error fetching files: {str(e)}")
        return []

def render_prompt_input(on_submit, editor_content):
    col1, col2 = st.columns([5, 1])
    with col1:
        prompt = st.text_input(
            "Ask AI to optimize, explain, or debug your code...",
            key="prompt_input"
        )
    with col2:
        if st.button("Submit", use_container_width=True):
            on_submit(prompt, editor_content)

def render_code_editor():
    code = st.session_state.get("code_output", "# AI-generated code will appear here")
    new_code = st_ace(
        value=code,
        language="python",
        theme="monokai",
        height=500,
        font_size=14,
        keybinding="vscode",
        show_gutter=True,
        show_print_margin=False,
        wrap=True,
        auto_update=True
    )
    return new_code

def handle_prompt_submission(prompt, current_code):
    if not st.session_state.get("auth_token"):
        st.error("Authentication required")
        return

    try:
        response = requests.post(
            "https://wolfx0.com/rw/prompt",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {st.session_state.auth_token}",
                "Origin": "vscode-webview://"
            },
            json={
                "prompt": prompt,
                "thread_id": st.session_state.get("current_thread"),
                "new_thread": st.session_state.get("new_thread_flag", True),
                "code": current_code
            }
        )

        if response.status_code == 200:
            data = response.json()
            st.session_state.current_thread = data.get("thread_id")
            st.session_state.new_thread_flag = False
            st.session_state.code_output = data.get("response", "")
            st.rerun()
        elif response.status_code == 401:
            handle_logout()
            st.error("Session expired - please login again")
        else:
            st.error(f"API Error: {response.json().get('error', 'Unknown error')}")

    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")

def render_main_interface():
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        render_file_selector()
    with col2:
        if st.button("🧵 New Thread"):
            handle_new_thread()
    with col3:
        if st.button("Logout", type="primary"):
            handle_logout()
            st.rerun()

    # Editor and prompt
    current_code = render_code_editor()
    render_prompt_input(handle_prompt_submission, current_code)

