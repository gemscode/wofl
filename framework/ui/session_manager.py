import streamlit as st

def initialize_session_state():
    session_defaults = {
        'authenticated': False,
        'auth_token': None,
        'current_thread': None,
        'new_thread_flag': True,
        'code_output': "# AI-generated code will appear here\n"
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def handle_new_thread():
    st.session_state.update({
        'current_thread': None,
        'new_thread_flag': True,
        'code_output': "# New thread started..."
    })
    st.rerun()

