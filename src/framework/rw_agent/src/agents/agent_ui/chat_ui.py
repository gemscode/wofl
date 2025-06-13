import streamlit as st

st.set_page_config(page_title="R&W Chat", layout="wide")

st.title("R&W AI Companion")

# File selector
file = st.selectbox("Select Agent File", options=["agent1.py", "agent2.py"])

if st.button("🧵 New Thread"):
    st.session_state['thread'] = []

if st.button("Logout"):
    st.write("Logged out")  # Replace with real logic

# Monaco-like editor (Streamlit does not support Monaco, but you can use text_area)
code = st.text_area("AI-generated code will appear here", height=300)

if st.button("📋 Copy"):
    st.write("Copied!")  # Replace with clipboard logic

st.write("Quick Actions:")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🚀 Optimize"):
        st.session_state['prompt'] = "Optimize this code"
with col2:
    if st.button("💡 Explain"):
        st.session_state['prompt'] = "Explain this code"
with col3:
    if st.button("🐛 Debug"):
        st.session_state['prompt'] = "Debug this code"
with col4:
    if st.button("📝 Comment"):
        st.session_state['prompt'] = "Add comments"

prompt = st.text_input("Ask AI to optimize, explain, or debug your code...")

if st.button("Submit"):
    st.write(f"Prompt sent: {prompt}")

