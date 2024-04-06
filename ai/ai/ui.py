import streamlit as st

st.title('Read N Rag')
st.write('Mine Text documents for information  💻 ⛏️ ')

if "messages" not in st.session_state: # Initialize state history
    st.session_state["messages"] = []

if prompt := st.chat_input('What would you like to know about the text?'):
    st.session_state['messages'].append({'role': 'user', 'content': prompt})