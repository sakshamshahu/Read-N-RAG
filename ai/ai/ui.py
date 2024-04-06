import streamlit as st

st.title('Read N Rag')
st.write('Mine Text documents for information  💻 ⛏️ ')

if prompt := st.chat_input('Whats Up ?'):
    st.session_state['messages'].append({'role': 'user', 'content': prompt})