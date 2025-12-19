import streamlit as st
st.write("Google key loaded:", bool(st.secrets.get("GOOGLE_API_KEY")))
