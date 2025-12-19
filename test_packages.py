import pkg_resources
import streamlit as st

st.write("Installed packages:", [pkg.key for pkg in pkg_resources.working_set])
