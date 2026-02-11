import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("CB – Centre Back Engine")

df = pd.read_excel("data/CB_dosyasi_ready.xlsx")

st.write("CB Data Loaded")
st.write(df.head())
st.write("Columns:", list(df.columns))
