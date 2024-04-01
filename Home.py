import streamlit as st
from datetime import datetime

#! data 에 관련된 것은 바뀔 때마다 페이지 전체가 refresh 된다.

today = datetime.today().strftime("%H:%M:%S")

st.title(today)

model = st.selectbox("Choose your model", ("GPT-3", "GPT-4"))

if model == "GPT-3":
    st.write("cheap")
else:
    st.write("expensive")

name = st.text_input("What is your name?")

st.write(name)

value = st.slider("temperature", min_value=0.1, max_value=1.0)

st.write(value)
