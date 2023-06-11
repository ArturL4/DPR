import matplotlib.pyplot as plt
import numpy as np

from utils.utils import load_lottieurl
from streamlit_lottie import st_lottie

import streamlit as st

st.set_page_config(
    page_icon="🏠",
)

st.subheader("Deep learning")


lottie_file = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_en5yMC3Ds7.json")




col1, col2 = st.columns(2)


with col1:
    st.markdown(
        """
        This project is used to concentrate the topics from the deep learning lecture from Prof. Yang from the university of Stuttgart.
        """)

with col2:
    st_lottie(lottie_file)
