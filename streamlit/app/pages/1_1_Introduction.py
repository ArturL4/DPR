import streamlit as st
import os

# Abs path to current directory
dirname = os.path.join(os.path.dirname(__file__), "..")

st.header("Deep learning")
st.markdown("So before starting with our journey of deep learning we want to clarify some terms.")
st.markdown("Machine learning, deep learning, artificial intelligence are terms you hear all the time in the current times.")
st.markdown("""But what are the meanings of those?""")

#######################
st.subheader("What is machine learning?")
st.markdown("""To understand what machine learning is we have to consider the simple case of Signal processing.""")
st.markdown("""In signal processing your process a given :red[input signal] according to some :red[rule] and return the corresponding :red[output signal].""")
sp_ml = os.path.join(dirname, "images/introduction/sp_ml.png")
st.image(sp_ml)
st.markdown("""Now what is the difference between normal signal processing and machine learning?""")
st.markdown("""
The answer is quite simple. While your task for normal signal processing is to design some chained rules with a priori domain knowledge to receive your output, this is also called model-based. ML on the other hand is a data-driven approach.
With bigger and bigger data volume available it makes sense to design algorithms in such a way to optimize themselves using examples in form of data.

Data-driven approaches are mostly used if the relationship between input and output is too complicated to capture through regular SP domain knowledge.
""")
st.markdown("---")
#######################
st.subheader("What is deep learning?")
st.markdown("""Now what exactly is deep learning than? 
Deep learning is a :red[subfield of machine learning] which tries to mimic the human brain to get a high level of abstraction of given data.

Deep learning is an end-2-end learning algorithm which also include feature extraction from preprocessed data, which is generally not the case for machine learning.
""")
dl_ml_image = os.path.join(dirname, "images/introduction/dl_ml.png")
st.image(dl_ml_image)
st.markdown("---")
#######################
st.subheader("Examples")