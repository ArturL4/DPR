from collections import namedtuple
import math
import pandas as pd
import streamlit as st
import altair as alt
from vega_datasets import data
import plotly.express as px
import numpy as np

df = px.data.gapminder().query("country=='Canada'")
fig = px.line(df, x="year", y="lifeExp", title='Life expectancy in Canada')

st.write(fig)


map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [80, 50] + [47.6169191, 7.6709248],
    columns=['lat', 'lon'])

st.map(map_data)

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)

add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)   


left_column, right_column, r = st.columns(3)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")


'Starting a long computation...'

# Add a placeholder
import time
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'