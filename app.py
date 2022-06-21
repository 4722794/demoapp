import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import app_data

st.set_page_config(page_title="🤖 AI/O",page_icon="ðŸ§Š" )

PAGES = {
    "Linear Regression": app_data,
}

st.sidebar.title('Navigator')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()



