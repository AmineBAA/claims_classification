# -*- coding: utf-8 -*-
"""Poc.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TpCBT-dIPl7pTRG-zzMoY_9NqsuW0GUS
"""
import numpy as np
import sklearn
import pandas as pd
from sklearn.preprocessing import LabelEncoder



import re

import streamlit as st
import pandas as pd
import pickle

# Load your trained model
classify_claim = pickle.load(open('claims_classifier.pkl', 'rb'))

st.title('Claims Classification Tool')

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])
if uploaded_file is not None:
    df_test = pd.read_excel(uploaded_file)
    X_test = df_test.v2
    predictions = classify_claim(X_test) 
    df_test['predictions'] = predictions
    st.write(df_test)







