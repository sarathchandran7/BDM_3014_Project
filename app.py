import streamlit as st
import numpy as np
import pandas as pd
from hotelRecommendation import *


st.title('Hotel Recommendation')

#st.subheader('Please describe your ideal hotel/lodging')
trip =  st.text_input("Your Preferences")

    
country= st.selectbox(
    'Select the Country :',
    ('Netherlands', 'United Kingdom','France','Spain','Italy','Austria'))

def find_country(country):
    if(country=='Netherlands'):
        selected_country='NL'
    elif(country=='United Kingdom'):
        selected_country='UK'
    elif(country=='France'):
        selected_country='FR'
    elif(country=='Spain'):
        selected_country='ES'
    elif(country=='Italy'):
        selected_country='IT'
    elif(country=='Austria'):
        selected_country='AT'
    return selected_country

selected_country=find_country(country)


if st.button('Show Some Recommendations'):
    cleaned_data = data_read_clean()
    recommendation=recommender(cleaned_data, selected_country,trip)
    st.dataframe(recommendation)


st.subheader('Mention your Favourite hotel name to show similar options')
hotelname  = st.text_input("Hotel Name")

if st.button('Show similar Hotel'):
    data = data_read_clean()
    data.set_index('Hotel_Name',inplace=True)
    #len(data['Tags'].tolist())

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    feature=data['Tags'].tolist()
    tfidf_matrix = tf.fit_transform(feature)
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    recommendation=recommendations(data, hotelname,cosine_similarities)
    hotel=' '
    for item in recommendation:
        hotel=hotel+"\n"+item
    print(hotel)
    st.text(hotel)
