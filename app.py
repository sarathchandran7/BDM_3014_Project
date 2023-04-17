import streamlit as st
import numpy as np
import pandas as pd
from hotelRecommendation import *

# This is for a simple UI to show our Hotel Recommendations to user.
st.title('Hotel Recommendation')

#Please describe your preferenced feature here
trip =  st.text_input("Your Preferences")

#Select your country of preference
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
    #calling a function which return cleaned data
    cleaned_data = data_read_clean()
    #calling 'recommender' function in hotelRecommendation.py file by passing arguments such as 'cleaned data', preferences given by user, and country selected by user.
    recommendation=recommender(cleaned_data, selected_country,trip)
    #displaying the recommended hotel
    st.dataframe(recommendation)

# Provide your favourite hotel name and system will recommend similar hotels.
st.subheader('Mention your Favourite hotel name to show similar options')
hotelname  = st.text_input("Hotel Name")

if st.button('Show similar Hotel'):
    data = data_read_clean()
    data.set_index('Hotel_Name',inplace=True)
    
    #Converting the data in column 'Tags' to a matrix of TF-IDF features after removing stopwords.
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    feature=data['Tags'].tolist()
    tfidf_matrix = tf.fit_transform(feature)
    #Getting the cosine similarity between these two vectors.
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    #Calling 'recommendations' function in hotelRecommendation.py file. Arguments passed are 'cleaned data','name of the hotel user provided','cosine similarity'
    recommendation=recommendations(data, hotelname,cosine_similarities)
    #Showing recommendations 
    hotel=' '
    for item in recommendation:
        hotel=hotel+"\n"+item
    print(hotel)
    st.text(hotel)
