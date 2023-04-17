import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#function to clean the data before processing
def data_read_clean():
    #reading the dataset
    data=pd.read_csv("Sample-Hotel_reviews.csv")
    #For simplicity, we are assigning codes for the country.
    data.Hotel_Address = data.Hotel_Address.str.replace("Netherlands","NL")
    data.Hotel_Address = data.Hotel_Address.str.replace("United Kingdom","UK")
    data.Hotel_Address = data.Hotel_Address.str.replace("France","FR")
    data.Hotel_Address = data.Hotel_Address.str.replace("Spain","ES")
    data.Hotel_Address = data.Hotel_Address.str.replace("Italy","IT")
    data.Hotel_Address = data.Hotel_Address.str.replace("Austria","AT")

    #lambda function to split the HotelAddress and get the country name to change it into corresponding country code.
    data["countries"]=data.Hotel_Address.apply(lambda x:x.split(' ')[-1])

    #dropping unwanted columns
    data.drop(['Additional_Number_of_Scoring','Review_Date','Reviewer_Nationality','Negative_Review','Review_Total_Negative_Word_Counts','Total_Number_of_Reviews','Positive_Review','Review_Total_Positive_Word_Counts','Total_Number_of_Reviews_Reviewer_Has_Given','Reviewer_Score','days_since_review','lat','lng'],1,inplace=True)

    def impute(column):
        column=column[0]
        if(type(column) !=list):
            return "".join(literal_eval(column))
        else:
            return column
    
    #modifying 'Tags' column for processing.
    #Tag Column before calling imput function-[' Leisure trip ', ' Couple ', ' Superior Doub...',....]
    data["Tags"]=data[["Tags"]].apply(impute,axis=1)
    data['Tags']=data['Tags'].str.lower()
    data['countries']=data['countries'].str.lower()
    #Tag Column after above 3 rows of code-leisure trip couple superior double or twin
    return data
    

def recommender(data, location,description):
    
    description=description.lower()
    #dividing the preferences provided by the user into words
    word_tokenize(description)
    #Removing stopwords
    stop_words=stopwords.words('english')
    #applying lemmetization
    lemm=WordNetLemmatizer()
    filtered={word for word in description if not word in stop_words}
    filtered_set=set()
    for fs in filtered:
        filtered_set.add(lemm.lemmatize(fs))

    country=data[data['countries']==location.lower()]
    country=country.set_index(np.arange(country.shape[0]))
    list1=[];list2=[];cos=[];
    for i in range(country.shape[0]):
        #Doing the same procedures like tokenizing, lemmetazing with the 'Tags' column
        temp_token=word_tokenize(country["Tags"][i])
        temp_set=[word for word in temp_token if not word in stop_words]
        temp2_set=set()
        for s in temp_set:
            temp2_set.add(lemm.lemmatize(s))
        vector=temp2_set.intersection(filtered_set)
        cos.append(len(vector))
        
    country['similarity']=cos
    #Based on the similarities with user input and dataset, it will get the hotelName, AverageScore and HotelAddress
    country=country.sort_values(by='similarity',ascending=False)
    country.drop_duplicates(subset='Hotel_Name',keep='first',inplace=True)
    country.sort_values('Average_Score',ascending=False,inplace=True)
    country.reset_index(inplace=True)
    #Result will be shown to user as Top 10 high scored Hotels.
    return country[["Hotel_Name","Average_Score","Hotel_Address"]].head(10)




def recommendations(data,name, cosine_similarities):

    
    indices = pd.Series(data.index)
    recommended_hotels = []
    
    # gettin the index of the hotel that matches the name
    idx = indices[indices == name].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar hotels except itself
    top_10_indexes = list(score_series.iloc[1:11].index)
    print(top_10_indexes)
    print()
    
    # populating the list with the names of the top 10 matching hotels
    for i in top_10_indexes:
        recommended_hotels.append(list(data.index)[i]+', '+list(data.countries)[i].upper())
        
    return recommended_hotels



