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

def data_read_clean():
    data=pd.read_csv("Sample-Hotel_reviews.csv")

    data.Hotel_Address = data.Hotel_Address.str.replace("Netherlands","NL")
    data.Hotel_Address = data.Hotel_Address.str.replace("United Kingdom","UK")
    data.Hotel_Address = data.Hotel_Address.str.replace("France","FR")
    data.Hotel_Address = data.Hotel_Address.str.replace("Spain","ES")
    data.Hotel_Address = data.Hotel_Address.str.replace("Italy","IT")
    data.Hotel_Address = data.Hotel_Address.str.replace("Austria","AT")

    data["countries"]=data.Hotel_Address.apply(lambda x:x.split(' ')[-1])

    data.drop(['Additional_Number_of_Scoring','Review_Date','Reviewer_Nationality','Negative_Review','Review_Total_Negative_Word_Counts','Total_Number_of_Reviews','Positive_Review','Review_Total_Positive_Word_Counts','Total_Number_of_Reviews_Reviewer_Has_Given','Reviewer_Score','days_since_review','lat','lng'],1,inplace=True)

    def impute(column):
        column=column[0]
        if(type(column) !=list):
            return "".join(literal_eval(column))
        else:
            return column
    
    data["Tags"]=data[["Tags"]].apply(impute,axis=1)

    data['Tags']=data['Tags'].str.lower()
    data['countries']=data['countries'].str.lower()

    return data
    

def recommender(data, location,description):
    #dividing sentences into words
    description=description.lower()
    print(description)
    print(location)
    word_tokenize(description)
    #applying stopwords
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
        temp_token=word_tokenize(country["Tags"][i])
        temp_set=[word for word in temp_token if not word in stop_words]
        temp2_set=set()
        for s in temp_set:
            temp2_set.add(lemm.lemmatize(s))
        vector=temp2_set.intersection(filtered_set)
        cos.append(len(vector))
    country['similarity']=cos
    #print(data.head())
    country=country.sort_values(by='similarity',ascending=False)
    country.drop_duplicates(subset='Hotel_Name',keep='first',inplace=True)
    country.sort_values('Average_Score',ascending=False,inplace=True)
    country.reset_index(inplace=True)
    return country[["Hotel_Name","Average_Score","Hotel_Address"]].head(10)


#recommender('FR','Leisure trip')



def recommendations(data,name, cosine_similarities,):

    
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



#recommendations('The May Fair Hotel')