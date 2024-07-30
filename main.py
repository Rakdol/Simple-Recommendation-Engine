from typing import Optional
import streamlit as st
import numpy as np
import pandas as pd
import requests
import json

PORT = 5020

def get_query_recommend_response(query:str) -> dict:
    endpoint = f"http://127.0.0.1:{PORT}/query-recommend"
    reply = requests.post(endpoint, json={"query": query})
    response = {"messages": [reply.json()]}
    return response

def get_user_recommend_response(user_id:int, query:str) -> dict:
    endpoint = f"http://127.0.0.1:{PORT}/user-recommend"
    reply = requests.post(endpoint, json={"user_id": user_id, "query": query})
    response = {"messages": [reply.json()]}
    return response

def get_anime_data(anime_user_id_list:list) -> dict:
    endpoint = f"http://127.0.0.1:{PORT}/anime"
    reply = requests.post(endpoint, json={"user_id_list": anime_user_id_list})
    response = {"messages": [reply.json()]}
    return response

def get_webtoon_data(recommened_webtoon_id_list:list) -> dict:
    endpoint = f"http://127.0.0.1:{PORT}/webtoon"
    reply = requests.post(endpoint, json={"webtoon_id_list": recommened_webtoon_id_list})
    response = {"messages": [reply.json()]}
    return response

def display_response(df:pd.DataFrame, columns:list, title:str) -> None:
    df = df.loc[:, columns]
    st.title(title)
    st.dataframe(df)

def main():
    st.title("AI Webtoon Recommender 🤖")
    st.image("./assets/dataset-cover.jpg")
    
    
    option = st.selectbox('Choose recommendation type',('User ID based', 'Query based'))
    user_id = None
    if option == 'User ID based':
        user_id = st.number_input("Insert a number (Range 0 to 353404)", min_value=0 , max_value=353404, value=5, step=1)    
        query = st.text_input("Enter your query: ex) Recommend highly rated and subscribed action webtoon")

    query = st.text_input("Enter your query: ex) Recommend highly rated and subscribed action webtoon") if option == 'Query based' else None

    st.markdown("### Select your ID to receive recommendations for interesting webtoons!")
    

    if st.button("Recommend Webtoons"):
        if user_id is not None:
            st.write("The User id is ", user_id)
            response = get_user_recommend_response(user_id, str(query))
        else:
            response = get_query_recommend_response(query)

        data = response["messages"][0]
        webtoon = get_webtoon_data(data["webtoon_key"])
        
        st.markdown("# Interested Webtoon List")
        webtoon = pd.DataFrame(webtoon["messages"][0]["webtoon_data"])
        display_response(df=webtoon, columns=["Name", "Writer", "Genre", "Rating", "Summary"], title="Recommended Webtoons")
        
        if user_id is not None:
            animation = get_anime_data(data["anime_key"])
            anime = pd.DataFrame(animation["messages"][0]["anime_data"])
            display_response(df=anime, columns=["Name", "Producers", "Genres", "Score", "synopsis"], title="Related Animations")


# Run the Streamlit app
if __name__ == "__main__":
    main()
