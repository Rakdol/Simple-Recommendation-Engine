import streamlit as st
import numpy as np
import pandas as pd
import requests
import json


def get_recommend_response(user_input: int) -> dict:
    endpoint = "http://127.0.0.1:5020/recommend"
    reply = requests.post(endpoint, json={"user_data": user_input})
    response = {"messages": [reply.json()]}
    return response

def get_anime_data(anime_user_id_list:list) -> dict:
    endpoint = "http://127.0.0.1:5020/anime"
    reply = requests.post(endpoint, json={"user_id_list": anime_user_id_list})
    response = {"messages": [reply.json()]}
    return response

def get_webtoon_data(recommened_webtoon_id_list:list) -> dict:
    endpoint = "http://127.0.0.1:5020/webtoon"
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

    st.markdown("### Select your ID to receive recommendations for interesting webtoons!")
    user_input = st.number_input("Insert a number (Range 0 to 353404)", min_value=0 , max_value=353404, value=5, step=1)
    st.write("The current number is ", user_input)
    if st.button("Recommend Webtoons"):
        if user_input:
            # st.write(user_input)
            response = get_recommend_response(user_input)

            data = response["messages"][0]       
            animation = get_anime_data(data["anime_key"])
            webtoon = get_webtoon_data(data["webtoon_key"])
            
            # st.markdown("# Interested Webtoon List")
            webtoon = pd.DataFrame(webtoon["messages"][0]["webtoon_data"])
            display_response(df=webtoon, columns=["Name", "Writer", "Genre", "Rating", "Summary"], title="Recommended Webtoons")
            
            anime = pd.DataFrame(animation["messages"][0]["anime_data"])
            display_response(df=anime, columns=["Name", "Producers", "Genres", "Score", "synopsis"], title="Related Animations")
            

# Run the Streamlit app
if __name__ == "__main__":
    main()
