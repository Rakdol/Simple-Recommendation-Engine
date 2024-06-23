import streamlit as st
import numpy as np
import pandas as pd
import requests
import json

def main():
    st.title("AI Webtoon Recommender 🤖")
    st.image("dataset-cover.jpg")

    st.markdown("## Select the User ID for recommending some interesting webtoon!")
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
            webtoon = pd.DataFrame(webtoon["messages"][0]["webtoon_data"]).loc[:, ["Name", "Writer", "Genre", "Rating", "Summary"]]
            st.title("Recommended Webtoons")
            st.dataframe(webtoon)
            
            st.title("Related Anime List")
            anime = pd.DataFrame(animation["messages"][0]["anime_data"]).loc[:, ["Name", "Producers", "Genres", "Score", "synopsis"]]
            st.dataframe(anime)
            

def get_recommend_response(user_input: int) -> dict:
    endpoint = "http://127.0.0.1:5020/recommend"
    reply = requests.post(endpoint, json={"user_data": user_input})
    response = {"messages": [reply.json()]}
    return response

def get_anime_data(anime_user_id_list:list):
    endpoint = "http://127.0.0.1:5020/anime"
    reply = requests.post(endpoint, json={"user_id_list": anime_user_id_list})
    response = {"messages": [reply.json()]}
    return response

def get_webtoon_data(recommened_webtoon_id_list:list):
    endpoint = "http://127.0.0.1:5020/webtoon"
    reply = requests.post(endpoint, json={"webtoon_id_list": recommened_webtoon_id_list})
    response = {"messages": [reply.json()]}
    return response

def display_response(response:dict):
    pass
    

# Run the Streamlit app
if __name__ == "__main__":
    main()