

import requests


url = 'http://www.omdbapi.com/?apikey=a77df7ce&t=Avatar'
api_key = 'a77df7ce'

response = requests.request("GET", url)

testdata = response.text


# def make_url():
#     'http://www.omdbapi.com/?


# pip install --upgrade google-cloud-pubsub

import os
from google.cloud import pubsub_v1


def push_data (event_data):
    pub_sub_topic = "projects/big-data-292604/topics/movie"
    service_account_path = "./cre.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
    publisher = pubsub_v1.PublisherClient.from_service_account_file(
        service_account_path
    )
    publisher.publish(pub_sub_topic, event_data)

push_data(str.encode(testdata))


# create database on Bigquery:

# CREATE DATABASE movie_recommendation;
# CREATE TABLE movie_data(
# imdbID VARCHAR(255),
# Title VARCHAR(255),
# Year INT(25),
# Rated VARCHAR(255),
# Released VARCHAR(255),
# Runtime VARCHAR(255),
# Genre VARCHAR(255),
# Actors VARCHAR(255),
# Director VARCHAR(255),
# Writer VARCHAR(255),
# imdbrating INT(25),
# Language VARCHAR(255),
# Plot VARCHAR(255),
# Country VARCHAR(255),
# Type VARCHAR(255),
# Awards VARCHAR(255),
# PRIMARY KEY(ImdbID));
