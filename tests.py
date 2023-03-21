import pytest

# importing functions from the main
from main import get_video_id, getvideotitle, fetchcomments, preprocess, predict, getsentituberesults

import googleapiclient.discovery # youtube api
import os
from dotenv import load_dotenv # loading api keys from enviroment
import pickle # for loading models into API
import nltk # used for preprocessing of fetched comments
from collections import defaultdict
import pandas as pd

# NLTK for prerpcoessing
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn


# Loading environment variables from .env file
load_dotenv()
api_key = os.environ.get("API_KEY")
api_service_name = "youtube"
api_version = "v3"

#creating YouTube API with api key
youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = api_key)

# load saved sentiment analysis model
sentiment_model = pickle.load(open("models/sentiment-analysis-pipeline.pkl", "rb"))
print("> Sentiment Model loaded successfully!")

# load saved sarcasm analysis model
sarcasm_model = pickle.load(open("models/sarcasm-analysis-pipeline.pkl", "rb"))
print("> Sarcasm Model loaded successfully!")


def test_fetchcomments():
    video_id = "dQw4w9WgXcQ"
    no_of_comments = 188
    sort_by = "Newest first"
    comments = fetchcomments(video_id, no_of_comments, sort_by)
    assert len(comments) == no_of_comments

    video_id = "jNQXAC9IVRw"
    no_of_comments = 200
    sort_by = "Relevance"
    comments = fetchcomments(video_id, no_of_comments, sort_by)
    print(len(comments))
    assert len(comments) == no_of_comments


def test_preprocess():
    comments_df = pd.DataFrame({"rawcomment": ["This is a test comment.", "This is another test comment."]})
    preprocess(comments_df)
    assert comments_df["processed_text"].tolist() == ["['test', 'comment']", "['another', 'test', 'comment']"]

def test_predict():
    comments_df = pd.DataFrame({"processed_text": ["['test', 'comment']", "['another', 'test', 'comment']"]})
    predicted_df = predict(comments_df)
    assert "sentiment_predictions" in predicted_df.columns
    assert "sarcasm_predictions" in predicted_df.columns