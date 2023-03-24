import pytest

# importing functions from the main
from main import validatelink, get_video_id, getvideotitle, fetchcomments, preprocess, predict, getsentituberesults

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

def test_validatelink():
    assert validatelink("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert validatelink("https://www.youtube.com/embed/9bZkp7q19f0") == "https://www.youtube.com/embed/9bZkp7q19f0"
    assert validatelink("https://www.youtube.com/watch?v=WGwIb_sFMcE&list=RDWGwIb_sFMcE&start_radio=1") == "https://www.youtube.com/watch?v=WGwIb_sFMcE&list=RDWGwIb_sFMcE&start_radio=1"
    assert validatelink("https://www.google.com/watch?v=2Vv-BfVoq4g") == "INVALID URL"
    assert validatelink("https://www.amazon.com") == "INVALID URL"

def test_get_video_id():
    assert get_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert get_video_id("https://youtu.be/jNQXAC9IVRw") == "jNQXAC9IVRw"
    assert get_video_id("https://www.youtube.com/embed/9bZkp7q19f0") == "9bZkp7q19f0"
    assert get_video_id("https://m.youtube.com/watch?v=2Vv-BfVoq4g") == "2Vv-BfVoq4g"
    assert get_video_id("https://www.youtube.com/watch?v=2Vv-BfVoq4g&feature=youtu.be") == "2Vv-BfVoq4g"

    assert get_video_id("https://www.google.com/") == None
    assert get_video_id("https://www.youtube.com/") == None
    assert get_video_id("https://www.youtube.com/watch") == None
    assert get_video_id("https://www.youtube.com/watch?v=") == None
    assert get_video_id("https://www.youtube.com/watch?v=12345&feature=youtu.be") == "12345"

    
def test_getvideotitle():
    video_id = "dQw4w9WgXcQ"
    title = "Rick Astley - Never Gonna Give You Up (Official Music Video)"
    assert getvideotitle(video_id) == title

    video_id = "WGwIb_sFMcE"
    title = "Daneliya Tuleshova Sings a POWERFUL Rendition of \"Arcade\" by Duncan Laurence | AGT: All-Stars 2023"
    assert getvideotitle(video_id) == title

def test_fetchcomments():
    video_id = "jNQXAC9IVRw"
    no_of_comments = 300
    max_margin_batch_error = no_of_comments + 99
    sort_by = "Relevance"
    comments = fetchcomments(video_id, no_of_comments, sort_by)
    print(len(comments))
    assert (len(comments) == no_of_comments) or (len(comments) <= max_margin_batch_error)

    video_id = "TJ2ifmkGGus"
    no_of_comments = 300
    max_margin_batch_error = no_of_comments + 99
    sort_by = "Newest first"
    comments = fetchcomments(video_id, no_of_comments, sort_by)
    assert (len(comments) == no_of_comments) or (len(comments) <= max_margin_batch_error)


def test_preprocess():
    comments_df = pd.DataFrame({"rawcomment": ["This is a test comment.", "This is another test comment."]})
    preprocess(comments_df)
    assert comments_df["processed_text"].tolist() == ["['test', 'comment']", "['another', 'test', 'comment']"]

def test_predict():
    comments_df = pd.DataFrame({"processed_text": ["['test', 'comment']", "['another', 'test', 'comment']"]})
    predicted_df = predict(comments_df)
    assert "sentiment_predictions" in predicted_df.columns
    assert "sarcasm_predictions" in predicted_df.columns

def test_getsentituberesults():
    predicted_df = pd.DataFrame({
        "sentiment_predictions": [0, 1, 2],
        "sarcasm_predictions": [0, 1, 0]
    })
    predicted_df = getsentituberesults(predicted_df)
    assert "sentitube_results" in predicted_df.columns