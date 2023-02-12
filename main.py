from flask import Flask, jsonify, request
import requests
import os
import googleapiclient.discovery
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import urlparse # for formatting yt url
import pickle

from collections import defaultdict
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

# load saved sentiment model
sentiment_model = pickle.load(open("models/sentiment-analysis-pipeline.pkl", "rb"))
print("> Sentiment Model loaded successfully!")

app = Flask(__name__)


@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})
    
@app.route('/analyse')
def analyse():
    # Youtube Video ID from URL
        # enter youtube url here... 
    youtube_url = "https://www.youtube.com/watch?v=38sG0OWPcRI"

        # getting the video id from the youtube url
    url_data = urlparse(youtube_url)
    video_id = url_data.query[2::]
    print("video_id: ", video_id)

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = api_key)
    comments = []

    # Function to load comments
    def load_comments(match):
        for item in match["items"]:
            comment = item["snippet"]["topLevelComment"]
            text = comment["snippet"]["textDisplay"]
            print(text)
            comments.append(text)
            
    # Function to get comments from subsequent comment pages
    def get_comment_threads(youtube, video_id, nextPageToken):
        results = youtube.commentThreads().list(
            part="snippet",
            maxResults=100,
            videoId=video_id,
            textFormat="plainText",
            pageToken = nextPageToken
        ).execute()
        return results

    try:
        match = get_comment_threads(youtube, video_id, '')
        load_comments(match)
        next_page_token = match["nextPageToken"] # if the video has less than 100 top level comments this returns a keyerror
    except:
        data = pd.DataFrame(comments, columns=["rawcomment"])
        # data.to_csv("temp/temp_comments.csv", encoding="utf-8")
        
    try:
        while next_page_token and len(comments) < 250: # used to reduce waiting time. if the video has a lot of comments the waiting time will be massive
            match = get_comment_threads(youtube, video_id, next_page_token)
            next_page_token = match["nextPageToken"]  # if the video has less than 100 top level comments this returns a keyerror
            load_comments(match)
        data = pd.DataFrame(comments, columns=["rawcomment"])
        # data.to_csv("temp/temp_comments.csv", encoding="utf-8")
    except:
        data = pd.DataFrame(comments, columns=["rawcomment"])
        # data.to_csv("temp/temp_comments.csv", encoding="utf-8")

    # copying rawcomments to a new column for preprocessing
    data['processed_text'] = data['rawcomment']
    # Step - a : Remove blank rows if any.
    data['processed_text'].dropna(inplace=True)

    # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    data['processed_text'] = [str(entry).lower() for entry in data['processed_text']]

    # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
    data['processed_text'] = [word_tokenize(entry) for entry in data['processed_text']]

    # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in enumerate(data['processed_text']):
        # Declaring Empty List to store the words that follow the rules for this step
        Processed_text = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Processed_text.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        data.loc[index,'processed_text'] = str(Processed_text)


    sentiment_preds = sentiment_model.predict(data["processed_text"])

    # adding sarcasm predictions column to dataframe
    data['sentiment_predictions'] = sentiment_preds

    data_dict = data.to_dict(orient="records")
    print(data_dict)
    return jsonify(data_dict)

@app.route('/results')
def results():
    response = requests.get("http://localhost:5000/analyse")
    data = response.json()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
