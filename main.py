from flask import Flask, jsonify, request
import requests
import os
import googleapiclient.discovery
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import urlparse # for formatting yt url
import pickle
import nltk
from flask_cors import CORS

from collections import defaultdict

# Download dependency
for dependency in (
    "punkt",
    "stopwords",
    "wordnet",
    "porter_test",
    "maxent_treebank_pos_tagger",
    "averaged_perceptron_tagger"
):
    nltk.download(dependency)

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

# load saved sentiment analysis model
sentiment_model = pickle.load(open("models/sentiment-analysis-pipeline.pkl", "rb"))
print("> Sentiment Model loaded successfully!")

# load saved sarcasm analysis model
sarcasm_model = pickle.load(open("models/sarcasm-analysis-pipeline.pkl", "rb"))
print("> Sarcasm Model loaded successfully!")


app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})
    
@app.route('/analyse',methods=['GET'])
def analyse():
    # Youtube Video ID from URL
        # enter youtube url here... 
    youtube_url = request.args.get('userinput', default = "", type = str)

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
            # print(text)
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
        comments_df = pd.DataFrame(comments, columns=["rawcomment"])
        # data.to_csv("temp/temp_comments.csv", encoding="utf-8")
        
    try:
        while next_page_token and len(comments) < 250: # used to reduce waiting time. if the video has a lot of comments the waiting time will be massive
            match = get_comment_threads(youtube, video_id, next_page_token)
            next_page_token = match["nextPageToken"]  # if the video has less than 100 top level comments this returns a keyerror
            load_comments(match)
        comments_df = pd.DataFrame(comments, columns=["rawcomment"])
        # data.to_csv("temp/temp_comments.csv", encoding="utf-8")
    except:
        comments_df = pd.DataFrame(comments, columns=["rawcomment"])
        # data.to_csv("temp/temp_comments.csv", encoding="utf-8")

    print("Comments fetched successfully.")

    # copying rawcomments to a new column for preprocessing
    comments_df['processed_text'] = comments_df['rawcomment']
    # Step - a : Remove blank rows if any.
    comments_df['processed_text'].dropna(inplace=True)

    # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    comments_df['processed_text'] = [str(entry).lower() for entry in comments_df['processed_text']]

    # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
    comments_df['processed_text'] = [word_tokenize(entry) for entry in comments_df['processed_text']]

    # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in enumerate(comments_df['processed_text']):
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
        comments_df.loc[index,'processed_text'] = str(Processed_text)
    
    print("Comments preprocessed successfully.")

    # adding sentiment and sarcasm predictions columns to dataframe
    comments_df['sentiment_predictions'] = sentiment_model.predict(comments_df["processed_text"])
    comments_df['sarcasm_predictions'] = sarcasm_model.predict(comments_df["processed_text"])

    print("Sentiment and sarcasm was predicted successfully.")

    # copying the dataframe and dropping the column used for preprocessing
    processed_df = comments_df.copy()
    processed_df = processed_df.drop(['processed_text'], axis=1)

    processed_df_dict = processed_df.to_dict(orient="index")
    return jsonify(processed_df_dict)

@app.route('/results' , methods=['GET'])
def results():

    youtube_url = request.args.get('userinput', default = "", type = str) # this is used to get the youtube_url with get method
    response = requests.get("http://localhost:5000/analyse?userinput=" + youtube_url) #the analyse method is called with the userinput (yt url) to get analysis reuslts
    df = pd.DataFrame.from_dict(response.json(), orient="index")
    print(df)
    total_comments = int(df.shape[0])
    # counts from sentiment analysis
    positive_count = int((df['sentiment_predictions'] == 2).sum())
    neutral_count = int((df['sentiment_predictions'] == 1).sum())
    negative_count = int((df['sentiment_predictions'] == 0).sum())
    #counts from sarcasm analysis
    sarcastic_count = int((df['sarcasm_predictions'] == 1).sum())
    nonsarcastic_count = int((df['sarcasm_predictions'] == 0).sum())
    
    print(total_comments, positive_count, neutral_count, negative_count, sarcastic_count, nonsarcastic_count)
    # comments_dict = df["rawcomment"].to_dict(orient="index")
    #return jsonify(newDict)
    results = {
        'Total Comments':total_comments,
        'Positve Comments':positive_count,
        'Neutral Comments':neutral_count,
        'Negative Comments':negative_count,
        'Sarcastic Comments':sarcastic_count,
        'Nonsarcastic Comments':nonsarcastic_count
    }
    return (results)

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
