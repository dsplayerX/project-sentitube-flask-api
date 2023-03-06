from flask import Flask, jsonify, request # redirect, url_for
from flask_cors import CORS
import requests
import os
import googleapiclient.discovery
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs # for formatting yt url
import pickle
import nltk


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
#CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})


@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})

@app.route('/testresults', methods=['GET'])
def testresults():
    # Getting the UserInput
    user_input = request.args.get('userinput', default = "", type = str)

    print("User-Input: ", user_input)
    yt_url = validatelink(user_input)
    if yt_url == "INVALID URL":
        return ("Invalid URL")
    vid_id = get_video_id(yt_url)
    fetched_comments = fetchcomments(vid_id, 1000)
    processed_comments = preprocess(fetched_comments)
    predicted_comments= predict(processed_comments)
    sentitube_comments= getsentituberesults(predicted_comments)


    total_comments = int(predicted_comments.shape[0])
    # counts from sentiment analysis
    positive_count = int((predicted_comments['sentiment_predictions'] == 2).sum())
    neutral_count = int((predicted_comments['sentiment_predictions'] == 1).sum())
    negative_count = int((predicted_comments['sentiment_predictions'] == 0).sum())
    #counts from sarcasm analysis
    sarcastic_count = int((predicted_comments['sarcasm_predictions'] == 1).sum())
    nonsarcastic_count = int((predicted_comments['sarcasm_predictions'] == 0).sum())
    #counts from sentitube results
    senti_positive_count = int((predicted_comments['sentitube_results'] == 'positive').sum())
    senti_neutral_count = int((predicted_comments['sentitube_results'] == 'neutral').sum())
    senti_negative_count = int((predicted_comments['sentitube_results'] == 'negative').sum())
    sentidiscard = int((predicted_comments['sentitube_results'] == 'discard').sum())



    print(total_comments, positive_count, neutral_count, negative_count, sarcastic_count, nonsarcastic_count, senti_positive_count, senti_neutral_count, senti_negative_count)
    # comments_dict = predicted_comments["rawcomment"].to_dict(orient="index")
    #return jsonify(newDict)
    results = {
        'Positive Comments':positive_count,
        'Neutral Comments':neutral_count,
        'Negative Comments':negative_count,
        'Sarcastic Comments':sarcastic_count,
        'Nonsarcastic Comments':nonsarcastic_count,
        'Total Comments':total_comments,
        'Sentitube Positve' :senti_positive_count,
        'Sentitube Neutral' :senti_neutral_count,
        'Sentitube Negative' :senti_negative_count,
        'Discard': sentidiscard,
    }
    return jsonify(results)

@app.route('/analysisresults', methods=['POST'])
def analysisresults():
    # Getting the UserInput
    # user_input = request.args.get('userinput', default = "", type = str)
    data = request.get_json()
    user_input = data["userinput"]
    print("User-Input: ", user_input)
    yt_url = validatelink(user_input)
    if yt_url == "INVALID URL":
        return ("Invalid URL")
    vid_id = get_video_id(yt_url)
    fetched_comments = fetchcomments(vid_id, 250)
    processed_comments = preprocess(fetched_comments)
    predicted_comments= predict(processed_comments)

    total_comments = int(predicted_comments.shape[0])
    # counts from sentiment analysis
    positive_count = int((predicted_comments['sentiment_predictions'] == 2).sum())
    neutral_count = int((predicted_comments['sentiment_predictions'] == 1).sum())
    negative_count = int((predicted_comments['sentiment_predictions'] == 0).sum())
    #counts from sarcasm analysis
    sarcastic_count = int((predicted_comments['sarcasm_predictions'] == 1).sum())
    nonsarcastic_count = int((predicted_comments['sarcasm_predictions'] == 0).sum())
    
    print(total_comments, positive_count, neutral_count, negative_count, sarcastic_count, nonsarcastic_count)
    # comments_dict = predicted_comments["rawcomment"].to_dict(orient="index")
    #return jsonify(newDict)
    results = {
        'Positive Comments':positive_count,
        'Neutral Comments':neutral_count,
        'Negative Comments':negative_count,
        'Sarcastic Comments':sarcastic_count,
        'Nonsarcastic Comments':nonsarcastic_count,
        'Total Comments':total_comments
    }
    return jsonify(results)


@app.route('/percomment_results' , methods=['GET'])
def percomment_results():
    # Getting the UserInput
    user_input = request.args.get('userinput', default = "", type = str)

    yt_url = validatelink(user_input)
    vid_id = get_video_id(yt_url)
    fetched_comments = fetchcomments(vid_id, 250)
    processed_comments = preprocess(fetched_comments)
    predicted_comments= predict(processed_comments)
    
    predicted_comments_dict = predicted_comments.to_dict(orient="index")

    return jsonify(predicted_comments_dict)

@app.route('/extensionresults', methods=['GET'])
def extensionresults():
    # Getting the UserInput
    user_input = request.args.get('userinput', default = "", type = str)

    yt_url = validatelink(user_input)
    vid_id = get_video_id(yt_url)
    fetched_comments = fetchcomments(vid_id, 200)
    processed_comments = preprocess(fetched_comments)
    predicted_comments= predict(processed_comments)

    total_comments = int(predicted_comments.shape[0])
    # counts from sentiment analysis
    positive_count = int((predicted_comments['sentiment_predictions'] == 2).sum())
    neutral_count = int((predicted_comments['sentiment_predictions'] == 1).sum())
    negative_count = int((predicted_comments['sentiment_predictions'] == 0).sum())
    #counts from sarcasm analysis
    sarcastic_count = int((predicted_comments['sarcasm_predictions'] == 1).sum())
    nonsarcastic_count = int((predicted_comments['sarcasm_predictions'] == 0).sum())
    
    print(total_comments, positive_count, neutral_count, negative_count, sarcastic_count, nonsarcastic_count)
    # comments_dict = predicted_comments["rawcomment"].to_dict(orient="index")
    #return jsonify(newDict)
    results = {
        'Positive Comments':positive_count,
        'Neutral Comments':neutral_count,
        'Negative Comments':negative_count,
        'Sarcastic Comments':sarcastic_count,
        'Nonsarcastic Comments':nonsarcastic_count,
        'Total Comments':total_comments
    }
    return jsonify(results)

# GLOBAL FUNCTIONS
def validatelink(user_input):
    

    ### OLD VALIDATION
    # # Function for validating the URL (is a valid URL or not?)
    # def is_url(url):
    #     try:
    #         result = urlparse(url)
    #         return all([result.scheme, result.netloc])
    #     except ValueError:
    #         return False

    # # Validating the user input to see if it's a valid URL
    # if (is_url(user_input)):
    #     print("Valid URL")
    # else:
    #     print("Invalid URL")
    #     return ("Please enter a valid URL.")

    # Function for validing whether a url is from YouTube domain or not
    def is_valid_youtube_url(url):
        youtube_hostnames = ("www.youtube.com", "youtube.com", "m.youtube.com", "youtu.be")
        parsed_url = urlparse(url)
        # print(parsed_url)
        if parsed_url.hostname in youtube_hostnames:
            query_params = parse_qs(parsed_url.query)
            # print(query_params)
            # checks whether the URL contains a common "v" parameter in its query string or starts with the "/embed/" path
            if "v" in query_params:
                return True
            elif parsed_url.path.startswith("/embed/"):
                return True
            return False
        else:
            return False
    
    # Validating the user_input to check if it's a YouTube URL or not
    if user_input and is_valid_youtube_url(user_input):
        print("User input is a Valid YouTube URL")
    else:
        print("User input is a INVALID YouTube URL")
        return ("INVALID URL")
        # FIND A WAY FOR THIS EXCEPTION TO BE HANDLEDDD!!!!
    
    # If all the validations are passed, the user_input is assigned as the youtube_url to extract the video id
    youtube_url = user_input
    print("Valid YouTube URL entered.")
    return youtube_url


def get_video_id(youtube_url):
    # Extracting the Video ID from the YouTube URL
    url_data = urlparse(youtube_url)
    video_id = url_data.query[2::]

    print("YouTube Video_ID: ", video_id)
    return video_id


def fetchcomments(video_id, no_of_comments):
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = api_key)
    comments = []

    # Function to load comments and append necessary details to the comments array
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
        while next_page_token and len(comments) < no_of_comments: # used to reduce waiting time. if the video has a lot of comments the waiting time will be massive
            match = get_comment_threads(youtube, video_id, next_page_token)
            next_page_token = match["nextPageToken"]  # if the video has less than 100 top level comments this returns a keyerror
            load_comments(match)
        comments_df = pd.DataFrame(comments, columns=["rawcomment"])
        # data.to_csv("temp/temp_comments.csv", encoding="utf-8")
    except:
        comments_df = pd.DataFrame(comments, columns=["rawcomment"])
        # data.to_csv("temp/temp_comments.csv", encoding="utf-8")

    print("Comments fetched successfully.")
    # print(comments_df)
    comments_df_dict = comments_df.to_dict(orient="index")
    return comments_df_dict


def preprocess(comments_df_dict):
    comments_df = pd.DataFrame.from_dict(comments_df_dict, orient="index")
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

    return comments_df

def predict(comments_df):
    # adding sentiment and sarcasm predictions columns to dataframe
    comments_df['sentiment_predictions'] = sentiment_model.predict(comments_df["processed_text"])
    print("Sentiments predicted successfully.")

    comments_df['sarcasm_predictions'] = sarcasm_model.predict(comments_df["processed_text"])
    print("Sarcasm predicted successfully.")

    # copying the dataframe and dropping the column used in preprocessing
    predicted_df = comments_df.copy()
    predicted_df = predicted_df.drop(['processed_text'], axis=1)

    return predicted_df

def getsentituberesults(predicted_df):

    predicted_df['sentitube_results'] = predicted_df.apply(lambda row: (
    'negative' if row['sentiment_predictions'] == 0 or
     row['sentiment_predictions'] == 2 and row['sarcasm_predictions'] == 1 else
    'neutral' if row['sentiment_predictions'] == 1 else
    'positive' if row['sentiment_predictions'] == 2 and row['sarcasm_predictions'] == 0 else
    'discard'
    ), axis=1)
    final_df = predicted_df.copy()
    print(final_df)
    final_df.to_csv('temp/temp_comments.csv')
    return final_df

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
