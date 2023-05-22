from flask import Flask, jsonify, request, abort # redirect, url_for
from flask_cors import CORS # fix for CORS error
import os
import datetime

import googleapiclient.discovery # youtube api
import pandas as pd
from dotenv import load_dotenv # loading api keys from enviroment
from urllib.parse import urlparse, parse_qs # for formatting yt url
import pickle # for loading models into API
import nltk # used for preprocessing of fetched comments
from collections import defaultdict

import openai # openai api
from youtube_transcript_api import YouTubeTranscriptApi # getting transcript from youtube video

# Download dependency
for dependency in (
    "punkt",
    "stopwords",
    "wordnet",
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

# loading openai api key from enviroment
open_api_key = os.environ.get("OPENAI_API_KEY")

# configuring youtube API parameters
api_key = os.environ.get("API_KEY")
api_service_name = "youtube"
api_version = "v3"

#creating YouTube API with api key
youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = api_key)
print("> YouTube API connected successfully!")

# secrest keys for EmailJS email client
service_key = os.environ.get('EMAIL_SERVICE_KEY')
template_key = os.environ.get('EMAIL_TEMPLATE_KEY')
secret_key = os.environ.get('EMAIL_SECRET_KEY')
print("> Email client secret keys loaded successfully!")

# load saved sentiment analysis model
sentiment_model = pickle.load(open("models/sentiment-analysis-pipeline.pkl", "rb"))
print("> Sentiment Model loaded successfully!")

# load saved sarcasm analysis model
sarcasm_model = pickle.load(open("models/sarcasm-analysis-pipeline.pkl", "rb"))
print("> Sarcasm Model loaded successfully!")

# creating an flask app
app = Flask(__name__)

# wraaping app in CORS to remove CORS erros
CORS(app)
#CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})


@app.route('/')
def index():
    return jsonify({"SentiTube Backend API": "SentiTube by phoeniX"})


# Route for the website
@app.route('/analysisresults', methods=['POST'])
def analysisresults():
    
    current_time = datetime.datetime.now()
    print("####################################### \n> Website Request at:", current_time)
    # Getting the UserInput
    # user_input = request.args.get('userinput', default = "", type = str)
    data = request.get_json()
    user_input = data["userinput"] # youtube link user entered
    numresults = data["numresults"] # number of comments to fetch
    orderresults = data["orderresults"] # the sorting order of the comments
    print("User-Input: ", user_input)
    # yt_url = validatelink(user_input)
    # extracts the YouTube video ID from the validated link
    vid_id = get_video_id(user_input)
    print("YouTube Video_ID: ", vid_id)

    # get video title fro the video id
    vid_title = getvideotitle(vid_id)

    # retrieve number of related comments from the video id 
    fetched_comments = fetchcomments(vid_id, numresults, orderresults)
    processed_comments = preprocess(fetched_comments)
    # Use the trained model to predict the sentiment and sarcasm of the preprocessed comments
    predicted_comments= predict(processed_comments)
    sentitube_comments= getsentituberesults(predicted_comments)

    # Create a dictionary called final comments dict and kept comment results on it.
    final_comments_dict = sentitube_comments.to_dict(orient="index")

    # get total comments by getting the shape of df
    total_comments = int(sentitube_comments.shape[0])
    # counts from sentiment analysis
    positive_count = int((sentitube_comments['sentiment_predictions'] == 2).sum())
    neutral_count = int((sentitube_comments['sentiment_predictions'] == 1).sum())
    negative_count = int((sentitube_comments['sentiment_predictions'] == 0).sum())
    #counts from sarcasm analysis
    sarcastic_count = int((sentitube_comments['sarcasm_predictions'] == 1).sum())
    nonsarcastic_count = int((sentitube_comments['sarcasm_predictions'] == 0).sum())
    #counts from sentitube results
    senti_positive_count = int((sentitube_comments['sentitube_results'] == 'positive').sum())
    senti_neutral_count = int((sentitube_comments['sentitube_results'] == 'neutral').sum())
    senti_negative_count = int((sentitube_comments['sentitube_results'] == 'negative').sum())

    # calculating percenatages for custom feedback
    check_percentage = lambda pos_per: \
    0 if (pos_per == -1) \
    else 1 if (pos_per == 0) \
    else 2 if (pos_per > 0 and pos_per < 5) \
    else 2 if (pos_per >= 5 and pos_per < 10) \
    else 3 if (pos_per >= 10 and pos_per < 15) \
    else 4 if (pos_per >= 15 and pos_per < 20) \
    else 5 if (pos_per >= 20 and pos_per < 25) \
    else 6 if (pos_per >= 25 and pos_per < 30) \
    else 7 if (pos_per >= 30 and pos_per < 35) \
    else 8 if (pos_per >= 35 and pos_per < 40) \
    else 9 if (pos_per >= 40 and pos_per < 45) \
    else 10 if (pos_per >= 45 and pos_per < 50) \
    else 11 if (pos_per >= 50 and pos_per < 55) \
    else 12 if (pos_per >= 55 and pos_per < 60) \
    else 13 if (pos_per >= 60 and pos_per < 65) \
    else 14 if (pos_per >= 65 and pos_per < 70) \
    else 15 if (pos_per >= 70 and pos_per < 75) \
    else 16 if (pos_per >= 75 and pos_per < 80) \
    else 17 if (pos_per >= 80 and pos_per < 85) \
    else 18 if (pos_per >= 85 and pos_per < 90) \
    else 20 if (pos_per >= 90 and pos_per < 95) \
    else 20 if (pos_per >= 95 and pos_per < 100) \
    else 21 if (pos_per == 100) \
    else 0 

    # count all senti positive comments and senti negative comments
    senti_total_count = senti_positive_count + senti_negative_count
    # calaculate the senti positive persentage
    pos_per = -1
    if senti_total_count != 0:
        pos_per = (senti_positive_count/senti_total_count) * 100 
    final_percentage = check_percentage(pos_per)
    
    print("Results: T", total_comments, "/ Pos", positive_count, "/ Neu", neutral_count, "/ Neg", negative_count, "/ S", sarcastic_count, "/ NS", nonsarcastic_count, "/ SPos", senti_positive_count, "/ SNeg", senti_negative_count)
    print(check_percentage)

    summary = "" # initialize the summary to empty string
    if (open_api_key != "XXX"): # if the open api key is not XXX, then call the open api
        # Get the transcript of the YouTube video
        transcript = get_youtube_transcript(vid_id)
        # print(transcript)
        if transcript == "":
            summary = ""
        else:
            # Summarize the transcript
            summary = summarize_text(transcript, vid_title)
            # print("sum:", summary)
            if (summary == None):
                summary = ""
            # print(summary)
    
    results = {
        'Video Title': vid_title,
        'Video Id': vid_id,
        'Total Comments':total_comments,
        'Positive Comments':positive_count,
        'Neutral Comments':neutral_count,
        'Negative Comments':negative_count,
        'Sarcastic Comments':sarcastic_count,
        'Nonsarcastic Comments':nonsarcastic_count,
        'Sentitube Positve' :senti_positive_count,
        'Sentitube Neutral' :senti_neutral_count,
        'Sentitube Negative' :senti_negative_count,
        'CustomFeedbackNo' : final_percentage,
        'Comments Dictionary':final_comments_dict,
        'Video Summary' : summary,
    }

    # returns the dictionary as a JSON object using the 'jsonify()' method
    return jsonify(results)

# Route for the chrome extension
@app.route('/extensionresults', methods=['GET'])
def extensionresults():
    current_time = datetime.datetime.now()
    print("#######################################  \n> Extension Request at:", current_time)
    # Getting the UserInput
    user_input = request.args.get('userinput', default = "", type = str)
    numresults = 300
    orderresults = "Top comments"
    yt_url = validatelink(user_input)
    vid_id = get_video_id(yt_url)
    fetched_comments = fetchcomments(vid_id, numresults, orderresults)
    processed_comments = preprocess(fetched_comments)
    predicted_comments= predict(processed_comments)
    sentitube_comments= getsentituberesults(predicted_comments)

    total_comments = int(predicted_comments.shape[0])
    # counts from sentiment analysis
    positive_count = int((sentitube_comments['sentiment_predictions'] == 2).sum())
    neutral_count = int((sentitube_comments['sentiment_predictions'] == 1).sum())
    negative_count = int((sentitube_comments['sentiment_predictions'] == 0).sum())
    #counts from sarcasm analysis
    sarcastic_count = int((sentitube_comments['sarcasm_predictions'] == 1).sum())
    nonsarcastic_count = int((sentitube_comments['sarcasm_predictions'] == 0).sum())
    #counts from sentitube results
    senti_positive_count = int((sentitube_comments['sentitube_results'] == 'positive').sum())
    senti_neutral_count = int((sentitube_comments['sentitube_results'] == 'neutral').sum())
    senti_negative_count = int((sentitube_comments['sentitube_results'] == 'negative').sum())

    print("Results: T", total_comments, "/ Pos", positive_count, "/ Neu", neutral_count, "/ Neg", negative_count, "/ S", sarcastic_count, "/ NS", nonsarcastic_count, "/ SPos", senti_positive_count, "/ SNeg", senti_negative_count)

    # comments_dict = predicted_comments["rawcomment"].to_dict(orient="index")
    #return jsonify(newDict)
    results = {
        'Positive Comments':positive_count,
        'Neutral Comments':neutral_count,
        'Negative Comments':negative_count,
        'Sarcastic Comments':sarcastic_count,
        'Nonsarcastic Comments':nonsarcastic_count,
        'Sentitube Positve' :senti_positive_count,
        'Sentitube Neutral' :senti_neutral_count,
        'Sentitube Negative' :senti_negative_count,
        'Total Comments':total_comments
    }
    return jsonify(results)

# Route to send emailjs secrets from environment to the frontend
@app.route('/getemailsecrets')
def getemailsecrets():
    # Retrieve the secrets from environment variables

    return jsonify({
        "serviceKey": service_key,
        "templateKey": template_key,
        "secretKey": secret_key
    })

# Route for the website
@app.route('/customtextanalysis', methods=['POST'])
def customtextanalysis():
    
    current_time = datetime.datetime.now()
    print("####################################### \n> Website Request at:", current_time)

    # Getting the UserInput
    data = request.get_json()
    user_input = data["userinput"] # text user entered
    
    # turn the user input into a dataframe
    user_input_df = pd.DataFrame([user_input], columns=['rawcomment'])
    
    processed_comments = preprocess(user_input_df)
    # Use the trained model to predict the sentiment and sarcasm of the preprocessed comments
    predicted_comments= predict(processed_comments)
    sentitube_comments= getsentituberesults(predicted_comments)

    # Create a dictionary called final comments dict and kept comment results on it.
    final_comments_dict = sentitube_comments.to_dict(orient="index")
    # print(final_comments_dict)

    # returns the dictionary as a JSON object using the 'jsonify()' method
    return jsonify(final_comments_dict)


# ===============================================
# ============== GLOBAL FUNCTIONS ===============
# ===============================================

def validatelink(user_input):
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
    if is_valid_youtube_url(user_input):
        print("User input is a Valid YouTube URL")
        return user_input
        
    else:
        print("User input is a INVALID YouTube URL")
        return ("INVALID URL")
    
    
def get_video_id(youtube_url):
    # Extracting the Video ID from the YouTube URL
    youtube_hostnames = ("www.youtube.com", "youtube.com", "m.youtube.com", "youtu.be")
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname in youtube_hostnames:
        query_params = parse_qs(parsed_url.query)
        if "v" in query_params:
            return query_params["v"][0]
        elif parsed_url.path.startswith("/embed/"):
            return parsed_url.path.split("/")[-1]
        elif parsed_url.hostname == "youtu.be": # this works when you enter a shortened youtube url
            return parsed_url.path[1:]
    return None

    # return video_id
def getvideotitle(video_id):
    # Call the videos.list method with id and part parameters
    request = youtube.videos().list(id=video_id, part="snippet")
    response = request.execute()
    # Get the title from the snippet object
    title = response["items"][0]["snippet"]["title"]
    # Return the title as plain text
    return title

# function to fetch comments from given YouTube Video ID, number of comments and the sorting order
def fetchcomments(video_id, no_of_comments, sort_by):

    order_by = "relevance" # default order is relavance

    #if user want to sort differnetly, correct variable is assigned.
    if (sort_by == "Newest first"):
        order_by = "time"

    print("Comments to fetch: ", no_of_comments, "\nOrder by: ", order_by)

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
            order=str(order_by),
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

    if len(comments_df) == 0:
        print("Error while fetching comments.")
        abort(500, description="Could not fetch comments!")
    else:
        print("Comments fetched successfully.")
    # print(comments_df)
    # comments_df_dict = comments_df.to_dict(orient="index")
    comments_df = comments_df[:no_of_comments] # slicing the dataframe to the number of comments user wants incase of batch missmatch
    return comments_df


def preprocess(comments_df):
    try:
        # copying rawcomments to a new column for preprocessing
        comments_df['processed_text'] = comments_df['rawcomment']
        # Remove blank rows if any.
        comments_df['processed_text'].dropna(inplace=True)

        # Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
        comments_df['processed_text'] = [str(entry).lower() for entry in comments_df['processed_text']]
        # Tokenization : In this each entry in the corpus will be broken into set of words
        comments_df['processed_text'] = [word_tokenize(entry) for entry in comments_df['processed_text']]

        # Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
        # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        for index,entry in enumerate(comments_df['processed_text']):
            # Declaring Empty List to store the words that follow the rules for this step
            Processed_words = []
            # Initializing WordNetLemmatizer()
            word_Lemmatized = WordNetLemmatizer()
            # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
            for word, tag in pos_tag(entry):
                # Below condition is to check for Stop words and consider only alphabets
                if word not in stopwords.words('english') and word.isalpha():
                    word_Processed = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                    Processed_words.append(word_Processed)
            # The final processed set of words for each iteration will be stored in 'text_final'
            comments_df.loc[index,'processed_text'] = str(Processed_words)
    
        print("Comments preprocessed successfully.")

        return comments_df
    except:
        print("!!!Error while preprocessing.")
        abort(500, description="Could not preprocess comments!")

def predict(comments_df):
    try:
        # adding sentiment and sarcasm predictions columns to dataframe
        comments_df['sentiment_predictions'] = sentiment_model.predict(comments_df["processed_text"])
        print("Sentiments predicted successfully.")

        comments_df['sarcasm_predictions'] = sarcasm_model.predict(comments_df["processed_text"])
        print("Sarcasm predicted successfully.")

        # copying the dataframe and dropping the column used in preprocessing
        predicted_df = comments_df.copy()
        predicted_df = predicted_df.drop(['processed_text'], axis=1)

        return predicted_df
    except:
        print("!!!Error while predicting sentiments and sarcasms.")
        abort(500, description="Could not predict sentiments and/or sarcasm!")

def getsentituberesults(predicted_df):
    try:
        predicted_df['sentitube_results'] = predicted_df.apply(lambda row: (
        'negative' if row['sentiment_predictions'] == 0 or
        row['sentiment_predictions'] == 2 and row['sarcasm_predictions'] == 1 else
        'neutral' if row['sentiment_predictions'] == 1 else
        'positive' if row['sentiment_predictions'] == 2 and row['sarcasm_predictions'] == 0 else
        'discarded'
        ), axis=1)
        final_df = predicted_df.copy()
        #print(final_df)
        #final_df.to_csv('temp/temp_comments.csv')
        return final_df
    except:
        print("!!!Couldn't get SentiTube results.")
        abort(500, description="Could not get SentiTube results!")


def get_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([caption['text'] for caption in transcript])

        return transcript_text
    except:
        print("!!!Error while fetching transcript.")
        #abort(500, description="Could not fetch transcript!")

def summarize_text(text, title):
    try:
        # Set up OpenAI API credentials
        openai.api_key = open_api_key  # Replace with your OpenAI API key

        # Define the chat completion prompt
        prompt = f"start as 'This video is about' and summarize the following in no more than 60 words: Video Title:{title} and transcript:{text}."

        # Generate the response using OpenAI's ChatGPT model
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=150,  # Adjust the desired length of the summary
            temperature=0.3,  # Adjust the level of randomness in the response
            n=1,
            stop=None,
            timeout=15  # Adjust the timeout as needed
        )

        # Extract the summarized text from the API response
        summary = response.choices[0].text.strip()
        print(response)
        return summary
    except:
        print("!!!Error while summarizing text.")
        #abort(500, description="Could not summarize text!")

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
