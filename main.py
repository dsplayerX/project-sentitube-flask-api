from flask import Flask, jsonify, request, abort # redirect, url_for
from flask_cors import CORS # fix for CORS error
import os
import googleapiclient.discovery # youtube api
import pandas as pd
from dotenv import load_dotenv # loading api keys from enviroment
from urllib.parse import urlparse, parse_qs # for formatting yt url
import pickle # for loading models into API
import nltk # used for preprocessing of fetched comments
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

youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = api_key)

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

@app.route('/testsenitituberesults', methods=['GET'])
def testresults():
    # Getting the UserInput
    user_input = request.args.get('userinput', default = "", type = str)

    print("User-Input: ", user_input)
    yt_url = validatelink(user_input)
    if yt_url == "INVALID URL":
        return ("Invalid URL")
    vid_id = get_video_id(yt_url)
    fetched_comments = fetchcomments(vid_id, 250)
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
    numresults = data["numresults"]
    print("User-Input: ", user_input)
    # yt_url = validatelink(user_input)
    # extracts the YouTube video ID from the validated link
    vid_id = get_video_id(user_input)
    print("YouTube Video_ID: ", vid_id)

    # get video title fro the video id
    vid_title = getvideotitle(vid_id)

    # retrieve number of related comments from the video id 
    fetched_comments = fetchcomments(vid_id, numresults)
    processed_comments = preprocess(fetched_comments)
    # Use the trained model to predict the sentiment and sarcasm of the preprocessed comments
    predicted_comments= predict(processed_comments)
    sentitube_comments= getsentituberesults(predicted_comments)

    # Create a dictionary called final comments dict and kept comment results on it.
    final_comments_dict = sentitube_comments.to_dict(orient="index")


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
    1 if (pos_per == 0) \
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
    pos_per = (senti_positive_count/senti_total_count) * 100    
    final_percentage = check_percentage(pos_per)
    
    print(total_comments, positive_count, neutral_count, negative_count, sarcastic_count, nonsarcastic_count, senti_positive_count, senti_neutral_count, senti_negative_count)
    print(check_percentage)
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
        'Comments Dictionary':final_comments_dict
    }
    # returns the dictionary as a JSON object using the 'jsonify()' method
    return jsonify(results)

@app.route('/extensionresults', methods=['GET'])
def extensionresults():
    # Getting the UserInput
    user_input = request.args.get('userinput', default = "", type = str)

    yt_url = validatelink(user_input)
    vid_id = get_video_id(yt_url)
    fetched_comments = fetchcomments(vid_id, 300)
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

    print(total_comments, positive_count, neutral_count, negative_count, sarcastic_count, nonsarcastic_count)
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
    # url_data = urlparse(youtube_url)
    # video_id = url_data.query[2::]
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


def fetchcomments(video_id, no_of_comments):
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
            order="relevance",
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
    return comments_df


def preprocess(comments_df):
    try:
        # comments_df = pd.DataFrame.from_dict(comments_df_dict, orient="index")
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

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
