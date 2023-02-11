# Imports
import pandas as pd
import googleapiclient.discovery
import joblib
import pickle

from collections import defaultdict

# NLTK for prerpcoessing
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from sklearn.feature_extraction.text import TfidfVectorizer

from urllib.parse import urlparse # for formatting yt url

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyD4ZIKBRTRVLnNMg47ZaXrsm6eIL6USHGA"

# enter youtube url here... 
youtube_url = "https://www.youtube.com/watch?v=kffacxfA7G4"

# getting the video id from the youtube url
url_data = urlparse(youtube_url)
video_id = url_data.query[2::]
print("video_id: ", video_id)

youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)
comments = []
authors = []

def load_comments(match):
    for item in match["items"]:
        comment = item["snippet"]["topLevelComment"]
        author = comment["snippet"]["authorDisplayName"]
        text = comment["snippet"]["textDisplay"]
        comments.append(text)
        authors.append(author)
    
        print("Comment by {}: {}".format(author, text))

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
    data.to_csv("comments.csv", encoding="utf-8")
    print(data)
    
try:
    while next_page_token and len(comments) < 1000: # used to reduce waiting time. if the video has a lot of comments the waiting time will be massive
        match = get_comment_threads(youtube, video_id, next_page_token)
        next_page_token = match["nextPageToken"]  # if the video has less than 100 top level comments this returns a keyerror
        load_comments(match)
    data = pd.DataFrame(comments, columns=["rawcomment"])
    data.to_csv("comments.csv", encoding="utf-8")
    print(data)
except:
    data = pd.DataFrame(comments, columns=["rawcomment"])
    data.to_csv("comments.csv", encoding="utf-8")
    print(data)

# Preprocess fetched data

data['comment'] = data['rawcomment']
# Step - a : Remove blank rows if any.
data['comment'].dropna(inplace=True)

# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
data['comment'] = [str(entry).lower() for entry in data['comment']]

# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
data['comment'] = [word_tokenize(entry) for entry in data['comment']]

# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(data['comment']):
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

print(data)
data.to_csv("processed-comments.csv", encoding="utf-8")


## sentiment analysis
# load saved sentiment model
sentiment_model = joblib.load('Senti-SVM-Model.pkl')
print("> Sentiment Model loaded successfully!")

# load vectors
sentiment_vectors = pickle.load(open("Senti-Tfdif-Vectors.pkl", 'rb'))
print("Vectors loaded successfully!")

vectorized_comments = sentiment_vectors.transform(data['processed_text'])

sentiment_preds = sentiment_model.predict(vectorized_comments)
print(sentiment_preds)

# adding sentiment predictions column to dataframe
data['sentiment_predictions'] = sentiment_preds


## sarcasm analysis
# load saved sarcasm model
sarcasm_model = joblib.load('Sarcasm-LRM-Model.pkl')
print("> Sarcasm Model loaded successfully!")
sarcasm_preds = sarcasm_model.predict(data["processed_text"])
print(sarcasm_preds)

# adding sarcasm predictions column to dataframe
data['sarcasm_predictions'] = sarcasm_preds

# saving full dataframe to csv file
data.to_csv('data-frame.csv')

# saving comments and predictions to the same csv file
pred_df = pd.DataFrame(data['rawcomment'])
pred_df['sentiment_predictions'] = sentiment_preds
pred_df['sarcasm_predictions'] = sarcasm_preds
pred_df.to_csv('predicted-comments.csv')
print(pred_df)

