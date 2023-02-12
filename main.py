from flask import Flask, jsonify, request
import requests
import os
import googleapiclient.discovery
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import urlparse # for formatting yt url

# Loading environment variables from .env file
load_dotenv()
api_key = os.environ.get("API_KEY")
api_service_name = "youtube"
api_version = "v3"


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
