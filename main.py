from flask import Flask, jsonify
import os
from googleapiclient.discovery import build
import pandas as pd

api_service_name = "youtube"
api_version = "v3"
api_key = ""

video_id = ""

resourse = build('youtube','v3', developerKey = api_key)


request = resourse.commentThreads().list(
                                part="snippet",
                                videoId=video_id,
                                maxResults=100,  # only returns 100 comments (even if this value was higher)
                                order="relevance")

response = request.execute()

items = response["items"]


app = Flask(__name__)


@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})
    
@app.route('/analyse')
def index():

    data = []
    for item in items:
        item_info = item["snippet"]
        topLevelComment = item_info["topLevelComment"]
        comment_info = topLevelComment["snippet"]
    
        data.append({
            "Comment By": comment_info["authorDisplayName"],
            "Comment Text": comment_info["textDisplay"],
            "Likes on Comment": comment_info["likeCount"],
            "Comment Date": comment_info["publishedAt"]
        })

    df = pd.DataFrame(data)
    df.to_csv("/temp/temp_comments.csv", encoding="utf-8")

    return df.to_json

@app.route('/results')
def index():
    return jsonify({"Results Route"})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
