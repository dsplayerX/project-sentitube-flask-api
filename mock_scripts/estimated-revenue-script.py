from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv # loading api keys from enviroment
import os # loading api keys from enviroment

load_dotenv() # loading api keys from enviroment
api_key = os.environ.get("API_KEY")

youtube = build('youtube', 'v3', developerKey=api_key)

def get_video_stats(video_id):
    try:
        # Retrieve video statistics using the YouTube API
        response = youtube.videos().list(
            part='statistics',
            id=video_id
        ).execute()

        # Extract statistics from the response
        items = response.get('items', [])
        if len(items) > 0:
            video_stats = items[0]['statistics']
            return video_stats
        else:
            return None

    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred: {e.content}")

def calculate_estimated_revenue(view_count, like_count, comment_count):
    # Constants for estimation (to be adjusted based on actual data)
    CPM = 1.5  # Estimated revenue per 1000 ad impressions in USD
    CTR = 0.02  # Estimated click-through rate
    CPC = 0.2  # Estimated cost per click in USD
    CPP = 0.05  # Estimated cost per comment in USD

    # Calculate estimated revenue
    ad_impressions = view_count // 1000
    estimated_revenue_impressions = ad_impressions * CPM

    estimated_revenue_likes = like_count * CTR * CPC
    estimated_revenue_comments = comment_count * CPP

    total_estimated_revenue = estimated_revenue_impressions + estimated_revenue_likes + estimated_revenue_comments
    return total_estimated_revenue

# Example usage
video_id = 'XrPZSq5YXqc'  # Replace with the actual YouTube video ID
video_stats = get_video_stats(video_id)

if video_stats:
    view_count = int(video_stats.get('viewCount', 0))
    like_count = int(video_stats.get('likeCount', 0))
    comment_count = int(video_stats.get('commentCount', 0))

    estimated_revenue = calculate_estimated_revenue(view_count, like_count, comment_count)
    print(f"Estimated revenue: ${estimated_revenue:.2f}")
else:
    print("No video statistics found.")
#https://www.youtube.com/watch?v=