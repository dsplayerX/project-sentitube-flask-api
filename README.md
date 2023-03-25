# SentiTube-Flask-API

Sentitube-Flask-API is a Flask-based web API for sentiment analysis of YouTube comments. The API uses Google's YouTube API to fetch comments from a specified YouTube video and analyzes both sentiment and sarcasm of the comments to give more accurate sentiment results.

You can access the SentiTube webapp by clicking on [this link](https://dsplayerx.github.io/project-sentitube-webapp). In addition to the frontend, there are other repositories available for the SentiTube project.

## Other "Project SentiTube" Repositories

Access other "Project SentiTube" repositories from below links.

- [React Website (Frontend)](https://github.com/dsplayerX/project-sentitube-flask-api) - The frontend of SentiTube where the users can interact with and use our service.
- [The Chrome Extension](https://github.com/dsplayerX/project-sentitube-chrome-extension) - A Chrome Extension that a user can easily access when watching a YouTube video that shows basic results to the user at a glance.
- [Project SentiTube Tests](https://github.com/dsplayerX/project-sentitube-tests) - This repository contains all the past machine learning modelling and mockup apis and webapps.

## Features

- Fetches comments from a YouTube video using YouTube API.
- Applies natural language processing techniques for sentiment and sarcasm analysis of the comments.
- Provides sentiment and sarcasm analysis results in JSON format.

## Installation

1. Clone the repository to your local machine.
2. Create a virtual Python environment `python -m venv venv`
3. Activate 'venv' Scripts `venv\Scripts\activate`
4. Install the required dependencies using `pip install -r requirements.txt.
5. Set up environment variables in a .env file.

### Environment Variables

The API uses the following environment variables:

- `API_KEY` - API key for the YouTube API
- `EMAIL_SERVICE_KEY` - Secret key for the EmailJS email client service
- `EMAIL_TEMPLATE_KEY` - Secret key for the EmailJS email client template
- `EMAIL_SECRET_KEY` - Secret key for the EmailJS email client

## Usage

To start the Flask API, run `python main.py`. By default, the API will be available at `http://localhost:5000`.

## API Endpoints

### POST `/analysisresults`

This endpoint takes in a YouTube video URL and returns sentiment analysis results for the selected number of comments on the video, ordered by either top or neweest comments. The sentiment and sarcasm analysis is performed using a pre-trained machine learning model, and the results are returned as a JSON object with the following format: ~ `{"Comments Dictionary: {"0":{"rawcomment:"Comment", "sarcasm_predictions": X, "sentiment_predictions": X, "sentitube_results": "result"}....} "CustomFeedbackNo": X,
"Negative Comments": X,
"Neutral Comments":X,
"Nonsarcastic Comments": X,
"Positive Comments": X,
"Sarcastic Comments": X,
"Sentitube Negative": X,
"Sentitube Neutral": X,
"Sentitube Positve": X,
"Total Comments": X,
"Video Id": "VideoID",
"Video Title": "VideoTitle"}`

### GET `/extensionresults`

This endpoint takes in a YouTube video URL and returns data for the Sentitube's chrome extension. The data is returned as a JSON object with the following format: ~ `{
    "Negative Comments": X,
    "Neutral Comments": X,
    "Nonsarcastic Comments": X,
    "Positive Comments": X,
    "Sarcastic Comments": X,
    "Sentitube Negative": X,
    "Sentitube Neutral": X,
    "Sentitube Positve": X,
    "Total Comments": X
}`

### GET `/getemailsecrets`

This enpoints returnes the loaded EmailJS API keys as a JSON object to the frontend website to be used in the contact us system.

## Contributers

- [dsplayerX](https://github.com/dsplayerX) - Developer / Tester
- [Tharu0418](https://github.com/TharU0418) - Developer
- [MrDarkDrago](https://github.com/MrDarkDrago) - Tester
- [Sanuri20](https://github.com/Sanuri20) - Tester

## Acknowledgments

- [YouTube API](https://developers.google.com/youtube/v3) - For fetching comments from a given YouTube video.
- [Twitter and Reddit Sentimental analysis Dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset) - Used to train the sentiment model.
- [News Headline Sarcasm Detection](https://www.kaggle.com/gcdatkin/news-headline-sarcasm-detection) - Used to train the sarcasm model.
- [NLP Preprocessing with NLTK](https://towardsdatascience.com/nlp-preprocessing-with-nltk-3c04ee00edc0) - Referenced to implement the NLTK preprocessing.
- [NLTK Lemmatization](https://www.holisticseo.digital/python-seo/nltk/lemmatize) - Refrences to learn the NLTK LEmmatization using WordNetLemmatizer.
