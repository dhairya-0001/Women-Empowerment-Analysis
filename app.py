# Import necessary libraries
import tweepy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob
from flask import Flask, request, jsonify
import nltk
import re
from nltk.corpus import stopwords

# Download stopwords once
nltk.download('stopwords')

# Twitter API credentials - replace with your actual credentials
API_KEY = 'your_api_key'
API_SECRET_KEY = 'your_api_secret_key'
ACCESS_TOKEN = 'your_access_token'
ACCESS_TOKEN_SECRET = 'your_access_token_secret'

# Authenticate to Twitter API
auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# Function to collect tweets by hashtag
def get_tweets(hashtag, count=100):
    tweets = tweepy.Cursor(api.search_tweets, q=hashtag, lang="en").items(count)
    tweet_data = [{"text": tweet.text, "created_at": tweet.created_at} for tweet in tweets]
    return pd.DataFrame(tweet_data)

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#', '', text)        # Remove hashtag symbol
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing to all tweets in the DataFrame
def preprocess_tweets(df):
    df['processed_text'] = df['text'].apply(preprocess_text)
    return df

# Sentiment labeling based on TextBlob (positive sentiment as 1, negative as 0)
def label_sentiment(text):
    return 1 if TextBlob(text).sentiment.polarity > 0 else 0

# Fetch sample tweets and prepare data
df = get_tweets("#womensafety", count=200)
df = preprocess_tweets(df)
df['label'] = df['text'].apply(label_sentiment)  # Add labels for training

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['label'], test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train an SVM model
model = SVC(kernel='linear')
model.fit(X_train_vec, y_train)

# Evaluate model performance
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Real-time sentiment prediction function
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    sentiment = model.predict(vectorized_text)
    return "Positive" if sentiment[0] == 1 else "Negative"

# Set up Flask API
app = Flask(__name__)

@app.route('/check_safety', methods=['POST'])
def check_safety():
    data = request.json
    tweet_text = data.get('tweet')
    sentiment = predict_sentiment(tweet_text)
    response = {
        "text": tweet_text,
        "sentiment": sentiment,
        "alert": "High risk" if sentiment == "Negative" else "Safe"
    }
    return jsonify(response)

# Run Flask application
if __name__ == '__main__':
    app.run(debug=True)
