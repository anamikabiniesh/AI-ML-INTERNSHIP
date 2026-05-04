# Restaurant Review Sentiment Analysis - Version 2

import nltk
import string
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required resources (run first time only)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Sample Reviews
reviews = [
    "The pizza was fantastic and service was excellent",
    "Terrible experience, food was cold and staff were rude",
    "Good taste but a bit expensive",
    "Absolutely amazing atmosphere and delicious desserts",
    "Not worth the money, very disappointing",
    "Average place, nothing impressive",
    "Loved the hospitality and the quality of food",
    "Worst dining experience ever"
]

# ----------------------------
# Text Cleaning Function
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    words = [word for word in tokens if word not in stop_words]
    return words

# ----------------------------
# Sentiment Classification
# ----------------------------
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.2:
        return "Positive 😊", polarity
    elif polarity < -0.2:
        return "Negative 😡", polarity
    else:
        return "Neutral 😐", polarity

# ----------------------------
# Main Program
# ----------------------------
print("\n===== Restaurant Review Sentiment Analysis (Version 2) =====\n")

for i, review in enumerate(reviews, 1):
    print(f"Review {i}: {review}")
    
    cleaned = clean_text(review)
    print("Cleaned Words:", cleaned)
    
    sentiment, score = get_sentiment(review)
    print("Polarity Score:", round(score, 3))
    print("Sentiment:", sentiment)
    
    print("-" * 60)