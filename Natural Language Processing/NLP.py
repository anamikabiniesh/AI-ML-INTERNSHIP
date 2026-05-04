# Restaurant Review Sentiment Analysis (Fixed Version)

import nltk
import string
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required data (run once)
nltk.download('punkt')
nltk.download('stopwords')

# Store stopwords once (better performance)
stop_words = set(stopwords.words('english'))

# Sample restaurant reviews
reviews = [
    "The food was absolutely delicious and the service was great",
    "I hated the food, it was cold and tasteless",
    "Amazing experience, will visit again!",
    "Very bad service and the food was not good",
    "The ambience was nice but the food was average",
    "Totally loved it, highly recommended",
    "Worst restaurant ever, very disappointed",
    "Food was okay, nothing special"
]

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    filtered_words = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return filtered_words

print("\n--- Restaurant Review Sentiment Analysis ---\n")

# Analyze each review
for review in reviews:
    print("Original Review:", review)

    # Preprocessing
    processed_words = preprocess_text(review)
    print("Processed Words:", processed_words)

    # Sentiment Analysis
    analysis = TextBlob(review)
    polarity = analysis.sentiment.polarity

    print("Sentiment Score:", polarity)

    # Classification
    if polarity > 0:
        print("Sentiment: Positive 😊")
    elif polarity < 0:
        print("Sentiment: Negative 😡")
    else:
        print("Sentiment: Neutral 😐")

    print("-" * 60)