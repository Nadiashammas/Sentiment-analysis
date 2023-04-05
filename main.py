import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

text = "I really enjoyed this movie. The acting was great and the plot was engaging."

# Create a SentimentIntensityAnalyzer object
sia = SentimentIntensityAnalyzer()

# Analyze the sentiment of the text
sentiment_scores = sia.polarity_scores(text)

# Print the sentiment scores
print(sentiment_scores)
