import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('Musical_instruments_reviews.csv')
df['sentiment'] = df['overall'].map({5: 1, 4: 1, 3: 0, 2: -1, 1: -1})

positive_df = df[df['sentiment'] == 1]
neutral_df = df[df['sentiment'] == 0]
negative_df = df[df['sentiment'] == -1]

sample_size = min(len(positive_df), len(neutral_df), len(negative_df))

positive_sample = positive_df.sample(sample_size, random_state=42)
neutral_sample = neutral_df.sample(sample_size, random_state=42)
negative_sample = negative_df.sample(sample_size, random_state=42)

balanced_df = pd.concat([positive_sample, neutral_sample, negative_sample])

train_df, test_df = train_test_split(balanced_df, test_size=0.2, random_state=42)

def preprocess_text(text):
    words = text.lower().split()
    return words

def create_vocabulary(corpus):
    vocab = set()
    for text in corpus:
        words = preprocess_text(text)
        vocab.update(words)
    return vocab

vocab = create_vocabulary(train_df['summary'])
#Example vector: [0   2   0   3    1    0    0    0    1]
def text_to_bow(text, vocab):
    words = preprocess_text(text)
    vector = np.zeros(len(vocab))
    for word in words:
        if word in vocab:
            vector[list(vocab).index(word)] += 1
    return vector
#Model is trained on vectors of each entry in train_df
X_train = np.array([text_to_bow(text, vocab) for text in train_df['summary']])
y_train = train_df['sentiment'].values

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

def predict_sentiment(review, vocab):
    review_vector = text_to_bow(review, vocab).reshape(1, -1)
    sentiment = nb_classifier.predict(review_vector)[0]
    return sentiment

new_review = "Good product"
predicted_sentiment = predict_sentiment(new_review, vocab)
print(f"Predicted Sentiment: {predicted_sentiment}")
