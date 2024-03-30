import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

def generate_ngrams(text, n):
    words = text.split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    return ngrams

def create_vocabulary(corpus, n):
    vocab = set()
    for text in corpus:
        ngrams = generate_ngrams(text, n)
        vocab.update(ngrams)
    return vocab

n = 2
vocab = create_vocabulary(train_df['summary'], n)

def text_to_vectors(text, vocab, n):
    text_ngrams = generate_ngrams(text, n)
    vector = np.zeros(len(vocab))
    for ngram in text_ngrams:
        if ngram in vocab:
            vector[list(vocab).index(ngram)] += 1
    return vector

X_train = np.array([text_to_vectors(text, vocab, n) for text in train_df['summary']])
y_train = train_df['sentiment'].values

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

def predict_sentiment(review, vocab, n):
    review_vector = text_to_vectors(review, vocab, n).reshape(1, -1)
    sentiment = model.predict(review_vector)[0]
    return sentiment

new_review = "Good product"
predicted_sentiment = predict_sentiment(new_review, vocab, n)
print(f"Predicted Sentiment: {predicted_sentiment}")