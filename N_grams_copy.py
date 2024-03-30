import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\annad\OneDrive\Documents\Programmes\Python\NLP Lab\Musical_instruments_reviews.csv")
df["sentiment"] = df["overall"].map({1: -1, 2: -1, 3: 0, 4: 1, 5: 1})
positive_df = df[df["sentiment"] == 1]
negative_df = df[df["sentiment"] == -1]
neutral_df = df[df["sentiment"] == 0]

sample_size = min(len(positive_df), len(neutral_df), len(negative_df))

balanced_positive = positive_df.sample(sample_size, random_state=42)
balanced_neutral = neutral_df.sample(sample_size, random_state=42)
balanced_negative = negative_df.sample(sample_size, random_state=42)

balanced_df = pd.concat([balanced_positive, balanced_negative, balanced_neutral])
train_df, test_df = train_test_split(balanced_df, random_state=42, test_size=0.2)

def generate_ngrams(text, n):
    words = text.split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(' '.join(words[i:i + n]))
    return ngrams

def generate_vocab(corpus, n):
    vocab = set()
    for text in corpus:
        curr_ngrams = generate_ngrams(text, n)
        vocab.update(curr_ngrams)
    return vocab

def generate_vector(text, n):
    vector = np.zeros(len(vocab))
    curr_ngrams = generate_ngrams(text, n)
    for ngram in curr_ngrams:
        vector[list(vocab).index(ngram)] += 1
    return vector

N = 2
vocab = generate_vocab(balanced_df["summary"], N)

X_train = [generate_vector(text, N) for text in train_df["summary"]]
y_train = train_df["sentiment"].values

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

X_test = [generate_vector(text, N) for text in test_df["summary"]]
y_test = test_df["sentiment"].values

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Testing Data: {accuracy}")

review = "Hello"
prediction = model.predict(generate_vector(review, N).reshape(1, -1))
print(prediction[0])
