import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\annad\OneDrive\Documents\Programmes\Python\NLP Lab\Musical_instruments_reviews.csv")
df['sentiment'] = df['overall'].map({5: 1, 4: -1, 3: 0, 2: -1, 1: -1})

positive_df = df[df['sentiment'] == 1]
neutral_df = df[df['sentiment'] == 0]
negative_df = df[df['sentiment'] == -1]

sample_size = min(len(positive_df), len(neutral_df), len(negative_df))

positive_sample = positive_df.sample(sample_size, random_state=42)
neutral_sample = neutral_df.sample(sample_size, random_state=42)
negative_sample = negative_df.sample(sample_size, random_state=42)

balanced_df = pd.concat([positive_sample, neutral_sample, negative_sample])

train_df, test_df = train_test_split(balanced_df, test_size=0.2, random_state=42)

def calculate_tfidf(corpus):
    tf = {}
    for doc in corpus:
        for word in doc.split():
            tf[word] = tf.get(word, 0) + 1

    idf = {}
    for doc in corpus:
        for word in set(doc.split()):
            idf[word] = idf.get(word, 0) + 1

    tfidf = {}
    for word, tf_value in tf.items():
        idf_value = np.log(len(corpus) / (idf[word] + 1))
        tfidf[word] = tf_value * idf_value

    return tfidf

tfidf_dict = calculate_tfidf(train_df['summary'])
train_tfidf = train_df['summary'].apply(lambda x: sum(tfidf_dict.get(word, 0) for word in x.split()))
test_tfidf = test_df['summary'].apply(lambda x: sum(tfidf_dict.get(word, 0) for word in x.split()))

X_train = np.array(train_tfidf.tolist()).reshape(-1, 1)
y_train = train_df['sentiment'].values
  
X_test = np.array(test_tfidf.tolist()).reshape(-1, 1)
y_test = test_df['sentiment'].values

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

def predict_sentiment(review, threshold=0.2):
    review_tfidf = sum(tfidf_dict.get(word, 0) for word in review.split())
    X_new = np.array([review_tfidf]).reshape(-1, 1)
    probability_positive = model.predict_proba(X_new)[0, 1]

    if probability_positive > threshold:
        return 'Positive'
    else:
        return 'Negative'

def evaluate_accuracy(X_test, y_test, threshold=0.2):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

new_review = "Very good product"
predicted_sentiment = predict_sentiment(new_review)
print(f"Predicted Sentiment: {predicted_sentiment}")

accuracy = evaluate_accuracy(X_test, y_test)
print(f"Accuracy on Testing Data: {accuracy}")
