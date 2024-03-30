import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the CSV file
df = pd.read_csv(r"C:\Users\annad\OneDrive\Documents\Programmes\Python\NLP Lab\spam.csv")

# Define a function to calculate TF-IDF
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

# Calculate TF-IDF for the email content
tfidf_dict = calculate_tfidf(df['v2'])
df['tfidf'] = df['v2'].apply(lambda x: sum(tfidf_dict.get(word, 0) for word in x.split()))

# Split the dataset into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
X_test = test_df["v2"]
y_test = test_df["v1"]
# Prepare the training data
X_train = np.array(train_df['tfidf'].tolist()).reshape(-1, 1)
y_train = train_df['v1'].apply(lambda x: 1 if x == 'spam' else 0).values

# Initialize and train the RandomForest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Define a function to predict spam/ham
def predict_spam_ham(email_content, threshold=0.5):
    email_tfidf = sum(tfidf_dict.get(word, 0) for word in email_content.split())
    X_new = np.array([email_tfidf]).reshape(-1, 1)

    probability_spam = model.predict_proba(X_new)[0, 1]

    if probability_spam > threshold:
        return 'spam'
    else:
        return 'ham'

# Test the model with a new email
new_email_content = "Congratulations! You've won a prize."
predicted_label = predict_spam_ham(new_email_content)
print(f"Predicted Label: {predicted_label}")