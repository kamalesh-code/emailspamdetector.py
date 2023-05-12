import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Function to read email files from a folder
def read_emails(folder, is_spam):
    emails = []
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r', encoding='utf-8', errors='ignore') as f:
            emails.append({'email': f.read(), 'is_spam': is_spam})
    return emails

# Load the dataset
spam_emails = read_emails('spam', 1)
ham_emails = read_emails('ham', 0)
data = pd.DataFrame(spam_emails + ham_emails)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['email'], data['is_spam'], test_size=0.2, random_state=42)

# Convert emails into feature vectors
vectorizer_nb = CountVectorizer()
X_train_vectors_nb = vectorizer_nb.fit_transform(X_train)
X_test_vectors_nb = vectorizer_nb.transform(X_test)

# Train the Naive Bayes classifier
clf_nb = MultinomialNB()
clf_nb.fit(X_train_vectors_nb, y_train)

# Evaluate the Naive Bayes model
y_pred_nb = clf_nb.predict(X_test_vectors_nb)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# Convert emails into feature vectors for SVM
vectorizer_svm = TfidfVectorizer()
X_train_vectors_svm = vectorizer_svm.fit_transform(X_train)
X_test_vectors_svm = vectorizer_svm.transform(X_test)

# Train the SVM classifier
clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(X_train_vectors_svm, y_train)

# Evaluate the SVM model
y_pred_svm = clf_svm.predict(X_test_vectors_svm)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Save the models and vectorizers
joblib.dump(clf_nb, 'naive_bayes_model.pkl')
joblib.dump(vectorizer_nb, 'vectorizer_nb.pkl')
joblib.dump(clf_svm, 'svm_model.pkl')
joblib.dump(vectorizer_svm, 'vectorizer_svm.pkl')

def predict_email(email_text, clf, vectorizer):
    email_vector = vectorizer.transform([email_text])
    prediction = clf.predict(email_vector)[0]
    return 'Spam' if prediction else 'Not spam'

# Get user input from the terminal
email_text = input("Enter the email text to predict with Naive Bayes: ")

# Predict whether the email is spam or not using Naive Bayes
clf_nb = joblib.load('naive_bayes_model.pkl')
vectorizer_nb = joblib.load('vectorizer_nb.pkl')
result_nb = predict_email(email_text, clf_nb, vectorizer_nb)
print("Naive Bayes Prediction:", result_nb)

# Get user input from the terminal
email_text = input("Enter the email text to predict with SVM: ")

# Predict whether the email is spam or not using SVM
clf_svm = joblib.load('svm_model.pkl')
vectorizer_svm = joblib.load('vectorizer_svm.pkl')
result_svm = predict_email(email_text, clf_svm, vectorizer_svm)
print("SVM Prediction:", result_svm)
