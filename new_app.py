import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the saved model and vectorizer
clf = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def predict_email(email_text, clf, vectorizer):
    email_vector = vectorizer.transform([email_text])
    prediction = clf.predict(email_vector)[0]
    return 'Spam' if prediction else 'Not spam'

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        email_text = request.form['email']
        prediction = predict_email(email_text, clf, vectorizer)
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
