
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from flask import Flask,render_template,request,app
from flask import Response
app = Flask(__name__)


model = pickle.load(open("C:/Users/tusha/OneDrive/Desktop/nlp project/Dataset/artifacts/model.pkl", "rb"))
vectorizer = pickle.load(open("C:/Users/tusha/OneDrive/Desktop/nlp project/Dataset/artifacts/vertorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict_data():
    input_text = request.form["text"]
    input_text_preprocessed = [input_text.lower().replace('[^\w\s]', '')]
    input_text_features = vectorizer.transform(input_text_preprocessed)
    prediction = model.predict(input_text_features)

    if prediction[0] == 1:
        prediction_label = "True news"
    else:
        prediction_label = "Fake news"

    return render_template("home.html",result=prediction_label)

if __name__ == "__main__":
    app.run(debug=True)
