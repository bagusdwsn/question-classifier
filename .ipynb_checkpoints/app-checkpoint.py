from flask import Flask, request, jsonify
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
import joblib
from nltk.corpus import stopwords
import nltk

# nltk.download('stopwords')

app = Flask(__name__)

# load model and vectorizer
loaded_model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')  

def preprocess_text(text):
    # lowercase
    text = text.lower()
    # strip string
    text = re.sub(r'[^\sA-Za-z]', '', text)
    # stopwords removal
    stop_words = set(stopwords.words('indonesian'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    new_question = data['question']  
    new_question_preprocessed = preprocess_text(new_question)  

    new_question_counts = vectorizer.transform([new_question_preprocessed])  

    predicted_topic = loaded_model.predict(new_question_counts)

    return jsonify({
        'topic': predicted_topic[0],
        'cleaned_text': new_question_preprocessed
    })

if __name__ == '__main__':
    app.run(debug=True)
