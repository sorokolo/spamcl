import flask
import pandas as pd
import pandas as pd
import numpy as np
import numpy as np
import codecs
import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
import re
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
import nltk
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model 
from flask import Flask, request, render_template



model = load_model("model/network.h5") 
# loading
with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

label2int = {'0': 0, '1': 1}
int2label = {0: '0', 1: '1'}
SEQUENCE_LENGTH = 100 

def preprocess_text(sen):
    sentence = remove_tags(sen)
    sentence = re.sub('[^a-zA-Z]', ' ', str(sentence))
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', str(sentence))
    sentence = re.sub(r'\s+', ' ', str(sentence))

    return sentence

TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', str(text))

def get_predictions(text):
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    # pad the sequence
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    # get the prediction
    prediction = model.predict(sequence)[0]
    # one-hot encoded vector, revert using np.argmax
    return int2label[np.argmax(prediction)]


def show_predict(email_text):
    prediction = get_predictions(email_text)
    email_type = ""
    if(prediction == '1'):
        email_type = "The prediction is "+str(prediction)+" It is a Spam"
    else:
        email_type = "The prediction is "+str(prediction)+" It is not a Spam"
    
    return email_type




app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
  

    data = request.form.get('email_text')
    prediction = ""
    if data == None:
        prediction = 'Got None'
    else:
        # model.predict.predict returns a dictionarys
        prediction = show_predict(data)
    # return json.dumps(str(prediction))

    return render_template('main.html',  original_input={
            "email_text":data,}, 
        
        prediction=prediction)



if __name__ == "__main__":
    app.run(debug=True)
