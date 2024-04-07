import pickle
import re

import pandas as pd
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import sent_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer


def track_para_id(sentence, para_id):
    if sentence.strip()[0].isdigit():
        para_id = int(sentence.strip().split('.')[0])
    elif para_id==None:
        para_id = 0

    return para_id


def track_speaker(sentence,speaker):
    if sentence.startswith("LORD"):
        speaker = ' '.join(sentence.split()[:2]).lower() # Extract speaker name from sentence
    elif speaker == None:
        speaker = 'None'
    return speaker

def new_case(filename, train=False):
    nlp = spacy.load("en_core_web_sm")
    speaker = None
    para_id = None
    case = filename.split(".")[0]
    with open('data/UKHL_txt/'+filename, "r", encoding="utf-8") as file:
        text = file.read()
    doc = nlp(text)
    old_sentences = [sent.text for sent in doc.sents]
    sentences = []
    for sentence in old_sentences:
        split_sentences = sentence.split('\n')
        sentences.extend(split_sentences)

    sentences = [sent.strip() for sent in sentences if sent.strip()]

    '''sentences = re.split(r'[\n\.]', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]'''
    max_line = len(sentences)
    vectorizer = TfidfVectorizer()
    data = pd.read_csv("AI.csv")
    X = vectorizer.fit_transform(data['body'])
    if train:
        '''data = ori_data[['body', 'to']]
    
        grouped_data = data.groupby('body')['to'].apply(list).reset_index(name='to')
    
        
        X = vectorizer.fit_transform(grouped_data['body'])
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(grouped_data['to'])
    
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
        classifier = MultiOutputClassifier(LogisticRegression())
        classifier.fit(X_train, y_train)
    
        accuracy = accuracy_score_multilabel(y_test, y_pred)
        print("Accuracy for 'to':", accuracy)'''

        #grouped_data = data.groupby('body')['to'].apply(list).reset_index()

        #grouped_data['label'] = grouped_data['to'].apply(lambda x: 1 if 'self' in x else 0)

        '''vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(grouped_data['body'])
        y_to = grouped_data['label']'''



        y_to = data['to']
        print('start split')
        X_train, X_test, y_to_train, y_to_test = train_test_split(X, y_to, test_size=0.2, random_state=42)

        #model_to = LogisticRegression()
        model_to = RandomForestClassifier()
        model_to.fit(X_train, y_to_train)

        y_to_pred = model_to.predict(X_test)
        accuracy_to = accuracy_score(y_to_test, y_to_pred)

        print("Accuracy for 'to':", accuracy_to)
        with open('RF_to.pkl', 'wb') as f:
            pickle.dump(model_to, f)
    else:
        with open('RF_to.pkl', 'rb') as f:
            model_to = pickle.load(f)

    line_num =0
    results = []
    for sentence in sentences:
        speaker = track_speaker(sentence,speaker)
        para_id = track_para_id(sentence,para_id)
        new_X = vectorizer.transform([sentence])
        #predicted_to = mlb.inverse_transform(classifier.predict(new_X))
        predicted_to = model_to.predict(new_X)
        #print(predicted_to)
        pos = round(line_num/max_line, 1)
        results.append({'case': case ,'line': line_num ,'para_id': para_id,'body': sentence, 'from': speaker, 'to': predicted_to[0], 'relation':'NAN', 'pos': pos, 'mj':'NAN'})
        line_num += 1

    df = pd.DataFrame(results)
    df.to_csv('data/UKHL_csv/'+case + '.csv', index=False)
    return df

def rewrite_rel(predicted, filename):
    case = filename.split(".")[0]
    original_data = pd.read_csv('data/UKHL_csv/' + case + '.csv')

    for index, row in predicted.iterrows():
        original_data.loc[(original_data['line'] == row['line']), 'relation'] = row['relation']

    original_data.to_csv('data/UKHL_csv/' + case + '.csv', index=False)

def rewrite_mj(mj, filename):
    case = filename.split(".")[0]
    original_data = pd.read_csv('data/UKHL_csv/' + case + '.csv')
    list = []
    for index, row in mj.iterrows():
        list.append(row['mj'])

    original_data["mj"] = original_data["mj"].replace(original_data["mj"].unique(), list)
    original_data.to_csv('data/UKHL_csv/' + case + '.csv', index=False)

#new_case('UKHL20012.txt',False)