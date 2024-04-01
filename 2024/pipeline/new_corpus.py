import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import sent_tokenize


def track_speaker(sentence,speaker):
    if sentence.startswith("LORD"):
        speaker = ' '.join(sentence.split()[:2]).lower() # Extract speaker name from sentence

    return speaker

def new_case(filename):
    speaker = None
    case = filename.split(".")[0]
    with open('data/UKHL_txt/'+filename, "r", encoding="utf-8") as file:
        text = file.read()

    sentences = re.split(r'[\n\.]', text)
    max_line = len(sentences)

    data = pd.read_csv("AI.csv")

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['body'])
    y_to = data['to']
    print('start split')
    X_train, X_test, y_to_train, y_to_test = train_test_split(X, y_to, test_size=0.2, random_state=42)

    model_to = LogisticRegression()
    model_to.fit(X_train, y_to_train)

    y_to_pred = model_to.predict(X_test)
    accuracy_to = accuracy_score(y_to_test, y_to_pred)

    print("Accuracy for 'to':", accuracy_to)
    line_num =0
    results = []
    for sentence in sentences:
        speaker = track_speaker(sentence,speaker)
        new_X = vectorizer.transform([sentence])
        predicted_to = model_to.predict(new_X)
        pos = round(line_num/max_line, 1)
        if predicted_to != 'NAN':
            results.append({'case': case ,'line': line_num ,'body': sentence, 'from': speaker, 'to': predicted_to[0], 'relation':'NAN', 'pos': pos, 'mj':'NAN'})
        else:
            results.append({'case': case, 'line': line_num, 'body': sentence, 'from': 'NAN', 'to': predicted_to[0],'relation':'NAN', 'pos': pos, 'mj':'NAN'})
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

