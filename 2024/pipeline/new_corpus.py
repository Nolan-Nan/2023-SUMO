'''def new_corpus(self, file):
    # make new corpus depends on the given txt file
    all = []
    with open("data/UKHL/" + file, 'r') as f:
        lines = [i.strip("\n") for i in f.readlines()]

        ref_sent = self.ext_ref(lines)
        line_num, body, max_line = self.ext_sent(file)
        case_name = int(file.strip(".txt"))

        for line_num, body in zip(line_num, body):
            try:  # Each annotation must have from, to and relation filled in
                ref = ref_sent[line_num]
                ref_from = ref["from"]
                ref_to = ref["to"]
                relation = ref["rel"]
                position = round(line_num / max_line, 1)
                for r_f, r_t, rel in zip(ref_from, ref_to, relation):
                    print(case_name, line_num, body, r_f, r_t, rel, position, mj)
                    all.append([case_name, line_num, body, r_f, r_t, rel, position, mj])

            except:  # If there is no annotation, fills the blanks
                mj, ref_from, ref_to, relation = "NAN", "NAN", "NAN", "NAN"
                pos = round(line_num / max_line, 1)
                all.append([case_name, line_num, body, ref_from, ref_to, relation, pos, mj])
        corpus = pd.DataFrame.from_records(all, index="case",
                                           columns=["case", "line", "body", "from", "to", "relation",
                                                    "pos", "mj"])
        save_data("new_corpus", corpus)
        return corpus'''
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
    df.to_csv(case + '.csv', index=False)
    return df

