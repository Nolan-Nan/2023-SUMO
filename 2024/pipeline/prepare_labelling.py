import re

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def prepare_labelling(filepath):
    filename = filepath.split("/")[-1]
    case = filename.split(".")[0]
    data = pd.read_csv('data/UKHL_csv/' + case + '.csv')
    data.rename(columns={'case': 'case_id','line':'sentence_id','from':'judge','body':'text'}, inplace=True)
    data['judge'] = data['judge'].str.replace('lord', '').str.strip()
    data['agree'] = np.where(data['relation'] == 'fullagr', data['to'], 'NONE')

    data['ackn'] = np.where(data['relation'] == 'ackn', data['to'], 'NONE')

    data['outcome'] = np.where((data['relation'] =='fullagr') | (data['relation'] =='self') & (data['to'].notnull()), 1, 'NONE')
    data['role'] = np.nan
    data.drop(columns=['to'], inplace=True)

    data.to_csv('data/UKHL_corpus/'+ case + '.csv', index=False)
    data.to_csv('data/UKHL_corpus2/' + case + '.csv', index=False)