from .item_selector import ItemSelector
from .parameters import Parameters

from ..corpus.storage import save_data, load_data


from .deparser import Deparser

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import re
import nltk
import sklearn 

""" models """
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model  import SGDClassifier
from sklearn.naive_bayes import BernoulliNB

""" scklearntools """
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

""" undersampling """
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

class PosTags(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        pos_tags = []
        for text in posts:
            text = nltk.word_tokenize(text)
            text = nltk.pos_tag(text)
            tags = [pos[1] for pos in text]
            tags = " ".join(tags)
            pos_tags.append({"tag" : tags})

        all = pd.DataFrame(pos_tags)

        return all["tag"]

class Dependencies(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        dep_list = []
        for text in posts:
            dep = Deparser()
            dependencies = dep.parse(text)
            dep_list.append({"dep" : dependencies})

        all = pd.DataFrame(dep_list)

        return all["dep"]

class Cue_phrase(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        # print("original", posts.shape())
        cue = ["For these reasons", "allow the appeal", "dismiss the appeal", "I have had the advantage", "I agree with it", "For the reasons"]
        cue_tags = []
        for text in posts:
            record = "N"
            for c in cue:
                if c in text:
                    record += str(cue.index(c))
                else:
                    record += "N"
            # print(record)
            cue_tags.append({"myner" : record})

        return cue_tags

class NER(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        # print("original", posts.shape())
        ner_tags = []
        for text in posts:
            if "Lord" in text or "Lady" in text:
                ner_tags.append({"myner" : 1})
            else:
                ner_tags.append({"myner" : 0})

        return ner_tags

class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        stats = [{'length': len(text)} for text in posts]
        return stats

class PositionStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        stats = [{'position': text} for text in posts]
        return stats

class NumStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        num_tags = []
        for text in posts:
            if self.hasNumbers(text):
                num_tags.append({"num" : 1})
            else:
                num_tags.append({"num" : 0})

        return num_tags

    def hasNumbers(self, inputString):
        return any(char.isdigit() for char in inputString)

class Classifier:

    def __init__(self, corpus, test_size, train):
        self.corpus = corpus[~corpus["relation"].isin(["outcome"])]
        self.X, self.y = self.format_train_corp(self.corpus)
        self.classifiers = Parameters().classifiers
        self.params = Parameters().parameters
        self.test_size = test_size
        self.train = train

    def get_prediction(self, MJ_corpus):
        if self.train:
            classifier = self.best_classifier()
            save_data("classifier", classifier) #save
        else:
            classifier = load_data("classifier") #NOTE write test

        X, y = self.format_corp(MJ_corpus)
        predictions = classifier.predict(X)
        MJ_corpus["predictions"] = predictions.tolist()
        predicted = MJ_corpus[(MJ_corpus["predictions"] == "fullagr") | (MJ_corpus["predictions"] == "ackn")][["case", "line", "predictions"]]
        predicted.columns = ["case", "line", "relation"]
        return predicted

    def format_train_corp(self, corpus):
        """
        Formats corpus to fit X, y format, removes unecessary categories.
        """

        '''corpus['relation'] = corpus['relation'].map({"NAN": "NAN", "other": "NAN",
        "error": "NAN", 'fullagr': "fullagr", 'ackn': "ackn", 'outcome': "NAN",
        'partagr': "NAN", 'partdisa': "NAN", 'fulldisa': "NAN", 'factagr': "NAN"})'''

        corpus.loc[:, 'relation'] = corpus['relation'].map({"NAN": "NAN", "other": "NAN",
                                                            "error": "NAN", 'fullagr': "fullagr",
                                                            'ackn': "ackn", 'outcome': "NAN",
                                                            'partagr': "NAN", 'partdisa': "NAN",
                                                            'fulldisa': "NAN", 'factagr': "NAN"})

        corpus = self.corp_downsample(corpus)
        X = corpus[["body", "pos"]]
        y = corpus["relation"]


        return X, y

    def format_corp(self, corpus):
        """
        Formats corpus to fit X, y format, removes unecessary categories.
        """

        corpus['relation'] = corpus['relation'].map({"NAN": "NAN", "other": "NAN",
        "error": "NAN", 'fullagr': "fullagr", 'ackn': "ackn", 'outcome': "NAN",
        'partagr': "NAN", 'partdisa': "NAN", 'fulldisa': "NAN", 'factagr': "NAN"})

        X = corpus[["body", "pos"]]
        y = corpus["relation"]


        return X, y

    def new_classifier(self, parameters, classifier_type):
        """
        Pipeline for the classifier.
        """
        # scikit pipeline
        text_clf = Pipeline([

            # Use FeatureUnion to combine the BoW and TextStats features
            ('union', FeatureUnion(
                transformer_list=[

                    # Pipeline for Stanford Dependencies
                    # ('Dependencies', Pipeline([
                    #     ('position', ItemSelector("body")),
                    #     ('dep', Dependencies()),  # returns a list of dicts
                    #     ('dep_vect', TfidfVectorizer()),
                    # ])),

                    # sentence has a number or not?
                    # ('num_stats', Pipeline([
                    #     ('position', ItemSelector("body")),
                    #     ('num', NumStats()),  # returns a list of dicts
                    #     ('cue_vect', DictVectorizer()),
                    # ])),

                    # Pipeline for Cue phrases
                    ('Cue_phrases', Pipeline([
                        ('position', ItemSelector("body")),
                        ('cue', Cue_phrase()),  # returns a list of dicts
                        ('cue_vect', DictVectorizer()),
                    ])),

                    # Pipeline for Position
                    ('Postion_stats', Pipeline([
                        ('position', ItemSelector("pos")),  # returns a list of dicts
                        ('stats', PositionStats()),
                        ('position_vect', DictVectorizer()),
                    ])),

                    # Pipeline for NER
                    ('ner_hot', Pipeline([
                        ('position', ItemSelector("body")),
                        ('ner', NER()),  # returns a list of dicts
                        ('ner_vect', DictVectorizer()),
                    ])),

                    # Pipeline for pos-tags
                    ('pos_tags', Pipeline([
                        ('position', ItemSelector("body")),
                        ('tags', PosTags()),  # returns a list of dicts
                        ('pos_vect',TfidfVectorizer()),
                    ])),

                    # Pipeline for sentence lenght
                    # ('body_stats', Pipeline([
                    #     ('position', ItemSelector("body")),
                    #     ('stats', TextStats()),  # returns a list of dicts
                    #     ('len_vect', DictVectorizer()),  # list of dicts -> feature matrix
                    # ])),

                    # Pipeline for standard bag-of-words model for body
                    ('body_bow', Pipeline([
                        ('body', ItemSelector("body")),
                        ('vect', TfidfVectorizer()),
                    ])),
                ],
            )),
            classifier_type
        ])

        # Find best parameters
        # cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        # clf = GridSearchCV(text_clf, parameters, cv=cv, scoring = "f1_weighted", verbose = 2, n_jobs=-1)

        return text_clf

    def best_classifier(self):
        """
        Select best performing classifier.
        """
        max, best, name = 0, None, None

        for classifier in self.classifiers:
            clf, mean = self.train_classifier(classifier)
            if max < mean:
                max = mean
                best = clf
                name = classifier

        print("The Best Classifier is: ", name)


        # For final-best classifier, fit full set
        final = best.fit(self.X, self.y)

        return final

    def corp_downsample(self, corpus):

        positives = corpus.loc[corpus["relation"] == "fullagr"]
        negatives = corpus.loc[corpus["relation"] == "NAN"]
        other = corpus.loc[corpus["relation"] == "ackn"]

        len_pos = len(positives)
        len_neg = len(negatives)

        down_neg = negatives.sample(len_pos, random_state=42)
        pos_indx = positives.index.values.tolist()
        down_neg = down_neg.index.values.tolist()
        other = other.index.values.tolist()


        indexes = pos_indx + down_neg + other
        print("size of our training corpus", len(indexes), len(other), len(negatives))
        downSampled = self.corpus.loc[indexes]
        return downSampled

    def get_informative(self, clf, X, y):

        word_vectorizer = TfidfVectorizer()
        # tags = clf.classes_
        # print(tags)
        # ['NAN' 'ackn' 'fullagr']

        X = X["body"]
        X = word_vectorizer.fit_transform(X)

        # svm = SVC(kernel='linear')
        svm = LogisticRegression()
        svm.fit(X, y)
        
        print(X)

        n = 10

        labelid = list(svm.classes_).index('fullagr')
        feature_names = word_vectorizer.get_feature_names_out()
        topn = sorted(zip(svm.coef_[labelid], feature_names))[-n:]
        for feat, coef in topn:
            print("fullagr", "feature", feat, "coef", coef)

        labelid = list(svm.classes_).index('ackn')
        feature_names = word_vectorizer.get_feature_names_out()
        topn = sorted(zip(svm.coef_[labelid], feature_names))[-n:]
        for feat, coef in topn:
            print("ackn", "feature", feat, "coef", coef)

        labelid = list(svm.classes_).index('NAN')
        feature_names = word_vectorizer.get_feature_names_out()
        topn = sorted(zip(svm.coef_[labelid], feature_names))[-n:]
        for feat, coef in topn:
            print("NAN", "feature", feat, "coef", coef)

    def train_classifier(self, classifier):
        """
        Evaluate a classifier using nested cross validation.
        https://stackoverflow.com/questions/42228735/scikit-learn-gridsearchcv-with-multiple-repetitions/42230764#42230764
        """

        # Find parameters to test
        self.classifier = self.new_classifier(self.params[classifier]["parameters"], self.params[classifier]["mod"])
        print("CREATED NEW CLASSIFIER!!!")
        clf = self.classifier

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
        self.X, self.y, test_size= self.test_size, random_state=42) # getting close - want to go here to get the pos
        print("XXXXXXX")
        #print(X_train)
        with open('readme.txt', 'w') as f:
            f.write(str(X_train))
        # Fit training set
        clf.fit(X_train, y_train)

        self.get_informative(clf, X_train, y_train)

        # Print single fold prediction report
        predicted = clf.predict(X_test)
        print(classification_report(predicted, y_test))
        print(confusion_matrix(y_test, predicted))

        # Nested classifier: http://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#id2
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_score = cross_val_score(clf, self.X, self.y, cv=cv, scoring = "f1_weighted",n_jobs=None)
        mean = np.mean(cv_score)

        print("\n\n##### Name of Classifier: ", classifier, "#####")
        print("\ncv F1-weighted data: ", cv_score)
        print("cv F1-weighted mean: ", mean, "\n")

        return clf, mean
