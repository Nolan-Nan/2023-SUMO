import sys

from item_selector import ItemSelector
from parameters import Parameters
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

class Classifier:

    def __init__(self, corpus, downsample, info, key, title):
        self.info = info
        self.matrices = []
        self.params = Parameters().parameters
        self.downsample = downsample
        self.title = title
        self.key = key
        self.corpus = corpus
        self.X = self.corpus
        self.y = self.corpus[self.title]
        self.classifiers = Parameters().classifiers
        self.bestClassifier = self.best_classifier()

    def get_classifier(self, parameters, classifier_type):
        # scikit pipeline
        text_clf = Pipeline([

            # Use FeatureUnion to combine the BoW and TextStats features
            ('union', FeatureUnion(
                transformer_list=[

                    #NOTE work in progress
                    # # Pipeline for TextStats features
                    # ('location', Pipeline([
                    #     ('selector', ItemSelector(key='location')),
                    #     ('vect', DictVectorizer()),
                    # ])),

                    # Pipeline for standard bag-of-words model for body
                    ('body_bow', Pipeline([
                        ('selector', ItemSelector(key=self.key)),
                        ('vect', TfidfVectorizer()),
                    ])),

                ],

                # weight components in FeatureUnion
                transformer_weights={
                    # 'location': 1,
                    'body_bow': 1,
                },
            )),

            ('fs', SelectKBest(chi2)), #NOTE expect problems with informative features
            classifier_type
        ])

        # Find best parameters
        clf = GridSearchCV(text_clf, parameters)
        # For paralelism run GridSearchCV(text_clf, parameters, n_jobs=-1) instead

        return clf

    def corp_downsample(self):

        positives = self.corpus.loc[self.corpus[self.title] == "Conc"]
        negatives = self.corpus.loc[self.corpus[self.title] == "None"]
        len_pos = len(positives)
        len_neg = len(negatives)

        down_neg = negatives.sample(len_pos, random_state=42)
        pos_indx = positives.index.values.tolist()
        down_neg = down_neg.index.values.tolist()


        indexes = pos_indx + down_neg
        downSampled = self.corpus.loc[indexes]
        return downSampled

    def best_classifier(self):

        max = 0
        best = None

        for classifier in self.classifiers:
            clf, mean = self.train_classifier(classifier)
            if max < mean:
                max = mean
                best = clf

        # For final classifier
        final = best.fit(self.X, self.y)

        if self.info == True:
            for matrix in self.matrices:
                self.plot_confusion_matrix(matrix[0], ["Conc", "None"], title = matrix[1])
            plt.show()

        return final

    def train_classifier(self, classifier):

        if self.downsample == True:
            down = self.corp_downsample()
            self.X = down
            self.y = down[self.title]

        X_train, X_test, y_train, y_test = train_test_split(
        self.X, self.y, test_size=0.33, random_state=42)

        self.classifier = self.get_classifier(self.params[classifier]["parameters"], self.params[classifier]["mod"])
        clf = self.classifier

        print("FIT")
        clf.fit(X_train, y_train)

        print("CV")
        predicted = clf.predict(X_test)
        cv_score = cross_val_score(clf, self.X, self.y, cv=10)
        mean = np.mean(cv_score)

        if self.info == True:
            print("SHOW")
            self.show_details(clf, classifier, mean, predicted, X_test, y_test, X_train, y_train, self.X, self.y, cv_score)

        print("END")
        mean = 1
        return clf, mean

    def show_details(self, clf, classifier, mean, predicted, X_test, y_test, X_train, y_train, X, y, cv_score):
        print("\n\n##### Name of Classifier: ", classifier, "#####")
        print(classification_report(predicted, y_test))

        # Confusion Matrix
        confMatrix = confusion_matrix(y_test, predicted)
        self.matrices.append([confMatrix, classifier])

        # Accuracies
        print("\ncv accuracy data: ", cv_score)
        print("cv accuracy mean: ", mean, "\n")

        # Mislabeled data
        self.matrix_values(X_test, y_test, predicted)

        # Most informative features
        self.get_informative(clf, X_train, y_train)

        # TODO Learning curve
        # self.plot_learning_curve(clf, classifier, X, y)


    @staticmethod
    def matrix_values(X_test, y_test, predicted):
        mislabeled = X_test[predicted != y_test]
        print("False-Negatives:\n", mislabeled[mislabeled.label == "Conc"])
        print("\nFalse-Positives:\n", mislabeled[mislabeled.label == "None"]) #["body"].values

    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization\n', title)

        print(cm)

        plt.figure(title)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show(block=False)

    def get_informative(self, clf, X, y):
        print("\nMost Informative Features: ")
        best = clf.best_estimator_.named_steps["fs"].get_support() #true if best ft.
        values = clf.best_estimator_.named_steps["fs"].scores_
        best_para1 = clf.best_params_['union__body_bow__vect__use_idf'] #best parameters
        best_para2 = clf.best_params_['union__body_bow__vect__ngram_range']
        tf = TfidfVectorizer(use_idf = best_para1, ngram_range = best_para2)
        tf.fit(X[self.key], y)
        names = pd.DataFrame.from_items([("names", tf.get_feature_names()), ("values", values.tolist())]) #get names of features
        best_features = names[best == True]
        print(best_features.sort_values("values", ascending=False))

    @staticmethod
    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                            train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Generate a simple plot of the test and traning learning curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : integer, cross-validation generator, optional
            If an integer is passed, it is the number of folds (defaults to 3).
            Specific cross-validation objects can be passed, see
            sklearn.cross_validation module for the list of possible objects
        """

        plt.figure()
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=5, n_jobs=1, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.grid("on")
        if ylim:
            plt.ylim(ylim)
        plt.title(title)
        plt.show()
