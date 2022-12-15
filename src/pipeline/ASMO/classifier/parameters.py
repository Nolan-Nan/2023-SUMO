from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class Parameters:
    def __init__(self):
        self.parameters = self.set_params()
        self.classifiers = ["svc", "lr", "nb"]

    @staticmethod
    def set_params():
        parameters_svc = {
        'union__body_bow__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'union__body_bow__vect__use_idf': (True, False),
        'svc__C': [1, 10, 100]
        }
        svc = ('svc', SVC(kernel='linear'))

        parameters_nb = {
        'union__body_bow__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'union__body_bow__vect__use_idf': (True, False),
        }
        nb = ('nb', MultinomialNB())

        parameters_lr = {
        'union__body_bow__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'union__body_bow__vect__use_idf': (True, False),
        }
        lr = ('lr', LogisticRegression())

        mod_setting = {}
        mod_setting["svc"] = {"parameters": parameters_svc, "mod": svc}
        mod_setting["nb"] = {"parameters": parameters_nb, "mod": nb}
        mod_setting["lr"] = {"parameters": parameters_lr, "mod": lr}

        return mod_setting
