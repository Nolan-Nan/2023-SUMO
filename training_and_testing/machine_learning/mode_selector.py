
"""
classifiers and feature selection

@author: rozano aulia (original author) / amy conroy (implemented RF classifier)
"""


class mode_selector:
    def __init__(self, location, quotation, asmo,
        sent_length, tfidf_max, rhet_role, wordlist, rhet_y, rel_y,  new_entities, new_cue_phrases):
        self.feature_dict = {
            1 : location,
            2 : quotation,
            3 : asmo,
            4 : sent_length,
            5 : tfidf_max,
            6 : rhet_role,
            7 : wordlist,
            8 : new_entities,
            9 : new_cue_phrases
        }
        self.feature_opt = {
            'location' : 1,
            'quotation' : 2,
            'asmo' : 3,
            'sent_length' : 4,
            'tfidf' : 5,
            'rhet_role' : 6,
            'wordlist' : 7,
            'entities' : 8,
            'cue_phrases' : 9
        }
        self.target_dict = {
            1 : rhet_y,
            2 : rel_y
        }
        self.target_opt = {
            'rhetorical' : 1,
            'relevance' : 2
        }

        #CLASSIFIERS

        from sklearn.tree import DecisionTreeClassifier
        bin_DTC = DecisionTreeClassifier(max_features=None, class_weight='balanced', max_depth=None, min_samples_leaf=1)
        multi_DTC = DecisionTreeClassifier(criterion='gini', max_features=None, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0)

        from sklearn.ensemble import RandomForestClassifier
        bin_rfc = RandomForestClassifier(n_estimators = 150, random_state = 66, class_weight='balanced', max_features=0.6, max_depth=None, min_samples_leaf=45,  min_samples_split=2)
        multi_rfc = RandomForestClassifier(criterion='gini', n_estimators = 150, random_state = 66, max_features=0.6, max_depth=None, min_samples_split=2, min_samples_leaf=45)

        from sklearn.svm import SVC
        bin_SVC = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', class_weight='balanced', decision_function_shape='ovr')
        multi_SVC = SVC(C=1.0, kernel='rbf',degree=3, gamma='scale', decision_function_shape='ovo')

        from sklearn.linear_model import LogisticRegression
        bin_LR = LogisticRegression(class_weight='balanced', solver='liblinear', multi_class='ovr')
        multi_LR = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)

        from sklearn.naive_bayes import BernoulliNB
        bin_NB = BernoulliNB()

        from sklearn.naive_bayes import MultinomialNB
        multi_NB = MultinomialNB()

        from sklearn.neighbors import KNeighborsClassifier
        bin_KN = KNeighborsClassifier(n_neighbors=7, algorithm='auto')
        multi_KN = KNeighborsClassifier(n_neighbors=8, algorithm='auto')

        #using roc for multiclass : https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5

        #binary classifiers
        self.binclf_dict = {
            1 : bin_DTC,
            2 : bin_SVC,
            3 : bin_LR,
            4 : bin_NB,
            5 : bin_KN,
            6 : bin_rfc
        }
        self.multiclf_dict = {
            1 : multi_DTC,
            2 : multi_SVC,
            3 : multi_LR,
            4 : multi_NB,
            5 : multi_KN,
            6 : multi_rfc
        } #options for the user
        self.clf_opt = {
            "DTC" : 1,
            "SVC" : 2,
            "LR" : 3,
            "NB" : 4,
            "KN" : 5,
            "RF" : 6
        }

    def select_features(self, num_of_features):
        import numpy as np
        print("List of features and their key: ")
        print(self.feature_opt)
        feat_names = ""
        for v in range(0, int(num_of_features)):
            feat_key = input("Which feature key? ")
            if v == 0:
                features = self.feature_dict[int(feat_key)]
            else:
                features = np.vstack((features, self.feature_dict[int(feat_key)]))
            feat_names = feat_names + str([k for k,v in self.feature_opt.items() if v == int(feat_key)]) + "+"
        feat_names = feat_names[:-1]
        features = np.vstack((features,)).T
        return features, feat_names

    def select_target(self):
        print("List of targets and their key: ")
        print(self.target_opt)
        target_key = input("Which target key? ")
        target = self.target_dict[int(target_key)]

        # rhetorical classifier
        if int(target_key) == 1:
            label = "multinomial"
        # relevance classifier
        else:
            label = "binary"
        target_name = [k for k,v in self.target_opt.items() if v == int(target_key)]

        return target, label, target_name

    def select_classifier(self, label):
        print("List of classifiers and their key: ")
        print(self.clf_opt)

        clf_key = input("Which classifier key? ")
        if label == "multinomial":
            print("Using multinomial classifier ")
            clf = self.multiclf_dict[int(clf_key)]
        if label == "binary":
            print("Using binary classifier ")
            clf = self.binclf_dict[int(clf_key)]
        clf_name = [k for k,v in self.clf_opt.items() if v == int(clf_key)]
        return clf, clf_name
