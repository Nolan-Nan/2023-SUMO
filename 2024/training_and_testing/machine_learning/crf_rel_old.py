
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crf implemented for relevance classifier

@author: amyconroy
"""

import matplotlib.pyplot as plt
from collections import Counter
plt.style.use('ggplot')

from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

import numpy as np
import csv
import scipy
import matplotlib.pyplot as plt

import pickle

class crf():
    def initialize(self):
         #Target/label
        ##relevance target
        self.rel_y = np.array([])

        ##rhetorical target
        self.rhet_y = np.array([])

        #List of features
        ##for asmo feature-set
        self.agree_X = np.array([])
        self.outcome_X = np.array([])

        ##for location feature-set
        self.loc1_X = np.array([]); self.loc2_X = np.array([]); self.loc3_X = np.array([])
        self.loc4_X = np.array([]); self.loc5_X = np.array([]); self.loc6_X = np.array([])
        self.sentlen_X = np.array([])
        self.rhet_X = np.array([])
        self.tfidf_max_X = np.array([])
        self.tfidf_top20_X = np.array([])
        self.wordlist_X = np.array([])
        self.pasttense_X = np.array([])

        #Hachey and Grover's original features
        self.HGloc1_X = np.array([]); self.HGloc2_X = np.array([]); self.HGloc3_X = np.array([])
        self.HGloc4_X = np.array([]); self.HGloc5_X = np.array([]); self.HGloc6_X = np.array([])
        self.tfidf_HGavg_X = np.array([])
        self.HGsentlen_X = np.array([])
        self.qb_X = np.array([])
        self.inq_X = np.array([])

        ##for entities feature-set
        self.enamex_X = np.array([])
        self.legalent_X = np.array([])

        ## updated entities feature-set
        self.citationent_X = np.array([])

        # spacy entities
        self.loc_ent_X = np.array([])
        self.org_ent_X = np.array([])
        self.date_ent_X = np.array([])
        self.person_ent_X = np.array([])
        self.fac_ent_X = np.array([])
        self.norp_ent_X = np.array([])
        self.gpe_ent_X = np.array([])
        self.event_ent_X = np.array([])
        self.law_ent_X = np.array([])
        self.time_ent_X = np.array([])
        self.work_of_art_ent_X = np.array([])
        self.ordinal_ent_X = np.array([])
        self.cardinal_ent_X = np.array([])
        self.money_ent_X = np.array([])
        self.percent_ent_X = np.array([])
        self.product_ent_X = np.array([])
        self.quantity_ent_X = np.array([])
        self.spacy = np.array([])
        self.total_spacy_X = np.array([])

    # all values are 0, thus non-beneficial in ml
        # self.caseent_X = np.array([])
        ##for cue phrase feature-set
        self.asp_X = np.array([])
        self.modal_X = np.array([])
        self.voice_X = np.array([])
        self.negcue_X = np.array([])
        self.tense_X = np.array([])


        self.location = np.array([])
        self.quotation = np.array([])
        self.asmo = np.array([])
        self.cue_phrase = np.array([])
        self.sent_length = np.array([])
        self.tfidf_top20 = np.array([])
        self.rhet_role = np.array([])
        self.spacy = np.array([])

        # other data
        self.judgename = []
        self.rhetlabel = []

        # new cue phrases
        # modal data on the entire sentence (count and boolean values)
        self.modal_dep_bool_X = np.array([])
        self.modal_dep_count_X = np.array([])

        # verb data on the first verb
        self.new_modal_X = np.array([])
        self.new_tense_X = np.array([])
        self.new_dep_X = np.array([])
        self.new_tag_X = np.array([])
        self.new_negative_X = np.array([])
        self.new_stop_X = np.array([])
        self.new_voice_X = np.array([])

        # data on the token after the verb
        self.second_pos_X = np.array([])
        self.second_dep_X = np.array([])
        self.second_tag_X = np.array([])
        self.second_stop_X = np.array([])


    def train_crf(self, rare_feat_cutoff=5, trace=3):

        from seq_init import create_tagged_sentences_list
        sentences_list = []

        # TRAIN THEN SAVE
        self.initialize()
        self.pull_training_data()

        X_train = self.create_speeches_features_list()
        y_train = self.sent_to_rellabel()
      #  print(y_train)

        self.initialize()
        self.pull_testing_data()
        X_test = self.create_speeches_features_list()
        y_test = self.sent_to_rellabel()
       # print(y_test)


        '''# TEST / TRAIN
        crf = sklearn_crfsuite.CRF(
              algorithm='lbfgs',
              c1=0.5,
              c2=0.05,
              max_iterations=100,
              all_possible_transitions=True
              )

        crf.fit(X_train, y_train)

        labels = list(crf.classes_)
      #  labels.remove('NONE')
        print(labels)

        y_pred = crf.predict(X_test)
        print(y_pred)
        print(metrics.flat_f1_score(y_test, y_pred,
                      average='micro', labels=labels))
        y_test = [label for y in y_test for label in y]
        y_pred = [label for y in y_pred for label in y]
        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
            )

        print(sklearn.metrics.classification_report(
            y_test, y_pred, labels=sorted_labels, digits=3
            ))
        

        print("Top likely transitions:")

        self.print_transitions(Counter(crf.transition_features_).most_common(20))

        print("\nTop unlikely transitions:")
        self.print_transitions(Counter(crf.transition_features_).most_common()[-20:])

        print("Top positive:")
        self.print_state_features(Counter(crf.state_features_).most_common(30))

        print("\nTop negative:")
        self.print_state_features(Counter(crf.state_features_).most_common()[-30:])

        import eli5
        from PIL import Image

        expl = eli5.explain_weights(crf, top=10)
        print(expl)
        print((expl.targets[0].target, expl.targets[0].score, expl.targets[0].proba))
        text = eli5.formatters.text.format_as_text(expl)
        print(text)


# CROSS VAL
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=100,
            all_possible_transitions=True
            )
        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
            }



# use the same metric for evaluation
        f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='micro', labels=np.unique(labels))




# search

        rs = RandomizedSearchCV(crf, params_space,
                        cv=10,
                        verbose=1,
                        n_jobs=1,
                        n_iter=50,
                        scoring=f1_scorer)
        rs.fit(X_train, y_train)

        # crf = rs.best_estimator_
        print('best params:', rs.best_params_)
        print('best CV score:', rs.best_score_)
        print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

        crf = rs.best_estimator_'''
        f = open("RELEVANCE.pickle", "rb")
        crf = pickle.load(f)
        f.close()
        y_pred = crf.predict(X_test)
        print(y_pred)
        labels = list(crf.classes_)
      #  labels.remove('NONE')
        print(labels)
        y_pred = [label for y in y_pred for label in y]
        print(sklearn.metrics.classification_report(
            y_test, y_pred, labels=labels, digits=3
        ))

        print(len(Counter(crf.transition_features_).most_common(20)))

        from sklearn.metrics import plot_confusion_matrix
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(y_test, y_pred))
     #   plot_confusion_matrix(crf, X_test, y_test)  # doctest: +SKIP
     #   plt.show()

        '''f = open('RELEVANCE.pickle', 'wb')
        pickle.dump(crf, f)
        f.close()'''

# =============================================================================
#         print("Top positive:")
#         self.print_state_features(Counter(crf.state_features_).most_common(30))
#
#         print("\nTop negative:")
#         self.print_state_features(Counter(crf.state_features_).most_common()[-30:])
# =============================================================================

        import eli5
        expl = eli5.explain_weights(crf, top=10)
        print(expl)
        print((expl.targets[0].target, expl.targets[0].score, expl.targets[0].proba))
        text = eli5.formatters.text.format_as_text(expl)
        print(text)





# =============================================================================
#         f = open("crf.pickle", "rb")
#         classifier = pickle.load(f)
#         f.close()
#         _x = [s.parameters['c1'] for s in crf.grid_scores_]
#         _y = [s.parameters['c2'] for s in crf.grid_scores_]
#         _c = [s.mean_validation_score for s in crf.grid_scores_]
#
#         fig = plt.figure()
#
#         fig.set_size_inches(12, 12)#
#         ax = plt.gca()
#         ax.set_yscale('log')
#         ax.set_xscale('log')
#         ax.set_xlabel('C1')
#         ax.set_ylabel('C2')
#         ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(
#              min(_c), max(_c)
#         ))
#
#         ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0,0,0])
#
#         print("Dark blue => {:0.4}, dark red => {:0.4}".format(min(_c), max(_c)))
# =============================================================================

    def print_state_features(self, state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-8s %s" % (weight, label, attr))

    def print_transitions(self, trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

    def sent_to_rhetlabel(self):
        labels = self.rhetlabel

        rhet_labels = []

        for label in labels:
            if label == 'FACT':
                label = '2.0'
            if label == 'PROCEEDINGS':
                label = '3.0'
            if label == 'BACKGROUND':
                label = '4.0'
            if label == 'FRAMING':
                label = '5.0'
            if label == 'DISPOSAL':
                label = '6.0'
            if label == 'TEXTUAL':
                label = '1.0'
            if label == 'NONE':
                label = '0.0'
            individual_label = []
            individual_label.append(label)
            rhet_labels.append(individual_label)

        return rhet_labels

    def sent_to_rellabel(self):
        labels = self.rel_y
        print("TEST LABELS")
       # print(labels)
        print(len(labels))
        rhet_labels = []


        for label in labels:
            label = (int(label))
            if label == 1:
                label = 'yes'
            elif label == 0:
                label = 'no'

            individual_label = []
            individual_label.append(label)
            rhet_labels.append(individual_label)

        return rhet_labels


    def create_speeches_features_list(self):
        all_featureset = []

        # init
        previous_judgename = ''
        y = 0
        newspeech = True
        featureset = []
        tagcount = 0 # this is the counter for each sentence in a speech
        judges = self.judgename
        newSpeechLookAheadBy1 = False # checks if the judges are different
        newSpeechLookAheadBy2 = False # indicates a new speech
        tags = self.rhetlabel
        tag_history = self.sent_to_rhetlabel()
        relevant = self.sent_to_rellabel()

      #  print(judges)

        for judge in judges:
             featureset = []
             newSpeechLookAheadBy1 = False
             newSpeechLookAheadBy2 = False
             tag = tags[y]
             rel = relevant[y]

       #      print(judge)
      #       print(len(judges))

             if len(judges) == y+2:
                 newSpeechLookAheadBy2 = True
             elif len(judges) == y+1:
                 newSpeechLookAheadBy1 = True
             elif judges[y+1] != judge:
                 newSpeechLookAheadBy1 = True
             elif judges[y+2] != judge:
                 newSpeechLookAheadBy2 = True
             if judge != previous_judgename:
                 tagcount = 1
                 newspeech = True
                 rel_history = []
               #  tag_history = [] # previously assigned tags for that speech
                 featureset.append(self.get_features(tagcount, y, tag_history, newspeech,
                                                     newSpeechLookAheadBy1, newSpeechLookAheadBy2, tag, rel_history))
                 all_featureset.append(featureset)
                 rel_history.append(rel)
                # tag_history.append(tag)
             #    print(tag_history)
                 y += 1
                 tagcount += 1
             else:
                 newspeech = False
                 featureset.append(self.get_features(tagcount, y, tag_history, newspeech,
                                                      newSpeechLookAheadBy1, newSpeechLookAheadBy2, tag, rel_history))
                 all_featureset.append(featureset)
                 rel_history.append(rel)
               #  tag_history.append(tag)
             #    print(tag_history)
                 y += 1
                 tagcount += 1
             previous_judgename = judge

        print(len(all_featureset))
        return all_featureset

    def get_features(self, sentence_id, y, tag_history, newspeech, newSpeechLookAheadBy1, newSpeechLookAheadBy2, tag, rel_history):
            # creates a dict
        # rhetorical tags are strings, all others are int (null if the start of the sentence)
            sentence_id = (int(sentence_id))


            sentence_features = {}

  # NB - need not a super long way to access the 2D array
        # s refers to the sentence, r refers to rhetorical role
        # ensure that we don't go out of bounds
        # this is not going to safeguard against going past the end of a speech
     #   if self.sent_length[y+1] != None and self.sent_length[y+2] != None:
            if newspeech: # first sentence of a speech, sentence 0 reserved for a new case start
                sentence_features.update({"r-1" : "<START>",
                                      "r-2 r-1" : "<START> <START>", # previous label and current features
                                      'bias': 1.0,
                                      "r" : tag,
                                      "rel-1" : "<START>",
                                      "rel-2 rel-1" : "<START> <START>",
                                      "length" : (self.sent_length[y]),
                                      "length+1" : (self.sent_length[y+1]),
                                      "length+2" : (self.sent_length[y+2]),
                                      "tfdif" : (self.tfidf_top20[y]),
                                      "tfdif+1" : (self.tfidf_top20[y+1]),
                                      "tfdif+2" : (self.tfidf_top20[y+2]),
                                      "loc1" : (self.loc1_X[y]),
                                      "loc1+1" : (self.loc1_X[y+1]),
                                      "loc1+2" : (self.loc1_X[y+2]),
                                      "loc2" : (self.loc2_X[y]),
                                      "loc2+1" : (self.loc2_X[y+1]),
                                      "loc2+2" : (self.loc2_X[y+2]),
                                      "loc3" : (self.loc3_X[y]),
                                      "loc3+1" : (self.loc3_X[y+1]),
                                      "loc3+2" : (self.loc3_X[y+2]),
                                      "loc4" : (self.loc4_X[y]),
                                      "loc4+1" : (self.loc4_X[y+1]),
                                      "loc4+2" : (self.loc4_X[y+2]),
                                      "loc5" : (self.loc5_X[y]),
                                      "loc5+1" : (self.loc5_X[y+1]),
                                      "loc5+2" : (self.loc5_X[y+2]),
                                      "loc6" : (self.loc6_X[y]),
                                      "loc6+1" : (self.loc6_X[y+1]),
                                      "loc6+2" : (self.loc6_X[y+2]),
                                      "quote1" : (self.inq_X[y]),
                                      "quote1+1" : (self.inq_X[y+1]),
                                      "quote1+2" : (self.inq_X[y+2]),
                                      "quote2" : (self.qb_X[y]),
                                      "quote2+1" : (self.qb_X[y+1]),
                                      "quote2+2" : (self.qb_X[y+2]),
                                      "asmo1" : (self.agree_X[y]),
                                      "asmo1+1" : (self.agree_X[y+1]),
                                      "asmo1+2" : (self.agree_X[y+2]),
                                      "asmo2" : (self.outcome_X[y]),
                                      "asmo2+1" : (self.outcome_X[y+1]),
                                      "asmo2+2" : (self.outcome_X[y+2]),
                                      "cue1" : (self.modal_dep_bool_X[y]),
                                      "cue1+1" : (self.modal_dep_bool_X[y+1]),
                                      "cue1+2" : (self.modal_dep_bool_X[y+2]),
                                      "cue2" : (self.modal_dep_count_X[y]),
                                      "cue2+1" : (self.modal_dep_count_X[y+1]),
                                      "cue2+2" : (self.modal_dep_count_X[y+2]),
                                      "cue3" : (self.new_modal_X[y]),
                                      "cue3+1" : (self.new_modal_X[y+1]),
                                      "cue3+2" : (self.new_modal_X[y+2]),
                                      "cue4" : (self.new_tense_X[y]),
                                      "cue4+1" : (self.new_tense_X[y+1]),
                                      "cue4+2" : (self.new_tense_X[y+2]),
                                      "cue5" : (self.new_dep_X[y]),
                                      "cue5+1" : (self.new_dep_X[y+1]),
                                      "cue5+2" : (self.new_dep_X[y+2]),
                                      "cue6" : (self.new_tag_X[y]),
                                      "cue6+1" : (self.new_tag_X[y+1]),
                                      "cue6+2" : (self.new_tag_X[y+2]),
                                      "cue7" : (self.new_negative_X[y]),
                                      "cue7+1" : (self.new_negative_X[y+1]),
                                      "cue7+2" : (self.new_negative_X[y+2]),
                                      "cue8" : (self.new_stop_X[y]),
                                      "cue8+1" : (self.new_stop_X[y+1]),
                                      "cue8+2" : (self.new_stop_X[y+2]),
                                      "cue9" : (self.new_voice_X[y]),
                                      "cue9+1" : (self.new_voice_X[y+1]),
                                      "cue9+2" : (self.new_voice_X[y+2]),
                                      "cue10" : (self.second_pos_X[y]),
                                      "cue10+1" : (self.second_pos_X[y+1]),
                                      "cue10+2" : (self.second_pos_X[y+2]),
                                      "cue11" : (self.second_dep_X[y]),
                                      "cue11+1" : (self.second_dep_X[y+1]),
                                      "cue11+2" : (self.second_dep_X[y+2]),
                                      "cue12" : (self.second_tag_X[y]),
                                      "cue12+1" : (self.second_tag_X[y+1]),
                                      "cue12+2" : (self.second_tag_X[y+2]),
                                      "cue13" : (self.second_stop_X[y]),
                                      "cue13+1" : (self.second_stop_X[y+1]),
                                      "cue13+2" : (self.second_stop_X[y+2]),
                                                                            "spacy1" : (self.loc_ent_X[y]),
                                      "spacy1+1" : (self.loc_ent_X[y+1]),
                                      "spacy1+2" : (self.loc_ent_X[y+2]),
                                      "spacy2" : (self.org_ent_X[y]),
                                      "spacy2+1" : (self.org_ent_X[y+1]),
                                      "spacy2+2" : (self.org_ent_X[y+2]),
                                      "spacy3" : (self.date_ent_X[y]),
                                      "spacy3+1" : (self.date_ent_X[y+1]),
                                      "spacy3+2" : (self.date_ent_X[y+2]),
                                      "spacy4" : (self.person_ent_X[y]),
                                      "spacy4+1" : (self.person_ent_X[y+1]),
                                      "spacy4+2" : (self.person_ent_X[y+2]),
                                      "spacy5": (self.fac_ent_X[y]),
                                      "spacy5+1": (self.fac_ent_X[y + 1]),
                                      "spacy5+2": (self.fac_ent_X[y + 2]),
                                      "spacy6": (self.norp_ent_X[y]),
                                      "spacy6+1": (self.norp_ent_X[y + 1]),
                                      "spacy6+2": (self.norp_ent_X[y + 2]),
                                      "spacy7": (self.gpe_ent_X[y]),
                                      "spacy7+1": (self.gpe_ent_X[y + 1]),
                                      "spacy7+2": (self.gpe_ent_X[y + 2]),
                                      "spacy8": (self.event_ent_X[y]),
                                      "spacy8+1": (self.event_ent_X[y + 1]),
                                      "spacy8+2": (self.event_ent_X[y + 2]),
                                      "spacy9": (self.law_ent_X[y]),
                                      "spacy9+1": (self.law_ent_X[y + 1]),
                                      "spacy9+2": (self.law_ent_X[y + 2]),
                                      "spacy10": (self.time_ent_X[y]),
                                      "spacy10+1": (self.time_ent_X[y + 1]),
                                      "spacy10+2": (self.time_ent_X[y + 2]),
                                      "spacy11": (self.work_of_art_ent_X[y]),
                                      "spacy11+1": (self.work_of_art_ent_X[y + 1]),
                                      "spacy11+2": (self.work_of_art_ent_X[y + 2]),
                                      "spacy12": (self.ordinal_ent_X[y]),
                                      "spacy12+1": (self.ordinal_ent_X[y + 1]),
                                      "spacy12+2": (self.ordinal_ent_X[y + 2]),
                                      "spacy13": (self.cardinal_ent_X[y]),
                                      "spacy13+1": (self.cardinal_ent_X[y + 1]),
                                      "spacy13+2": (self.cardinal_ent_X[y + 2]),
                                      "spacy14": (self.money_ent_X[y]),
                                      "spacy14+1": (self.money_ent_X[y + 1]),
                                      "spacy14+2": (self.money_ent_X[y + 2]),
                                      "spacy15": (self.percent_ent_X[y]),
                                      "spacy15+1": (self.percent_ent_X[y + 1]),
                                      "spacy15+2": (self.percent_ent_X[y + 2]),
                                      "spacy16": (self.product_ent_X[y]),
                                      "spacy16+1": (self.product_ent_X[y + 1]),
                                      "spacy16+2": (self.product_ent_X[y + 2]),
                                      "spacy17": (self.quantity_ent_X[y]),
                                      "spacy17+1": (self.quantity_ent_X[y + 1]),
                                      "spacy17+2": (self.quantity_ent_X[y + 2]),

                                          })
        # second word of the sentence
            elif sentence_id == 2:
                sentence_features.update({"r-1" : tag_history[y-1],
                                      "r-2 r-1" : "<START> %s" % (tag_history[y-1]),
                                      'bias': 1.0,
                                      "r" : tag,
                                      "rel-1" : rel_history[sentence_id-2],
                                      "rel-2 rel-1" : "<START> %s" % (rel_history[sentence_id-2]),
                                      "length" : (self.sent_length[y]),
                                      "length+1" : (self.sent_length[y+1]),
                                      "length+2" : (self.sent_length[y+2]),
                                      "length-1" : (self.sent_length[y-1]),
                                      "tfdif" : (self.tfidf_top20[y]),
                                      "tfdif+1" : (self.tfidf_top20[y+1]),
                                      "tfdif+2" : (self.tfidf_top20[y+2]),
                                      "tfdif-1" : (self.tfidf_top20[y-1]),
                                      "loc1" : (self.loc1_X[y]),
                                      "loc1+1" : (self.loc1_X[y+1]),
                                      "loc1+2" : (self.loc1_X[y+2]),
                                      "loc1-1" : (self.loc1_X[y-1]),
                                      "loc2" : (self.loc2_X[y]),
                                      "loc2+1" : (self.loc2_X[y+1]),
                                      "loc2+2" : (self.loc2_X[y+2]),
                                      "loc2-1" : (self.loc1_X[y-1]),
                                      "loc3" : (self.loc3_X[y]),
                                      "loc3+1" : (self.loc3_X[y+1]),
                                      "loc3+2" : (self.loc3_X[y+2]),
                                      "loc3-1" : (self.loc1_X[y-1]),
                                      "loc4" : (self.loc4_X[y]),
                                      "loc4+1" : (self.loc4_X[y+1]),
                                      "loc4+2" : (self.loc4_X[y+2]),
                                      "loc4-1" : (self.loc1_X[y-1]),
                                      "loc5" : (self.loc5_X[y]),
                                      "loc5+1" : (self.loc5_X[y+1]),
                                      "loc5+2" : (self.loc5_X[y+2]),
                                      "loc5-1" : (self.loc1_X[y-1]),
                                      "loc6" : (self.loc6_X[y]),
                                      "loc6+1" : (self.loc6_X[y+1]),
                                      "loc6+2" : (self.loc6_X[y+2]),
                                      "loc6-1" : (self.loc1_X[y-1]),
                                      "quote1" : (self.inq_X[y]),
                                      "quote1+1" : (self.inq_X[y+1]),
                                      "quote1+2" : (self.inq_X[y+2]),
                                      "quote1-1" : (self.inq_X[y-1]),
                                      "quote2" : (self.qb_X[y]),
                                      "quote2+1" : (self.qb_X[y+1]),
                                      "quote2+2" : (self.qb_X[y+2]),
                                      "quote2-1" : (self.qb_X[y-1]),
                                      "asmo1" : (self.agree_X[y]),
                                      "asmo1+1" : (self.agree_X[y+1]),
                                      "asmo1+2" : (self.agree_X[y+2]),
                                      "asmo1-1" : (self.agree_X[y-1]),
                                      "asmo2" : (self.outcome_X[y]),
                                      "asmo2+1" : (self.outcome_X[y+1]),
                                      "asmo2+2" : (self.outcome_X[y+2]),
                                      "asmo2-1" : (self.outcome_X[y-1]),
                                      "cue1" : (self.modal_dep_bool_X[y]),
                                      "cue1+1" : (self.modal_dep_bool_X[y+1]),
                                      "cue1+2" : (self.modal_dep_bool_X[y+2]),
                                      "cue1-1" : (self.modal_dep_bool_X[y-1]),
                                      "cue2" : (self.modal_dep_count_X[y]),
                                      "cue2+1" : (self.modal_dep_count_X[y+1]),
                                      "cue2+2" : (self.modal_dep_count_X[y+2]),
                                      "cue2-1" : (self.modal_dep_count_X[y-1]),
                                      "cue3" : (self.new_modal_X[y]),
                                      "cue3+1" : (self.new_modal_X[y+1]),
                                      "cue3+2" : (self.new_modal_X[y+2]),
                                      "cue3-1" : (self.new_modal_X[y-1]),
                                      "cue4" : (self.new_tense_X[y]),
                                      "cue4+1" : (self.new_tense_X[y+1]),
                                      "cue4+2" : (self.new_tense_X[y+2]),
                                      "cue4-1" : (self.new_tense_X[y-1]),
                                      "cue5" : (self.new_dep_X[y]),
                                      "cue5+1" : (self.new_dep_X[y+1]),
                                      "cue5+2" : (self.new_dep_X[y+2]),
                                      "cue5-1" : (self.new_dep_X[y-1]),
                                      "cue6" : (self.new_tag_X[y]),
                                      "cue6+1" : (self.new_tag_X[y+1]),
                                      "cue6+2" : (self.new_tag_X[y+2]),
                                      "cue6-1" : (self.new_tag_X[y-1]),
                                      "cue7" : (self.new_negative_X[y]),
                                      "cue7+1" : (self.new_negative_X[y+1]),
                                      "cue7+2" : (self.new_negative_X[y+2]),
                                      "cue7-1" : (self.new_negative_X[y-1]),
                                      "cue8" : (self.new_stop_X[y]),
                                      "cue8+1" : (self.new_stop_X[y+1]),
                                      "cue8+2" : (self.new_stop_X[y+2]),
                                      "cue8-1" : (self.new_stop_X[y-1]),
                                      "cue9" : (self.new_voice_X[y]),
                                      "cue9+1" : (self.new_voice_X[y+1]),
                                      "cue9+2" : (self.new_voice_X[y+2]),
                                      "cue9-1" : (self.new_voice_X[y-1]),
                                      "cue10" : (self.second_pos_X[y]),
                                      "cue10+1" : (self.second_pos_X[y+1]),
                                      "cue10+2" : (self.second_pos_X[y+2]),
                                      "cue10-1" : (self.second_pos_X[y-1]),
                                      "cue11" : (self.second_dep_X[y]),
                                      "cue11+1" : (self.second_dep_X[y+1]),
                                      "cue11+2" : (self.second_dep_X[y+2]),
                                      "cue11-1" : (self.second_dep_X[y-1]),
                                      "cue12" : (self.second_tag_X[y]),
                                      "cue12+1" : (self.second_tag_X[y+1]),
                                      "cue12+2" : (self.second_tag_X[y+2]),
                                      "cue12-1" : (self.second_tag_X[y-1]),
                                      "cue13" : (self.second_stop_X[y]),
                                      "cue13+1" : (self.second_stop_X[y+1]),
                                      "cue13+2" : (self.second_stop_X[y+2]),
                                      "cue13-1" : (self.second_stop_X[y-1]),
                                      "spacy1" : (self.loc_ent_X[y]),
                                      "spacy1+1" : (self.loc_ent_X[y+1]),
                                      "spacy1+2" : (self.loc_ent_X[y+2]),
                                      "spacy1-1" : (self.loc_ent_X[y-1]),
                                      "spacy2" : (self.org_ent_X[y]),
                                      "spacy2+1" : (self.org_ent_X[y+1]),
                                      "spacy2+2" : (self.org_ent_X[y+2]),
                                      "spacy2-1" : (self.org_ent_X[y-1]),
                                      "spacy3" : (self.date_ent_X[y]),
                                      "spacy3+1" : (self.date_ent_X[y+1]),
                                      "spacy3+2" : (self.date_ent_X[y+2]),
                                      "spacy3-1" : (self.date_ent_X[y-1]),
                                      "spacy4" : (self.person_ent_X[y]),
                                      "spacy4+1" : (self.person_ent_X[y+1]),
                                      "spacy4+2" : (self.person_ent_X[y+2]),
                                      "spacy4-1" : (self.person_ent_X[y-1]),
                                      "spacy5": (self.fac_ent_X[y]),
                                      "spacy5+1": (self.fac_ent_X[y + 1]),
                                      "spacy5+2": (self.fac_ent_X[y + 2]),
                                      "spacy5-1": (self.fac_ent_X[y - 1]),
                                      "spacy6": (self.norp_ent_X[y]),
                                      "spacy6+1": (self.norp_ent_X[y + 1]),
                                      "spacy6+2": (self.norp_ent_X[y + 2]),
                                      "spacy6-1": (self.norp_ent_X[y - 1]),
                                      "spacy7": (self.gpe_ent_X[y]),
                                      "spacy7+1": (self.gpe_ent_X[y + 1]),
                                      "spacy7+2": (self.gpe_ent_X[y + 2]),
                                      "spacy7-1": (self.gpe_ent_X[y - 1]),
                                      "spacy8": (self.event_ent_X[y]),
                                      "spacy8+1": (self.event_ent_X[y + 1]),
                                      "spacy8+2": (self.event_ent_X[y + 2]),
                                      "spacy8-1": (self.event_ent_X[y - 1]),
                                      "spacy9": (self.law_ent_X[y]),
                                      "spacy9+1": (self.law_ent_X[y + 1]),
                                      "spacy9+2": (self.law_ent_X[y + 2]),
                                      "spacy9-1": (self.law_ent_X[y - 1]),
                                      "spacy10": (self.time_ent_X[y]),
                                      "spacy10+1": (self.time_ent_X[y + 1]),
                                      "spacy10+2": (self.time_ent_X[y + 2]),
                                      "spacy10-1": (self.time_ent_X[y - 1]),
                                      "spacy11": (self.work_of_art_ent_X[y]),
                                      "spacy11+1": (self.work_of_art_ent_X[y + 1]),
                                      "spacy11+2": (self.work_of_art_ent_X[y + 2]),
                                      "spacy11-1": (self.work_of_art_ent_X[y - 1]),
                                      "spacy12": (self.ordinal_ent_X[y]),
                                      "spacy12+1": (self.ordinal_ent_X[y + 1]),
                                      "spacy12+2": (self.ordinal_ent_X[y + 2]),
                                      "spacy12-1": (self.ordinal_ent_X[y - 1]),
                                      "spacy13": (self.cardinal_ent_X[y]),
                                      "spacy13+1": (self.cardinal_ent_X[y + 1]),
                                      "spacy13+2": (self.cardinal_ent_X[y + 2]),
                                      "spacy13-1": (self.cardinal_ent_X[y - 1]),
                                      "spacy14": (self.money_ent_X[y]),
                                      "spacy14+1": (self.money_ent_X[y + 1]),
                                      "spacy14+2": (self.money_ent_X[y + 2]),
                                      "spacy14-1": (self.money_ent_X[y - 1]),
                                      "spacy15": (self.percent_ent_X[y]),
                                      "spacy15+1": (self.percent_ent_X[y + 1]),
                                      "spacy15+2": (self.percent_ent_X[y + 2]),
                                      "spacy15-1": (self.percent_ent_X[y - 1]),
                                      "spacy16": (self.product_ent_X[y]),
                                      "spacy16+1": (self.product_ent_X[y + 1]),
                                      "spacy16+2": (self.product_ent_X[y + 2]),
                                      "spacy16-1": (self.product_ent_X[y - 1]),
                                      "spacy17": (self.quantity_ent_X[y]),
                                      "spacy17+1": (self.quantity_ent_X[y + 1]),
                                      "spacy17+2": (self.quantity_ent_X[y + 2]),
                                      "spacy17-1": (self.quantity_ent_X[y - 1]),

                                          })

            elif newSpeechLookAheadBy1:
                sentence_features.update({"r-1" : tag_history[y-1],
                                      "r-2 r-1" : "%s %s" % (tag_history[y-2], tag_history[y-1]),
                                      'bias': 1.0,
                                      "r" : tag,
                                      "rel-1" : rel_history[sentence_id-2],
                                      "rel-2 rel-1" : "%s %s" % (rel_history[sentence_id-3], rel_history[sentence_id-2]),
                                      "length" : (self.sent_length[y]),
                                      "length-1" : (self.sent_length[y-1]),
                                      "length-2" : (self.sent_length[y-2]),
                                      "tfdif" : (self.tfidf_top20[y]),
                                      "tfdif-1" : (self.tfidf_top20[y-1]),
                                      "tfdif-2" : (self.tfidf_top20[y-2]),
                                      "loc1" : (self.loc1_X[y]),
                                      "loc1-1" : (self.loc1_X[y-1]),
                                      "loc1-2" : (self.loc1_X[y-2]),
                                      "loc2" : (self.loc1_X[y]),
                                      "loc2-1" : (self.loc1_X[y-1]),
                                      "loc2-2" : (self.loc1_X[y-2]),
                                      "loc3" : (self.loc1_X[y]),
                                      "loc3-1" : (self.loc1_X[y-1]),
                                      "loc3-2" : (self.loc1_X[y-2]),
                                      "loc4" : (self.loc1_X[y]),
                                      "loc4-1" : (self.loc1_X[y-1]),
                                      "loc4-2" : (self.loc1_X[y-2]),
                                      "loc5" : (self.loc1_X[y]),
                                      "loc5-1" : (self.loc1_X[y-1]),
                                      "loc5-2" : (self.loc1_X[y-2]),
                                      "loc6" :  (self.loc1_X[y]),
                                      "loc6-1" : (self.loc1_X[y-1]),
                                      "loc6-2" : (self.loc1_X[y-2]),
                                      "quote1" : (self.inq_X[y]),
                                      "quote1-1" : (self.inq_X[y-1]),
                                      "quote1-2" : (self.inq_X[y-2]),
                                      "quote2" :  (self.qb_X[y]),
                                      "quote2-1" : (self.qb_X[y-1]),
                                      "quote2-2" : (self.inq_X[y-2]),
                                      "asmo1" : (self.agree_X[y]),
                                      "asmo1-1" : (self.agree_X[y-1]),
                                      "asmo1-2" : (self.agree_X[y-2]),
                                      "asmo2" : (self.outcome_X[y]),
                                      "asmo2-1" : (self.outcome_X[y-1]),
                                      "asmo2-2" : (self.outcome_X[y-2]),
                                      "cue1" : (self.modal_dep_bool_X[y]),
                                      "cue1-2" : (self.modal_dep_bool_X[y-2]),
                                      "cue1-1" : (self.modal_dep_bool_X[y-1]),
                                      "cue2" : (self.modal_dep_count_X[y]),
                                      "cue2-2" : (self.modal_dep_count_X[y-2]),
                                      "cue2-1" : (self.modal_dep_count_X[y-1]),
                                      "cue3" : (self.new_modal_X[y]),
                                      "cue3-1" : (self.new_modal_X[y-1]),
                                      "cue3-2" : (self.new_modal_X[y-2]),
                                      "cue4" : (self.new_tense_X[y]),
                                      "cue4-2" : (self.new_tense_X[y-2]),
                                      "cue4-1" : (self.new_tense_X[y-1]),
                                      "cue5" : (self.new_dep_X[y]),
                                      "cue5-2" : (self.new_dep_X[y-2]),
                                      "cue5-1" : (self.new_dep_X[y-1]),
                                      "cue6" : (self.new_tag_X[y]),
                                      "cue6-2" : (self.new_tag_X[y-2]),
                                      "cue6-1" : (self.new_tag_X[y-1]),
                                      "cue7" : (self.new_negative_X[y]),
                                      "cue7-2" : (self.new_negative_X[y-2]),
                                      "cue7-1" : (self.new_negative_X[y-1]),
                                      "cue8" : (self.new_stop_X[y]),
                                      "cue8-2" : (self.new_stop_X[y-2]),
                                      "cue8-1" : (self.new_stop_X[y-1]),
                                      "cue9" : (self.new_voice_X[y]),
                                      "cue9-2" : (self.new_voice_X[y-2]),
                                      "cue9-1" : (self.new_voice_X[y-1]),
                                      "cue10" : (self.second_pos_X[y]),
                                      "cue10-2" : (self.second_pos_X[y-2]),
                                      "cue10-1" : (self.second_pos_X[y-1]),
                                      "cue11" : (self.second_dep_X[y]),
                                      "cue11-2" : (self.second_dep_X[y-2]),
                                      "cue11-1" : (self.second_dep_X[y-1]),
                                      "cue12" : (self.second_tag_X[y]),
                                      "cue12-2" : (self.second_tag_X[y-2]),
                                      "cue12-1" : (self.second_tag_X[y-1]),
                                      "cue13" : (self.second_stop_X[y]),
                                      "cue13-2" : (self.second_stop_X[y-2]),
                                      "cue13-1" : (self.second_stop_X[y-1]),

                                      "spacy1" : (self.loc_ent_X[y]),
                                      "spacy1-1" : (self.loc_ent_X[y-1]),
                                      "spacy1-2" : (self.loc_ent_X[y-2]),
                                      "spacy2" : (self.org_ent_X[y]),
                                      "spacy2-1" : (self.org_ent_X[y-1]),
                                      "spacy2-2" : (self.org_ent_X[y-2]),
                                      "spacy3" : (self.date_ent_X[y]),
                                      "spacy3-1" : (self.date_ent_X[y-1]),
                                      "spacy3-2" : (self.date_ent_X[y-2]),
                                      "spacy4" : (self.person_ent_X[y]),
                                      "spacy4-1" : (self.person_ent_X[y-1]),
                                      "spacy4-2" : (self.person_ent_X[y-2]),
                                      "spacy5": (self.fac_ent_X[y]),
                                      "spacy5-1": (self.fac_ent_X[y - 1]),
                                      "spacy5-2": (self.fac_ent_X[y - 2]),
                                      "spacy6": (self.norp_ent_X[y]),
                                      "spacy6-1": (self.norp_ent_X[y - 1]),
                                      "spacy6-2": (self.norp_ent_X[y - 2]),
                                      "spacy7": (self.gpe_ent_X[y]),
                                      "spacy7-1": (self.gpe_ent_X[y - 1]),
                                      "spacy7-2": (self.gpe_ent_X[y - 2]),
                                      "spacy8": (self.event_ent_X[y]),
                                      "spacy8-1": (self.event_ent_X[y - 1]),
                                      "spacy8-2": (self.event_ent_X[y - 2]),
                                      "spacy9": (self.law_ent_X[y]),
                                      "spacy9-1": (self.law_ent_X[y - 1]),
                                      "spacy9-2": (self.law_ent_X[y - 2]),
                                      "spacy10": (self.time_ent_X[y]),
                                      "spacy10-1": (self.time_ent_X[y - 1]),
                                      "spacy10-2": (self.time_ent_X[y - 2]),
                                      "spacy11": (self.work_of_art_ent_X[y]),
                                      "spacy11-1": (self.work_of_art_ent_X[y - 1]),
                                      "spacy11-2": (self.work_of_art_ent_X[y - 2]),
                                      "spacy12": (self.ordinal_ent_X[y]),
                                      "spacy12-1": (self.ordinal_ent_X[y - 1]),
                                      "spacy12-2": (self.ordinal_ent_X[y - 2]),
                                      "spacy13": (self.cardinal_ent_X[y]),
                                      "spacy13-1": (self.cardinal_ent_X[y - 1]),
                                      "spacy13-2": (self.cardinal_ent_X[y - 2]),
                                      "spacy14": (self.money_ent_X[y]),
                                      "spacy14-1": (self.money_ent_X[y - 1]),
                                      "spacy14-2": (self.money_ent_X[y - 2]),
                                      "spacy15": (self.percent_ent_X[y]),
                                      "spacy15-1": (self.percent_ent_X[y - 1]),
                                      "spacy15-2": (self.percent_ent_X[y - 2]),
                                      "spacy16": (self.product_ent_X[y]),
                                      "spacy16-1": (self.product_ent_X[y - 1]),
                                      "spacy16-2": (self.product_ent_X[y - 2]),
                                      "spacy17": (self.quantity_ent_X[y]),
                                      "spacy17-1": (self.quantity_ent_X[y - 1]),
                                      "spacy17-2": (self.quantity_ent_X[y - 2]),

                                          })
            elif newSpeechLookAheadBy2:
                sentence_features.update({"r-1" : tag_history[y-1],
                                      "r-2 r-1" : "%s %s" % (tag_history[y-2], tag_history[y-1]),
                                      "rel-1" : rel_history[sentence_id-2],
                                      "rel-2 rel-1" : "%s %s" % (rel_history[sentence_id-3], rel_history[sentence_id-2]),
                                      'bias': 1.0,
                                      "r" : tag,
                                      "length" : (self.sent_length[y]),
                                      "length+1" : (self.sent_length[y+1]),
                                      "length-1" : (self.sent_length[y-1]),
                                      "length-2" : (self.sent_length[y-2]),
                                      "tfdif" : (self.tfidf_top20[y]),
                                      "tfdif+1" : (self.tfidf_top20[y+1]),
                                      "tfdif-1" : (self.tfidf_top20[y-1]),
                                      "tfdif-2" : (self.tfidf_top20[y-2]),
                                      "loc1" : (self.loc1_X[y]),
                                      "loc1+1" : (self.loc1_X[y+1]),
                                      "loc1-1" : (self.loc1_X[y-1]),
                                      "loc1-2" : (self.loc1_X[y-2]),
                                      "loc2" : (self.loc2_X[y]),
                                      "loc2+1" : (self.loc2_X[y+1]),
                                      "loc2-1" : (self.loc1_X[y-1]),
                                      "loc2-2" : (self.loc1_X[y-2]),
                                      "loc3" : (self.loc3_X[y]),
                                      "loc3+1" : (self.loc3_X[y+1]),
                                      "loc3-1" : (self.loc1_X[y-1]),
                                      "loc3-2" : (self.loc1_X[y-2]),
                                      "loc4" : (self.loc4_X[y]),
                                      "loc4+1" : (self.loc4_X[y+1]),
                                      "loc4-1" : (self.loc1_X[y-1]),
                                      "loc4-2" : (self.loc1_X[y-2]),
                                      "loc5" : (self.loc5_X[y]),
                                      "loc5+1" : (self.loc5_X[y+1]),
                                      "loc5-1" : (self.loc1_X[y-1]),
                                      "loc5-2" : (self.loc1_X[y-2]),
                                      "loc6" : (self.loc6_X[y]),
                                      "loc6+1" : (self.loc6_X[y+1]),
                                      "loc6-1" : (self.loc1_X[y-1]),
                                      "loc6-2" : (self.loc1_X[y-2]),
                                      "quote1" : (self.inq_X[y]),
                                      "quote1+1" : (self.inq_X[y+1]),
                                      "quote1-1" : (self.inq_X[y-1]),
                                      "quote1-2" : (self.inq_X[y-2]),
                                      "quote2" : (self.qb_X[y]),
                                      "quote2+1" : (self.qb_X[y+1]),
                                      "quote2-1" : (self.qb_X[y-1]),
                                      "quote2-2" : (self.inq_X[y-2]),
                                      "asmo1" : (self.agree_X[y]),
                                      "asmo1+1" : (self.agree_X[y+1]),
                                      "asmo1-1" : (self.agree_X[y-1]),
                                      "asmo1-2" : (self.agree_X[y-2]),
                                      "asmo2" : (self.outcome_X[y]),
                                      "asmo2+1" : (self.outcome_X[y+1]),
                                      "asmo2-1" : (self.outcome_X[y-1]),
                                      "asmo2-2" : (self.outcome_X[y-2]),
                                      "cue1" : (self.modal_dep_bool_X[y]),
                                      "cue1+1" : (self.modal_dep_bool_X[y+1]),
                                      "cue1-2" : (self.modal_dep_bool_X[y-2]),
                                      "cue1-1" : (self.modal_dep_bool_X[y-1]),
                                      "cue2" : (self.modal_dep_count_X[y]),
                                      "cue2+1" : (self.modal_dep_count_X[y+1]),
                                      "cue2-2" : (self.modal_dep_count_X[y-2]),
                                      "cue2-1" : (self.modal_dep_count_X[y-1]),
                                      "cue3" : (self.new_modal_X[y]),
                                      "cue3+1" : (self.new_modal_X[y+1]),
                                      "cue3-1" : (self.new_modal_X[y-1]),
                                      "cue3-2" : (self.new_modal_X[y-2]),
                                      "cue4" : (self.new_tense_X[y]),
                                      "cue4+1" : (self.new_tense_X[y+1]),
                                      "cue4-2" : (self.new_tense_X[y-2]),
                                      "cue4-1" : (self.new_tense_X[y-1]),
                                      "cue5" : (self.new_dep_X[y]),
                                      "cue5+1" : (self.new_dep_X[y+1]),
                                      "cue5-2" : (self.new_dep_X[y-2]),
                                      "cue5-1" : (self.new_dep_X[y-1]),
                                      "cue6" : (self.new_tag_X[y]),
                                      "cue6+1" : (self.new_tag_X[y+1]),
                                      "cue6-2" : (self.new_tag_X[y-2]),
                                      "cue6-1" : (self.new_tag_X[y-1]),
                                      "cue7" : (self.new_negative_X[y]),
                                      "cue7+1" : (self.new_negative_X[y+1]),
                                      "cue7-2" : (self.new_negative_X[y-2]),
                                      "cue7-1" : (self.new_negative_X[y-1]),
                                      "cue8" : (self.new_stop_X[y]),
                                      "cue8+1" : (self.new_stop_X[y+1]),
                                      "cue8-2" : (self.new_stop_X[y-2]),
                                      "cue8-1" : (self.new_stop_X[y-1]),
                                      "cue9" : (self.new_voice_X[y]),
                                      "cue9+1" : (self.new_voice_X[y+1]),
                                      "cue9-2" : (self.new_voice_X[y-2]),
                                      "cue9-1" : (self.new_voice_X[y-1]),
                                      "cue10" : (self.second_pos_X[y]),
                                      "cue10+1" : (self.second_pos_X[y+1]),
                                      "cue10-2" : (self.second_pos_X[y-2]),
                                      "cue10-1" : (self.second_pos_X[y-1]),
                                      "cue11" : (self.second_dep_X[y]),
                                      "cue11+1" : (self.second_dep_X[y+1]),
                                      "cue11-2" : (self.second_dep_X[y-2]),
                                      "cue11-1" : (self.second_dep_X[y-1]),
                                      "cue12" : (self.second_tag_X[y]),
                                      "cue12+1" : (self.second_tag_X[y+1]),
                                      "cue12-2" : (self.second_tag_X[y-2]),
                                      "cue12-1" : (self.second_tag_X[y-1]),
                                      "cue13" : (self.second_stop_X[y]),
                                      "cue13+1" : (self.second_stop_X[y+1]),
                                      "cue13-2" : (self.second_stop_X[y-2]),
                                      "cue13-1" : (self.second_stop_X[y-1]),
                                                                            "spacy1" : (self.loc_ent_X[y]),
                                      "spacy1+1" : (self.loc_ent_X[y+1]),
                                      "spacy1-1" : (self.loc_ent_X[y-1]),
                                      "spacy1-2" : (self.loc_ent_X[y-2]),
                                      "spacy2" : (self.org_ent_X[y]),
                                      "spacy2+1" : (self.org_ent_X[y+1]),
                                      "spacy2-1" : (self.org_ent_X[y-1]),
                                      "spacy2-2" : (self.org_ent_X[y-2]),
                                      "spacy3" : (self.date_ent_X[y]),
                                      "spacy3+1" : (self.date_ent_X[y+1]),
                                      "spacy3-1" : (self.date_ent_X[y-1]),
                                      "spacy3-2" : (self.date_ent_X[y-2]),
                                      "spacy4" : (self.person_ent_X[y]),
                                      "spacy4+1" : (self.person_ent_X[y+1]),
                                      "spacy4-1" : (self.person_ent_X[y-1]),
                                      "spacy4-2" : (self.person_ent_X[y-2]),
                                      "spacy5": (self.fac_ent_X[y]),
                                      "spacy5-1": (self.fac_ent_X[y - 1]),
                                      "spacy5-2": (self.fac_ent_X[y - 2]),
                                      "spacy5+1": (self.fac_ent_X[y + 1]),
                                      "spacy6": (self.norp_ent_X[y]),
                                      "spacy6-1": (self.norp_ent_X[y - 1]),
                                      "spacy6-2": (self.norp_ent_X[y - 2]),
                                      "spacy6+1": (self.norp_ent_X[y + 1]),
                                      "spacy7": (self.gpe_ent_X[y]),
                                      "spacy7-1": (self.gpe_ent_X[y - 1]),
                                      "spacy7-2": (self.gpe_ent_X[y - 2]),
                                      "spacy7+1": (self.gpe_ent_X[y + 1]),
                                      "spacy8": (self.event_ent_X[y]),
                                      "spacy8-1": (self.event_ent_X[y - 1]),
                                      "spacy8-2": (self.event_ent_X[y - 2]),
                                      "spacy8+1": (self.event_ent_X[y + 1]),
                                      "spacy9": (self.law_ent_X[y]),
                                      "spacy9-1": (self.law_ent_X[y - 1]),
                                      "spacy9-2": (self.law_ent_X[y - 2]),
                                      "spacy9+1": (self.law_ent_X[y + 1]),
                                      "spacy10": (self.time_ent_X[y]),
                                      "spacy10-1": (self.time_ent_X[y - 1]),
                                      "spacy10-2": (self.time_ent_X[y - 2]),
                                      "spacy10+1": (self.time_ent_X[y + 1]),
                                      "spacy11": (self.work_of_art_ent_X[y]),
                                      "spacy11-1": (self.work_of_art_ent_X[y - 1]),
                                      "spacy11-2": (self.work_of_art_ent_X[y - 2]),
                                      "spacy11+1": (self.work_of_art_ent_X[y + 1]),
                                      "spacy12": (self.ordinal_ent_X[y]),
                                      "spacy12-1": (self.ordinal_ent_X[y - 1]),
                                      "spacy12-2": (self.ordinal_ent_X[y - 2]),
                                      "spacy12+1": (self.ordinal_ent_X[y + 1]),
                                      "spacy13": (self.cardinal_ent_X[y]),
                                      "spacy13-1": (self.cardinal_ent_X[y - 1]),
                                      "spacy13-2": (self.cardinal_ent_X[y - 2]),
                                      "spacy13+1": (self.cardinal_ent_X[y + 1]),
                                      "spacy14": (self.money_ent_X[y]),
                                      "spacy14-1": (self.money_ent_X[y - 1]),
                                      "spacy14-2": (self.money_ent_X[y - 2]),
                                      "spacy14+1": (self.money_ent_X[y + 1]),
                                      "spacy15": (self.percent_ent_X[y]),
                                      "spacy15-1": (self.percent_ent_X[y - 1]),
                                      "spacy15-2": (self.percent_ent_X[y - 2]),
                                      "spacy15+1": (self.percent_ent_X[y + 1]),
                                      "spacy16": (self.product_ent_X[y]),
                                      "spacy16-1": (self.product_ent_X[y - 1]),
                                      "spacy16-2": (self.product_ent_X[y - 2]),
                                      "spacy16+1": (self.product_ent_X[y + 1]),
                                      "spacy17": (self.quantity_ent_X[y]),
                                      "spacy17-1": (self.quantity_ent_X[y - 1]),
                                      "spacy17-2": (self.quantity_ent_X[y - 2]),
                                      "spacy17+1": (self.quantity_ent_X[y + 1]),

                                          })

            else:
                sentence_features.update({"r-1" : tag_history[y-1],
                                      "r-2 r-1" : "%s %s" % (tag_history[y-2], tag_history[y-1]),
                                      "rel-1" : rel_history[sentence_id-2],
                                      "rel-2 rel-1" : "%s %s" % (rel_history[sentence_id-3], rel_history[sentence_id-2]),
                                      'bias': 1.0,
                                      "r" : tag,
                                      "length" : (self.sent_length[y]),
                                      "length+1" : (self.sent_length[y+1]),
                                      "length+2" : (self.sent_length[y+2]),
                                      "length-1" : (self.sent_length[y-1]),
                                      "length-2" : (self.sent_length[y-2]),
                                      "tfdif" : (self.tfidf_top20[y]),
                                      "tfdif+1" : (self.tfidf_top20[y+1]),
                                      "tfdif+2" : (self.tfidf_top20[y+2]),
                                      "tfdif-1" : (self.tfidf_top20[y-1]),
                                      "tfdif-2" : (self.tfidf_top20[y-2]),
                                      "loc1" : (self.loc1_X[y]),
                                      "loc1+1" : (self.loc1_X[y+1]),
                                      "loc1+2" : (self.loc1_X[y+2]),
                                      "loc1-1" : (self.loc1_X[y-1]),
                                      "loc1-2" : (self.loc1_X[y-2]),
                                      "loc2" : (self.loc2_X[y]),
                                      "loc2+1" : (self.loc2_X[y+1]),
                                      "loc2+2" : (self.loc2_X[y+2]),
                                      "loc2-1" : (self.loc1_X[y-1]),
                                      "loc2-2" : (self.loc1_X[y-2]),
                                      "loc3" : (self.loc3_X[y]),
                                      "loc3+1" : (self.loc3_X[y+1]),
                                      "loc3+2" : (self.loc3_X[y+2]),
                                      "loc3-1" : (self.loc1_X[y-1]),
                                      "loc3-2" : (self.loc1_X[y-2]),
                                      "loc4" : (self.loc4_X[y]),
                                      "loc4+1" : (self.loc4_X[y+1]),
                                      "loc4+2" : (self.loc4_X[y+2]),
                                      "loc4-1" : (self.loc1_X[y-1]),
                                      "loc4-2" : (self.loc1_X[y-2]),
                                      "loc5" : (self.loc5_X[y]),
                                      "loc5+1" : (self.loc5_X[y+1]),
                                      "loc5+2" : (self.loc5_X[y+2]),
                                      "loc5-1" : (self.loc1_X[y-1]),
                                      "loc5-2" : (self.loc1_X[y-2]),
                                      "loc6" : (self.loc6_X[y]),
                                      "loc6+1" : (self.loc6_X[y+1]),
                                      "loc6+2" : (self.loc6_X[y+2]),
                                      "loc6-1" : (self.loc1_X[y-1]),
                                      "loc6-2" : (self.loc1_X[y-2]),
                                      "quote1" : (self.inq_X[y]),
                                      "quote1+1" : (self.inq_X[y+1]),
                                      "quote1+2" : (self.inq_X[y+2]),
                                      "quote1-1" : (self.inq_X[y-1]),
                                      "quote1-2" : (self.inq_X[y-2]),
                                      "quote2" : (self.qb_X[y]),
                                      "quote2+1" : (self.qb_X[y+1]),
                                      "quote2+2" : (self.qb_X[y+2]),
                                      "quote2-1" : (self.qb_X[y-1]),
                                      "quote2-2" : (self.inq_X[y-2]),
                                      "asmo1" : (self.agree_X[y]),
                                      "asmo1+1" : (self.agree_X[y+1]),
                                      "asmo1+2" : (self.agree_X[y+2]),
                                      "asmo1-1" : (self.agree_X[y-1]),
                                      "asmo1-2" : (self.agree_X[y-2]),
                                      "asmo2" : (self.outcome_X[y]),
                                      "asmo2+1" : (self.outcome_X[y+1]),
                                      "asmo2+2" : (self.outcome_X[y+2]),
                                      "asmo2-1" : (self.outcome_X[y-1]),
                                      "asmo2-2" : (self.outcome_X[y-2]),
                                      "cue1" : (self.modal_dep_bool_X[y]),
                                      "cue1+1" : (self.modal_dep_bool_X[y+1]),
                                      "cue1+2" : (self.modal_dep_bool_X[y+2]),
                                      "cue1-2" : (self.modal_dep_bool_X[y-2]),
                                      "cue1-1" : (self.modal_dep_bool_X[y-1]),
                                      "cue2" : (self.modal_dep_count_X[y]),
                                      "cue2+1" : (self.modal_dep_count_X[y+1]),
                                      "cue2+2" : (self.modal_dep_count_X[y+2]),
                                      "cue2-2" : (self.modal_dep_count_X[y-2]),
                                      "cue2-1" : (self.modal_dep_count_X[y-1]),
                                      "cue3" : (self.new_modal_X[y]),
                                      "cue3+1" : (self.new_modal_X[y+1]),
                                      "cue3+2" : (self.new_modal_X[y+2]),
                                      "cue3-1" : (self.new_modal_X[y-1]),
                                      "cue3-2" : (self.new_modal_X[y-2]),
                                      "cue4" : (self.new_tense_X[y]),
                                      "cue4+1" : (self.new_tense_X[y+1]),
                                      "cue4+2" : (self.new_tense_X[y+2]),
                                      "cue4-2" : (self.new_tense_X[y-2]),
                                      "cue4-1" : (self.new_tense_X[y-1]),
                                      "cue5" : (self.new_dep_X[y]),
                                      "cue5+1" : (self.new_dep_X[y+1]),
                                      "cue5+2" : (self.new_dep_X[y+2]),
                                      "cue5-2" : (self.new_dep_X[y-2]),
                                      "cue5-1" : (self.new_dep_X[y-1]),
                                      "cue6" : (self.new_tag_X[y]),
                                      "cue6+1" : (self.new_tag_X[y+1]),
                                      "cue6+2" : (self.new_tag_X[y+2]),
                                      "cue6-2" : (self.new_tag_X[y-2]),
                                      "cue6-1" : (self.new_tag_X[y-1]),
                                      "cue7" : (self.new_negative_X[y]),
                                      "cue7+1" : (self.new_negative_X[y+1]),
                                      "cue7+2" : (self.new_negative_X[y+2]),
                                      "cue7-2" : (self.new_negative_X[y-2]),
                                      "cue7-1" : (self.new_negative_X[y-1]),
                                      "cue8" : (self.new_stop_X[y]),
                                      "cue8+1" : (self.new_stop_X[y+1]),
                                      "cue8+2" : (self.new_stop_X[y+2]),
                                      "cue8-2" : (self.new_stop_X[y-2]),
                                      "cue8-1" : (self.new_stop_X[y-1]),
                                      "cue9" : (self.new_voice_X[y]),
                                      "cue9+1" : (self.new_voice_X[y+1]),
                                      "cue9+2" : (self.new_voice_X[y+2]),
                                      "cue9-2" : (self.new_voice_X[y-2]),
                                      "cue9-1" : (self.new_voice_X[y-1]),
                                      "cue10" : (self.second_pos_X[y]),
                                      "cue10+1" : (self.second_pos_X[y+1]),
                                      "cue10+2" : (self.second_pos_X[y+2]),
                                      "cue10-2" : (self.second_pos_X[y-2]),
                                      "cue10-1" : (self.second_pos_X[y-1]),
                                      "cue11" : (self.second_dep_X[y]),
                                      "cue11+1" : (self.second_dep_X[y+1]),
                                      "cue11+2" : (self.second_dep_X[y+2]),
                                      "cue11-2" : (self.second_dep_X[y-2]),
                                      "cue11-1" : (self.second_dep_X[y-1]),
                                      "cue12" : (self.second_tag_X[y]),
                                      "cue12+1" : (self.second_tag_X[y+1]),
                                      "cue12+2" : (self.second_tag_X[y+2]),
                                      "cue12-2" : (self.second_tag_X[y-2]),
                                      "cue12-1" : (self.second_tag_X[y-1]),
                                      "cue13" : (self.second_stop_X[y]),
                                      "cue13+1" : (self.second_stop_X[y+1]),
                                      "cue13+2" : (self.second_stop_X[y+2]),
                                      "cue13-2" : (self.second_stop_X[y-2]),
                                      "cue13-1" : (self.second_stop_X[y-1]),

                                      "spacy1" : (self.loc_ent_X[y]),
                                      "spacy1+1" : (self.loc_ent_X[y+1]),
                                      "spacy1+2" : (self.loc_ent_X[y+2]),
                                      "spacy1-1" : (self.loc_ent_X[y-1]),
                                      "spacy1-2" : (self.loc_ent_X[y-2]),
                                      "spacy2" : (self.org_ent_X[y]),
                                      "spacy2+1" : (self.org_ent_X[y+1]),
                                      "spacy2+2" : (self.org_ent_X[y+2]),
                                      "spacy2-1" : (self.org_ent_X[y-1]),
                                      "spacy2-2" : (self.org_ent_X[y-2]),
                                      "spacy3" : (self.date_ent_X[y]),
                                      "spacy3+1" : (self.date_ent_X[y+1]),
                                      "spacy3+2" : (self.date_ent_X[y+2]),
                                      "spacy3-1" : (self.date_ent_X[y-1]),
                                      "spacy3-2" : (self.date_ent_X[y-2]),
                                      "spacy4" : (self.person_ent_X[y]),
                                      "spacy4+1" : (self.person_ent_X[y+1]),
                                      "spacy4+2" : (self.person_ent_X[y+2]),
                                      "spacy4-1" : (self.person_ent_X[y-1]),
                                      "spacy4-2" : (self.person_ent_X[y-2]),
                                      "spacy5": (self.fac_ent_X[y]),
                                      "spacy5+1": (self.fac_ent_X[y + 1]),
                                      "spacy5+2": (self.fac_ent_X[y + 2]),
                                      "spacy5-1": (self.fac_ent_X[y - 1]),
                                      "spacy5-2": (self.fac_ent_X[y - 2]),
                                      "spacy6": (self.norp_ent_X[y]),
                                      "spacy6+1": (self.norp_ent_X[y + 1]),
                                      "spacy6+2": (self.norp_ent_X[y + 2]),
                                      "spacy6-1": (self.norp_ent_X[y - 1]),
                                      "spacy6-2": (self.norp_ent_X[y - 2]),
                                      "spacy7": (self.gpe_ent_X[y]),
                                      "spacy7+1": (self.gpe_ent_X[y + 1]),
                                      "spacy7+2": (self.gpe_ent_X[y + 2]),
                                      "spacy7-1": (self.gpe_ent_X[y - 1]),
                                      "spacy7-2": (self.gpe_ent_X[y - 2]),
                                      "spacy8": (self.event_ent_X[y]),
                                      "spacy8+1": (self.event_ent_X[y + 1]),
                                      "spacy8+2": (self.event_ent_X[y + 2]),
                                      "spacy8-1": (self.event_ent_X[y - 1]),
                                      "spacy8-2": (self.event_ent_X[y - 2]),
                                      "spacy9": (self.law_ent_X[y]),
                                      "spacy9+1": (self.law_ent_X[y + 1]),
                                      "spacy9+2": (self.law_ent_X[y + 2]),
                                      "spacy9-1": (self.law_ent_X[y - 1]),
                                      "spacy9-2": (self.law_ent_X[y - 2]),
                                      "spacy10": (self.time_ent_X[y]),
                                      "spacy10+1": (self.time_ent_X[y + 1]),
                                      "spacy10+2": (self.time_ent_X[y + 2]),
                                      "spacy10-1": (self.time_ent_X[y - 1]),
                                      "spacy10-2": (self.time_ent_X[y - 2]),
                                      "spacy11": (self.work_of_art_ent_X[y]),
                                      "spacy11+1": (self.work_of_art_ent_X[y + 1]),
                                      "spacy11+2": (self.work_of_art_ent_X[y + 2]),
                                      "spacy11-1": (self.work_of_art_ent_X[y - 1]),
                                      "spacy11-2": (self.work_of_art_ent_X[y - 2]),
                                      "spacy12": (self.ordinal_ent_X[y]),
                                      "spacy12+1": (self.ordinal_ent_X[y + 1]),
                                      "spacy12+2": (self.ordinal_ent_X[y + 2]),
                                      "spacy12-1": (self.ordinal_ent_X[y - 1]),
                                      "spacy12-2": (self.ordinal_ent_X[y - 2]),
                                      "spacy13": (self.cardinal_ent_X[y]),
                                      "spacy13+1": (self.cardinal_ent_X[y + 1]),
                                      "spacy13+2": (self.cardinal_ent_X[y + 2]),
                                      "spacy13-1": (self.cardinal_ent_X[y - 1]),
                                      "spacy13-2": (self.cardinal_ent_X[y - 2]),
                                      "spacy14": (self.money_ent_X[y]),
                                      "spacy14+1": (self.money_ent_X[y + 1]),
                                      "spacy14+2": (self.money_ent_X[y + 2]),
                                      "spacy14-1": (self.money_ent_X[y - 1]),
                                      "spacy14-2": (self.money_ent_X[y - 2]),
                                      "spacy15": (self.percent_ent_X[y]),
                                      "spacy15+1": (self.percent_ent_X[y + 1]),
                                      "spacy15+2": (self.percent_ent_X[y + 2]),
                                      "spacy15-1": (self.percent_ent_X[y - 1]),
                                      "spacy15-2": (self.percent_ent_X[y - 2]),
                                      "spacy16": (self.product_ent_X[y]),
                                      "spacy16+1": (self.product_ent_X[y + 1]),
                                      "spacy16+2": (self.product_ent_X[y + 2]),
                                      "spacy16-1": (self.product_ent_X[y - 1]),
                                      "spacy16-2": (self.product_ent_X[y - 2]),
                                      "spacy17": (self.quantity_ent_X[y]),
                                      "spacy17+1": (self.quantity_ent_X[y + 1]),
                                      "spacy17+2": (self.quantity_ent_X[y + 2]),
                                      "spacy17-1": (self.quantity_ent_X[y - 1]),
                                      "spacy17-2": (self.quantity_ent_X[y - 2]),

                                          })

            return sentence_features

        # here it assigns relevant training data to the np.array, then can use the index to pull out the right rows
    def pull_training_data(self):
        # open up the MLdata
        # create the sentences arrays at the same time? including relevant
        # role is the issue - create an object (dictionary) that has all the
        # relevant features - as above, plus s-1, and s-2 ...
        with open('./data/MLdata_train.csv', 'r') as infile:
            reader = csv.DictReader(infile)

        # for each row in the MLDATA cv file, get the corresponding result - add to array
            for row in reader:
                self.rel_y = np.append(self.rel_y, [float(row['align'])])
                self.agree_X = np.append(self.agree_X, [float(row['agree'])])
                self.outcome_X = np.append(self.outcome_X, [float(row['outcome'])])
                self.loc1_X = np.append(self.loc1_X, [float(row['loc1'])])
                self.loc2_X = np.append(self.loc2_X, [float(row['loc2'])])
                self.loc3_X = np.append(self.loc3_X, [float(row['loc3'])])
                self.loc4_X = np.append(self.loc4_X, [float(row['loc4'])])
                self.loc5_X = np.append(self.loc5_X, [float(row['loc5'])])
                self.loc6_X = np.append(self.loc6_X, [float(row['loc6'])])
                self.sentlen_X = np.append(self.sentlen_X, [float(row['sentlen'])])
                self.qb_X = np.append(self.qb_X, [float(row['quoteblock'])])
                self.inq_X = np.append(self.inq_X, [float(row['inline_q'])])
                self.rhet_X = np.append(self.rhet_X, [float(row['rhet'])])
                self.wordlist_X = np.append(self.wordlist_X, [float(row['wordlist'])])

                self.tfidf_top20_X = np.append(self.tfidf_top20_X, [float(row['tfidf_top20'])])
                self.asp_X = np.append(self.asp_X, [float(row['aspect'])])
                self.modal_X = np.append(self.modal_X, [float(row['modal'])])
                self.voice_X = np.append(self.voice_X, [float(row['voice'])])
                self.negcue_X = np.append(self.negcue_X, [float(row['negation'])])
                self.tense_X = np.append(self.tense_X, [float(row['tense'])])

                self.loc_ent_X = np.append(self.loc_ent_X, [float(row['loc ent'])])
                self.org_ent_X = np.append(self.org_ent_X, [float(row['org ent'])])
                self.date_ent_X = np.append(self.date_ent_X, [float(row['date ent'])])
                self.person_ent_X = np.append(self.person_ent_X, [float(row['person ent'])])
                self.fac_ent_X = np.append(self.fac_ent_X, [float(row['fac_ent'])])
                self.norp_ent_X = np.append(self.norp_ent_X, [float(row['norp_ent'])])
                self.gpe_ent_X = np.append(self.gpe_ent_X, [float(row['gpe_ent'])])
                self.event_ent_X = np.append(self.event_ent_X, [float(row['event_ent'])])
                self.law_ent_X = np.append(self.law_ent_X, [float(row['law_ent'])])
                self.time_ent_X = np.append(self.time_ent_X, [float(row['time_ent'])])
                self.work_of_art_ent_X = np.append(self.work_of_art_ent_X, [float(row['work_of_art_ent'])])
                self.ordinal_ent_X = np.append(self.ordinal_ent_X, [float(row['ordinal_ent'])])
                self.cardinal_ent_X = np.append(self.cardinal_ent_X, [float(row['cardinal_ent'])])
                self.money_ent_X = np.append(self.money_ent_X, [float(row['money_ent'])])
                self.percent_ent_X = np.append(self.percent_ent_X, [float(row['percent_ent'])])
                self.product_ent_X = np.append(self.product_ent_X, [float(row['product_ent'])])
                self.quantity_ent_X = np.append(self.quantity_ent_X, [float(row['quantity_ent'])])

                self.judgename.append(row['judgename'])
                self.rhetlabel.append(row['rhet label'])
                self.modal_dep_bool_X = np.append(self.modal_dep_bool_X, [float(row['cp dep bool'])])
                self.modal_dep_count_X = np.append(self.modal_dep_count_X, [float(row['cp dep count'])])

                self.new_modal_X = np.append(self.new_modal_X, [float(row['cp modal'])])
                self.new_tense_X = np.append(self.new_tense_X, [float(row['cp tense'])])
                self.new_dep_X = np.append(self.new_dep_X, [float(row['cp dep'])])
                self.new_tag_X = np.append(self.new_tag_X, [float(row['cp tag'])])
                self.new_negative_X = np.append(self.new_negative_X, [float(row['cp negative'])])
                self.new_stop_X = np.append(self.new_stop_X, [float(row['cp stop'])])
                self.new_voice_X = np.append(self.new_voice_X, [float(row['cp voice'])])

                self.second_pos_X = np.append(self.second_pos_X, [float(row['cp second pos'])])
                self.second_dep_X = np.append(self.second_dep_X, [float(row['cp second dep'])])
                self.second_tag_X = np.append(self.second_tag_X, [float(row['cp second tag'])])
                self.second_stop_X = np.append(self.second_stop_X, [float(row['cp second stop'])])

        self.location = self.loc1_X, self.loc2_X, self.loc3_X, self.loc4_X, self.loc5_X, self.loc6_X
        self.quote = self.inq_X, self.qb_X
        self.asmo = self.agree_X, self.outcome_X
    #    self.cue_phrase = self.asp_X, self.modal_X, self.voice_X, self.negcue_X, self.tense_X
        self.sent_length =  self.sentlen_X
        self.tfidf_top20 = self.tfidf_top20_X
        self.spacy = self.loc_ent_X, self.org_ent_X, self.date_ent_X, self.person_ent_X, self.fac_ent_X, self.norp_ent_X, \
                     self.gpe_ent_X, self.event_ent_X, self.law_ent_X, self.time_ent_X, self.work_of_art_ent_X, self.ordinal_ent_X, \
                     self.cardinal_ent_X, self.money_ent_X, self.percent_ent_X, self.product_ent_X, self.quantity_ent_X
        self.cue_phrase = self.modal_dep_bool_X, self.modal_dep_count_X, self.new_modal_X, self.new_tense_X, self.new_dep_X, self.new_tag_X, self.new_negative_X, self.new_stop_X, self.new_voice_X, self.second_pos_X, self.second_dep_X, self.second_tag_X, self.second_stop_X

    def pull_testing_data(self):
        # open up the MLdata
        # create the sentences arrays at the same time? including relevant
        # role is the issue - create an object (dictionary) that has all the
        # relevant features - as above, plus s-1, and s-2 ...
        with open('./data/MLdata_test.csv', 'r') as infile:
            reader = csv.DictReader(infile)

        # for each row in the MLDATA cv file, get the corresponding result - add to array
            for row in reader:
                self.rel_y = np.append(self.rel_y, [float(row['align'])])
                self.agree_X = np.append(self.agree_X, [float(row['agree'])])
                self.outcome_X = np.append(self.outcome_X, [float(row['outcome'])])
                self.loc1_X = np.append(self.loc1_X, [float(row['loc1'])])
                self.loc2_X = np.append(self.loc2_X, [float(row['loc2'])])
                self.loc3_X = np.append(self.loc3_X, [float(row['loc3'])])
                self.loc4_X = np.append(self.loc4_X, [float(row['loc4'])])
                self.loc5_X = np.append(self.loc5_X, [float(row['loc5'])])
                self.loc6_X = np.append(self.loc6_X, [float(row['loc6'])])
                self.sentlen_X = np.append(self.sentlen_X, [float(row['sentlen'])])
                self.qb_X = np.append(self.qb_X, [float(row['quoteblock'])])
                self.inq_X = np.append(self.inq_X, [float(row['inline_q'])])
                self.wordlist_X = np.append(self.wordlist_X, [float(row['wordlist'])])
                self.rhet_X = np.append(self.rhet_X, [float(row['rhet'])])

                self.tfidf_top20_X = np.append(self.tfidf_top20_X, [float(row['tfidf_top20'])])
                self.asp_X = np.append(self.asp_X, [float(row['aspect'])])
                self.modal_X = np.append(self.modal_X, [float(row['modal'])])
                self.voice_X = np.append(self.voice_X, [float(row['voice'])])
                self.negcue_X = np.append(self.negcue_X, [float(row['negation'])])
                self.tense_X = np.append(self.tense_X, [float(row['tense'])])

                self.loc_ent_X = np.append(self.loc_ent_X, [float(row['loc ent'])])
                self.org_ent_X = np.append(self.org_ent_X, [float(row['org ent'])])
                self.date_ent_X = np.append(self.date_ent_X, [float(row['date ent'])])
                self.person_ent_X = np.append(self.person_ent_X, [float(row['person ent'])])
                self.fac_ent_X = np.append(self.fac_ent_X, [float(row['fac_ent'])])
                self.norp_ent_X = np.append(self.norp_ent_X, [float(row['norp_ent'])])
                self.gpe_ent_X = np.append(self.gpe_ent_X, [float(row['gpe_ent'])])
                self.event_ent_X = np.append(self.event_ent_X, [float(row['event_ent'])])
                self.law_ent_X = np.append(self.law_ent_X, [float(row['law_ent'])])
                self.time_ent_X = np.append(self.time_ent_X, [float(row['time_ent'])])
                self.work_of_art_ent_X = np.append(self.work_of_art_ent_X, [float(row['work_of_art_ent'])])
                self.ordinal_ent_X = np.append(self.ordinal_ent_X, [float(row['ordinal_ent'])])
                self.cardinal_ent_X = np.append(self.cardinal_ent_X, [float(row['cardinal_ent'])])
                self.money_ent_X = np.append(self.money_ent_X, [float(row['money_ent'])])
                self.percent_ent_X = np.append(self.percent_ent_X, [float(row['percent_ent'])])
                self.product_ent_X = np.append(self.product_ent_X, [float(row['product_ent'])])
                self.quantity_ent_X = np.append(self.quantity_ent_X, [float(row['quantity_ent'])])

                self.judgename.append(row['judgename'])
                self.rhetlabel.append(row['rhet label'])

                self.modal_dep_bool_X = np.append(self.modal_dep_bool_X, [float(row['cp dep bool'])])
                self.modal_dep_count_X = np.append(self.modal_dep_count_X, [float(row['cp dep count'])])

                self.new_modal_X = np.append(self.new_modal_X, [float(row['cp modal'])])
                self.new_tense_X = np.append(self.new_tense_X, [float(row['cp tense'])])
                self.new_dep_X = np.append(self.new_dep_X, [float(row['cp dep'])])
                self.new_tag_X = np.append(self.new_tag_X, [float(row['cp tag'])])
                self.new_negative_X = np.append(self.new_negative_X, [float(row['cp negative'])])
                self.new_stop_X = np.append(self.new_stop_X, [float(row['cp stop'])])
                self.new_voice_X = np.append(self.new_voice_X, [float(row['cp voice'])])

                self.second_pos_X = np.append(self.second_pos_X, [float(row['cp second pos'])])
                self.second_dep_X = np.append(self.second_dep_X, [float(row['cp second dep'])])
                self.second_tag_X = np.append(self.second_tag_X, [float(row['cp second tag'])])
                self.second_stop_X = np.append(self.second_stop_X, [float(row['cp second stop'])])

        self.location = self.loc1_X, self.loc2_X, self.loc3_X, self.loc4_X, self.loc5_X, self.loc6_X
        self.quote = self.inq_X, self.qb_X
        self.asmo = self.agree_X, self.outcome_X
      #  self.cue_phrase = self.asp_X, self.modal_X, self.voice_X, self.negcue_X, self.tense_X
        self.sent_length =  self.sentlen_X
        self.tfidf_top20 = self.tfidf_top20_X
        self.spacy = self.loc_ent_X, self.org_ent_X, self.date_ent_X, self.person_ent_X, self.fac_ent_X, self.norp_ent_X, \
                     self.gpe_ent_X, self.event_ent_X, self.law_ent_X, self.time_ent_X, self.work_of_art_ent_X, self.ordinal_ent_X, \
                     self.cardinal_ent_X, self.money_ent_X, self.percent_ent_X, self.product_ent_X, self.quantity_ent_X
        self.cue_phrase = self.modal_dep_bool_X, self.modal_dep_count_X, self.new_modal_X, self.new_tense_X, self.new_dep_X, self.new_tag_X, self.new_negative_X, self.new_stop_X, self.new_voice_X, self.second_pos_X, self.second_dep_X, self.second_tag_X, self.second_stop_X


crf = crf()
crf.train_crf()
