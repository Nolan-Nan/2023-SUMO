
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recreation of hachey and grover's (2005, 2006) sequence modelling experiment,
based on Ratnaparkhi (1996, 1998) and Curran and Clark (2003)'s approaches.
using MXPOST implementation in java / converted to python, adjusted for including
speeches rather than sentences and sentences rather than words (for labels)

using the LR classifier from SKlearn. seq_init.py() initializes for the code
here.

NB - sentences in format:
    sentence = {
        'sentence_id' : sentence_id,
        'role' : role,
        'case' : case,
        'judge' : judgename
       }

speeches in format:
    speech = {
            'case' : row['case_id'],
            'judge' : judgename,
            'index' : speechindex
      }

@author: amyconroy
"""

import nltk
from nltk.classify import MaxentClassifier, accuracy
import sklearn
import numpy as np
import csv
import pickle

class seq():
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

        # black stone entities feature-set
        self.judge_blackstone = np.array([])
        self.blackstone = np.array([])
        self.provision_blackstone = np.array([])
        self.instrument_blackstone = np.array([])
        self.court_blackstone = np.array([])
        self.case_blackstone = np.array([])
        self.citation_blackstone = np.array([])

        # spacy entities
        self.loc_ent_X = np.array([])
        self.org_ent_X = np.array([])
        self.date_ent_X = np.array([])
        self.person_ent_X = np.array([])
        self.time_ent_X = np.array([])
        self.gpe_ent_X = np.array([])
        self.fac_ent_X = np.array([])
        self.ordinal_ent_X = np.array([])
        self.spacy = np.array([])
        self.total_spacy_X  = np.array([])

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
        self.blackstone = np.array([])
        self.spacy = np.array([])

        # other data
        self.judgename = []
        self.rhetlabel = []


    def train_seq(self, rare_feat_cutoff=5, trace=3):

        from seq_init import create_tagged_sentences_list
        sentences_list = []


        # TRAIN THEN SAVE
# =============================================================================
#         self.initialize()
#         self.pull_training_data()
#
#         training_list = self.create_speeches_features_list()
#         classifier = MaxentClassifier.train(training_list)
#         f = open('seq_maxent_classifier_smallfeatureset.pickle', 'wb')
#         pickle.dump(classifier, f)
#         f.close()
# =============================================================================


        # OPEN AND TEST

        f = open("seq_maxent_classifier.pickle", "rb")
        classifier = pickle.load(f)
        f.close()

        self.initialize() # to redo for testing data
        self.pull_testing_data()
        testing_list = self.create_speeches_features_list()
        print (nltk.classify.accuracy(classifier, testing_list))

        labels = self.sent_to_rhetlabel()

        classifier.show_most_informative_features(10)

        import collections
        predictions = collections.defaultdict(set)
        gold = collections.defaultdict(set)

        features = testing_list
        i = 0

        features_list = [feat[0] for feat in features]
     #   print(features_list)


        for feature in enumerate(features_list):
            feat = feature[1]
            observed = classifier.classify(feat)
            predictions[observed].add(i)
            print(predictions)
            label = labels[i]
            gold[label].add(i)
            print(gold)
            i += 1
            print(i)
            print('\n')

        i = 0
        from nltk.metrics.scores import precision, recall, f_measure
        for label in predictions:
            print(label, 'Precision:', precision(gold[label], predictions[label]))
            print(label, 'Recall:', recall(gold[label], predictions[label]))
            print(label, 'F1-Score:', f_measure(gold[label], predictions[label]))
            print()


    def sent_to_rhetlabel(self):
        labels = self.rhetlabel
        print("test")
        print(labels)
        rhet_labels = []

        for label in labels:
            rhet_labels.append(label)

        print(rhet_labels)
        return rhet_labels


    # function to get training data for each sentence
    # this is where we iterate through each speech, get the features and then
    # add them to the list
    # as per Ratnaparkhi - this is the tagger search procedure
    def create_speeches_features_list(self):
        featureset = []


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

        print(judges)

        for judge in judges:
             newSpeechLookAheadBy1 = False
             newSpeechLookAheadBy2 = False
             tag = tags[y]

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
                 tag_history = [] # previously assigned tags for that speech
                 featureset.append((self.get_features(tagcount, y, tag_history, newspeech, newSpeechLookAheadBy1, newSpeechLookAheadBy2), tag))
                 tag_history.append(tag)
                 print(tag_history)
                 y += 1
                 tagcount += 1
             else:
                 newspeech = False
                 featureset.append((self.get_features(tagcount, y, tag_history, newspeech,
                                                      newSpeechLookAheadBy1, newSpeechLookAheadBy2), tag))
                 tag_history.append(tag)
                 print(tag_history)
                 y += 1
                 tagcount += 1
             previous_judgename = judge

        return featureset




    # get the features of the current sentence
    # this is where i add in the other features of the sentence ie location, ASMO, etc
    # changing it so that rather than training on the sentence itself, training on its features
    # actually using self info rather than
    # speechCounter = the number of sentences in the speech
    def get_features(self, sentence_id, y, tag_history, newspeech, newSpeechLookAheadBy1, newSpeechLookAheadBy2):
            sentence_features = {} # creates a dict
        # rhetorical tags are strings, all others are int (null if the start of the sentence)
            print(type(sentence_id))
            sentence_id = (int(sentence_id))
            print(type(sentence_id))
            print(sentence_id)
            print(y)
            print(self.sent_length.size)

  # NB - need not a super long way to access the 2D array
        # s refers to the sentence, r refers to rhetorical role
        # ensure that we don't go out of bounds
        # this is not going to safeguard against going past the end of a speech
     #   if self.sent_length[y+1] != None and self.sent_length[y+2] != None:
            if newspeech: # first sentence of a speech, sentence 0 reserved for a new case start
                sentence_features.update({"r-1" : "<START>",
                                      "r-2 r-1" : "<START> <START>", # previous label and current features
                                      "length" : (self.sent_length[y]),
                                      "length+1" : (self.sent_length[y+1]),
                                      "length+2" : (self.sent_length[y+2]),
                                      "length-1" : (None),
                                      "length-2" : (None),
# =============================================================================
#                                       "tfdif" : (self.tfidf_top20[y]),
#                                       "tfdif+1" : (self.tfidf_top20[y+1]),
#                                       "tfdif+2" : (self.tfidf_top20[y+2]),
#                                       "tfdif-1" : (None),
#                                       "tfdif-2" : (None),
# =============================================================================
                                      "loc1" : (self.loc1_X[y]),
                                      "loc1+1" : (self.loc1_X[y+1]),
                                      "loc1+2" : (self.loc1_X[y+2]),
                                      "loc1-1" : (None),
                                      "loc1-2" : (None),
                                      "loc2" : (self.loc2_X[y]),
                                      "loc2+1" : (self.loc2_X[y+1]),
                                      "loc2+2" : (self.loc2_X[y+2]),
                                      "loc2-1" : (None),
                                      "loc2-2" : (None),
                                      "loc3" : (self.loc3_X[y]),
                                      "loc3+1" : (self.loc3_X[y+1]),
                                      "loc3+2" : (self.loc3_X[y+2]),
                                      "loc3-1" : (None),
                                      "loc3-2" : (None),
                                      "loc4" : (self.loc4_X[y]),
                                      "loc4+1" : (self.loc4_X[y+1]),
                                      "loc4+2" : (self.loc4_X[y+2]),
                                      "loc4-1" : (None),
                                      "loc4-2" : (None),
                                      "loc5" : (self.loc5_X[y]),
                                      "loc5+1" : (self.loc5_X[y+1]),
                                      "loc5+2" : (self.loc5_X[y+2]),
                                      "loc5-1" : (None),
                                      "loc5-2" : (None),
                                      "loc6" : (self.loc6_X[y]),
                                      "loc6+1" : (self.loc6_X[y+1]),
                                      "loc6+2" : (self.loc6_X[y+2]),
                                      "loc6-1" : (None),
                                      "loc6-2" : (None),
                                      "quote1" : (self.inq_X[y]),
                                      "quote1+1" : (self.inq_X[y+1]),
                                      "quote1+2" : (self.inq_X[y+2]),
                                      "quote1-1" : (None),
                                      "quote1-2" : (None),
                                      "quote2" : (self.qb_X[y]),
                                      "quote2+1" : (self.qb_X[y+1]),
                                      "quote2+2" : (self.qb_X[y+2]),
                                      "quote2-1" : (None),
                                      "quote2-2" : (None),
                                      "asmo1" : (self.agree_X[y]),
                                      "asmo1+1" : (self.agree_X[y+1]),
                                      "asmo1+2" : (self.agree_X[y+2]),
                                      "asmo1-1" : (None),
                                      "asmo1-2" : (None),
                                      "asmo2" : (self.outcome_X[y]),
                                      "asmo2+1" : (self.outcome_X[y+1]),
                                      "asmo2+2" : (self.outcome_X[y+2]),
                                      "asmo2-1" : (None),
                                      "asmo2-2" : (None),
# =============================================================================
#                                       "cue1" : (self.asp_X[y]),
#                                       "cue1+1" : (self.asp_X[y+1]),
#                                       "cue1+2" : (self.asp_X[y+2]),
#                                       "cue1-1" : (None),
#                                       "cue1-2" : (None),
#                                       "cue2" : (self.modal_X[y]),
#                                       "cue2+1" : (self.modal_X[y+1]),
#                                       "cue2+2" : (self.modal_X[y+2]),
#                                       "cue2-1" : (None),
#                                       "cue2-2" : (None),
#                                       "cue3" : (self.voice_X[y]),
#                                       "cue3+1" : (self.voice_X[y+1]),
#                                       "cue3+2" : (self.voice_X[y+2]),
#                                       "cue3-1" : (None),
#                                       "cue3-2" : (None),
#                                       "cue4" : (self.negcue_X[y]),
#                                       "cue4+1" : (self.negcue_X[y+1]),
#                                       "cue4+2" : (self.negcue_X[y+2]),
#                                       "cue4-1" : (None),
#                                       "cue4-2" : (None),
#                                       "cue5" : (self.tense_X[y]),
#                                       "cue5+1" : (self.tense_X[y+1]),
#                                       "cue5+2" : (self.tense_X[y+2]),
#                                       "cue5-1" : (None),
#                                       "cue5-2" : (None),
# =============================================================================
                                      "bl1" : (self.provision_blackstone[y]),
                                      "bl1+1" : (self.provision_blackstone[y+1]),
                                      "bl1+2" : (self.provision_blackstone[y+2]),
                                      "bl1-1" : (None),
                                      "bl1-2" : (None),
                                      "bl2" : (self.instrument_blackstone[y]),
                                      "bl2+1" : (self.instrument_blackstone[y+1]),
                                      "bl2+2" : (self.instrument_blackstone[y+2]),
                                      "bl2-1" : (None),
                                      "bl2-2" : (None),
                                      "bl3" : (self.court_blackstone [y]),
                                      "bl3+1" : (self.court_blackstone [y+1]),
                                      "bl3+2" : (self.court_blackstone [y+2]),
                                      "bl3-1" : (None),
                                      "bl3-2" : (None),
                                      "bl4" : (self.case_blackstone[y]),
                                      "bl4+1" : (self.case_blackstone[y+1]),
                                      "bl4+2" : (self.case_blackstone[y+2]),
                                      "bl4-1" : (None),
                                      "bl4-2" : (None),
                                      "bl5" : (self.citation_blackstone[y]),
                                      "bl5+1" : (self.citation_blackstone[y+1]),
                                      "bl5+2" : (self.citation_blackstone[y+2]),
                                      "bl5-1" : (None),
                                      "bl5-2" : (None),
                                      "bl6" : (self.judge_blackstone[y]),
                                      "bl6+1" : (self.judge_blackstone[y+1]),
                                      "bl6+2" : (self.judge_blackstone[y+2]),
                                      "bl6-1" : (None),
                                      "bl6-2" : (None),
# =============================================================================
#                                       "spacy1" : (self.loc_ent_X[y]),
#                                       "spacy1+1" : (self.loc_ent_X[y+1]),
#                                       "spacy1+2" : (self.loc_ent_X[y+2]),
#                                       "spacy1-1" : (None),
#                                       "spacy1-2" : (None),
#                                       "spacy2" : (self.org_ent_X[y]),
#                                       "spacy2+1" : (self.org_ent_X[y+1]),
#                                       "spacy2+2" : (self.org_ent_X[y+2]),
#                                       "spacy2-1" : (None),
#                                       "spacy2-2" : (None),
#                                       "spacy3" : (self.date_ent_X[y]),
#                                       "spacy3+1" : (self.date_ent_X[y+1]),
#                                       "spacy3+2" : (self.date_ent_X[y+2]),
#                                       "spacy3-1" : (None),
#                                       "spacy3-2" : (None),
#                                       "spacy4" : (self.person_ent_X[y]),
#                                       "spacy4+1" : (self.person_ent_X[y+1]),
#                                       "spacy4+2" : (self.person_ent_X[y+2]),
#                                       "spacy4-1" : (None),
#                                       "spacy4-2" : (None),
#                                       "spacy5" : (self.time_ent_X[y]),
#                                       "spacy5+1" : (self.time_ent_X[y+1]),
#                                       "spacy5+2" : (self.time_ent_X[y+2]),
#                                       "spacy5-1" : (None),
#                                       "spacy5-2" : (None),
#                                       "spacy6" : (self.gpe_ent_X[y]),
#                                       "spacy6+1" : (self.gpe_ent_X[y+1]),
#                                       "spacy6+2" : (self.gpe_ent_X[y+2]),
#                                       "spacy6-1" : (None),
#                                       "spacy6-2" : (None),
#                                       "spacy7" : (self.fac_ent_X[y]),
#                                       "spacy7+1" : (self.fac_ent_X[y+1]),
#                                       "spacy7+2" : (self.fac_ent_X[y+2]),
#                                       "spacy7-1" : (None),
#                                       "spacy7-2" : (None),
#                                       "spacy8" : (self.ordinal_ent_X[y]),
#                                       "spacy8+1" : (self.ordinal_ent_X[y+1]),
#                                       "spacy8+2" : (self.ordinal_ent_X[y+2]),
#                                       "spacy8-1" : (None),
#                                       "spacy8-2" : (None)
# =============================================================================
                                      })
        # second word of the sentence
            elif sentence_id == 2:
                sentence_features.update({"r-1" : tag_history[sentence_id-2],
                                      "r-2 r-1" : "<START> %s" % (tag_history[sentence_id-2]),
                                      "length" : (self.sent_length[y]),
                                      "length+1" : (self.sent_length[y+1]),
                                      "length+2" : (self.sent_length[y+2]),
                                      "length-1" : (self.sent_length[y-1]),
                                      "length-2" : (None),
# =============================================================================
#                                       "tfdif" : (self.tfidf_top20[y]),
#                                       "tfdif+1" : (self.tfidf_top20[y+1]),
#                                       "tfdif+2" : (self.tfidf_top20[y+2]),
#                                       "tfdif-1" : (self.tfidf_top20[y-1]),
#                                       "tfdif-2" : (None),
# =============================================================================
                                      "loc1" : (self.loc1_X[y]),
                                      "loc1+1" : (self.loc1_X[y+1]),
                                      "loc1+2" : (self.loc1_X[y+2]),
                                      "loc1-1" : (self.loc1_X[y-1]),
                                      "loc1-2" : (None),
                                      "loc2" : (self.loc2_X[y]),
                                      "loc2+1" : (self.loc2_X[y+1]),
                                      "loc2+2" : (self.loc2_X[y+2]),
                                      "loc2-1" : (self.loc1_X[y-1]),
                                      "loc2-2" : (None),
                                      "loc3" : (self.loc3_X[y]),
                                      "loc3+1" : (self.loc3_X[y+1]),
                                      "loc3+2" : (self.loc3_X[y+2]),
                                      "loc3-1" : (self.loc1_X[y-1]),
                                      "loc3-2" : (None),
                                      "loc4" : (self.loc4_X[y]),
                                      "loc4+1" : (self.loc4_X[y+1]),
                                      "loc4+2" : (self.loc4_X[y+2]),
                                      "loc4-1" : (self.loc1_X[y-1]),
                                      "loc4-2" : (None),
                                      "loc5" : (self.loc5_X[y]),
                                      "loc5+1" : (self.loc5_X[y+1]),
                                      "loc5+2" : (self.loc5_X[y+2]),
                                      "loc5-1" : (self.loc1_X[y-1]),
                                      "loc5-2" : (None),
                                      "loc6" : (self.loc6_X[y]),
                                      "loc6+1" : (self.loc6_X[y+1]),
                                      "loc6+2" : (self.loc6_X[y+2]),
                                      "loc6-1" : (self.loc1_X[y-1]),
                                      "loc6-2" : (None),
                                      "quote1" : (self.inq_X[y]),
                                      "quote1+1" : (self.inq_X[y+1]),
                                      "quote1+2" : (self.inq_X[y+2]),
                                      "quote1-1" : (self.inq_X[y-1]),
                                      "quote1-2" : (None),
                                      "quote2" : (self.qb_X[y]),
                                      "quote2+1" : (self.qb_X[y+1]),
                                      "quote2+2" : (self.qb_X[y+2]),
                                      "quote2-1" : (self.qb_X[y-1]),
                                      "quote2-2" : (None),
                                      "asmo1" : (self.agree_X[y]),
                                      "asmo1+1" : (self.agree_X[y+1]),
                                      "asmo1+2" : (self.agree_X[y+2]),
                                      "asmo1-1" : (self.agree_X[y-1]),
                                      "asmo1-2" : (None),
                                      "asmo2" : (self.outcome_X[y]),
                                      "asmo2+1" : (self.outcome_X[y+1]),
                                      "asmo2+2" : (self.outcome_X[y+2]),
                                      "asmo2-1" : (self.outcome_X[y-1]),
                                      "asmo2-2" : (None),
# =============================================================================
#                                       "cue1" : (self.asp_X[y]),
#                                       "cue1+1" : (self.asp_X[y+1]),
#                                       "cue1+2" : (self.asp_X[y+2]),
#                                       "cue1-1" : (self.asp_X[y-1]),
#                                       "cue1-2" : (None),
#                                       "cue2" : (self.modal_X[y]),
#                                       "cue2+1" : (self.modal_X[y+1]),
#                                       "cue2+2" : (self.modal_X[y+2]),
#                                       "cue2-1" : (self.modal_X[y-1]),
#                                       "cue2-2" : (None),
#                                       "cue3" : (self.voice_X[y]),
#                                       "cue3+1" : (self.voice_X[y+1]),
#                                       "cue3+2" : (self.voice_X[y+2]),
#                                       "cue3-1" : (self.voice_X[y-1]),
#                                       "cue3-2" : (None),
#                                       "cue4" : (self.negcue_X[y]),
#                                       "cue4+1" : (self.negcue_X[y+1]),
#                                       "cue4+2" : (self.negcue_X[y+2]),
#                                       "cue4-1" : (self.negcue_X[y-1]),
#                                       "cue4-2" : (None),
#                                       "cue5" : (self.tense_X[y]),
#                                       "cue5+1" : (self.tense_X[y+1]),
#                                       "cue5+2" : (self.tense_X[y+2]),
#                                       "cue5-1" : (self.tense_X[y-1]),
#                                       "cue5-2" : (None),
# =============================================================================
                                      "bl1" : (self.provision_blackstone[y]),
                                      "bl1+1" : (self.provision_blackstone[y+1]),
                                      "bl1+2" : (self.provision_blackstone[y+2]),
                                      "bl1-1" : (self.provision_blackstone[y-1]),
                                      "bl1-2" : (None),
                                      "bl2" : (self.instrument_blackstone[y]),
                                      "bl2+1" : (self.instrument_blackstone[y+1]),
                                      "bl2+2" : (self.instrument_blackstone[y+2]),
                                      "bl2-1" : (self.instrument_blackstone[y-1]),
                                      "bl2-2" : (None),
                                      "bl3" : (self.court_blackstone [y]),
                                      "bl3+1" : (self.court_blackstone [y+1]),
                                      "bl3+2" : (self.court_blackstone [y+2]),
                                      "bl3-1" : (self.court_blackstone [y-1]),
                                      "bl3-2" : (None),
                                      "bl4" : (self.case_blackstone[y]),
                                      "bl4+1" : (self.case_blackstone[y+1]),
                                      "bl4+2" : (self.case_blackstone[y+2]),
                                      "bl4-1" : (self.case_blackstone[y-1]),
                                      "bl4-2" : (None),
                                      "bl5" : (self.citation_blackstone[y]),
                                      "bl5+1" : (self.citation_blackstone[y+1]),
                                      "bl5+2" : (self.citation_blackstone[y+2]),
                                      "bl5-1" : (self.citation_blackstone[y-1]),
                                      "bl5-2" : (None),
                                      "bl6" : (self.judge_blackstone[y]),
                                      "bl6+1" : (self.judge_blackstone[y+1]),
                                      "bl6+2" : (self.judge_blackstone[y+2]),
                                      "bl6-1" : (self.judge_blackstone[y-1]),
                                      "bl6-2" : (None),
# =============================================================================
#                                       "spacy1" : (self.loc_ent_X[y]),
#                                       "spacy1+1" : (self.loc_ent_X[y+1]),
#                                       "spacy1+2" : (self.loc_ent_X[y+2]),
#                                       "spacy1-1" : (self.loc_ent_X[y-1]),
#                                       "spacy1-2" : (None),
#                                       "spacy2" : (self.org_ent_X[y]),
#                                       "spacy2+1" : (self.org_ent_X[y+1]),
#                                       "spacy2+2" : (self.org_ent_X[y+2]),
#                                       "spacy2-1" : (self.org_ent_X[y-1]),
#                                       "spacy2-2" : (None),
#                                       "spacy3" : (self.date_ent_X[y]),
#                                       "spacy3+1" : (self.date_ent_X[y+1]),
#                                       "spacy3+2" : (self.date_ent_X[y+2]),
#                                       "spacy3-1" : (self.date_ent_X[y-1]),
#                                       "spacy3-2" : (None),
#                                       "spacy4" : (self.person_ent_X[y]),
#                                       "spacy4+1" : (self.person_ent_X[y+1]),
#                                       "spacy4+2" : (self.person_ent_X[y+2]),
#                                       "spacy4-1" : (self.person_ent_X[y-1]),
#                                       "spacy4-2" : (None),
#                                       "spacy5" : (self.time_ent_X[y]),
#                                       "spacy5+1" : (self.time_ent_X[y+1]),
#                                       "spacy5+2" : (self.time_ent_X[y+2]),
#                                       "spacy5-1" : (self.time_ent_X[y-1]),
#                                       "spacy5-2" : (None),
#                                       "spacy6" : (self.gpe_ent_X[y]),
#                                       "spacy6+1" : (self.gpe_ent_X[y+1]),
#                                       "spacy6+2" : (self.gpe_ent_X[y+2]),
#                                       "spacy6-1" : (self.gpe_ent_X[y-1]),
#                                       "spacy6-2" : (None),
#                                       "spacy7" : (self.fac_ent_X[y]),
#                                       "spacy7+1" : (self.fac_ent_X[y+1]),
#                                       "spacy7+2" : (self.fac_ent_X[y+2]),
#                                       "spacy7-1" : (self.fac_ent_X[y-1]),
#                                       "spacy7-2" : (None),
#                                       "spacy8" : (self.ordinal_ent_X[y]),
#                                       "spacy8+1" : (self.ordinal_ent_X[y+1]),
#                                       "spacy8+2" : (self.ordinal_ent_X[y+2]),
#                                       "spacy8-1" : (self.ordinal_ent_X[y-1]),
#                                       "spacy8-2" : (None)
# =============================================================================
                                      })

            elif newSpeechLookAheadBy1:
                sentence_features.update({"r-1" : tag_history[sentence_id-2],
                                      "r-2 r-1" : "%s %s" % (tag_history[sentence_id-3], tag_history[sentence_id-2]),
                                      "length" : (self.sent_length[y]),
                                      "length+1" : (None),
                                      "length+2" : (None),
                                      "length-1" : (self.sent_length[y-1]),
                                      "length-2" : (self.sent_length[y-2]),
# =============================================================================
#                                       "tfdif" : (self.tfidf_top20[y]),
#                                       "tfdif+1" : (None),
#                                       "tfdif+2" : (None),
#                                       "tfdif-1" : (self.tfidf_top20[y-1]),
#                                       "tfdif-2" : (self.tfidf_top20[y-2]),
# =============================================================================
                                      "loc1" : (self.loc1_X[y]),
                                      "loc1+1" : (None),
                                      "loc1+2" : (None),
                                      "loc1-1" : (self.loc1_X[y-1]),
                                      "loc1-2" : (self.loc1_X[y-2]),
                                      "loc2" : (self.loc1_X[y]),
                                      "loc2+1" : (None),
                                      "loc2+2" : (None),
                                      "loc2-1" : (self.loc1_X[y-1]),
                                      "loc2-2" : (self.loc1_X[y-2]),
                                      "loc3" : (self.loc1_X[y]),
                                      "loc3+1" : (None),
                                      "loc3+2" : (None),
                                      "loc3-1" : (self.loc1_X[y-1]),
                                      "loc3-2" : (self.loc1_X[y-2]),
                                      "loc4" : (self.loc1_X[y]),
                                      "loc4+1" : (None),
                                      "loc4+2" : (None),
                                      "loc4-1" : (self.loc1_X[y-1]),
                                      "loc4-2" : (self.loc1_X[y-2]),
                                      "loc5" : (self.loc1_X[y]),
                                      "loc5+1" : (None),
                                      "loc5+2" : (None),
                                      "loc5-1" : (self.loc1_X[y-1]),
                                      "loc5-2" : (self.loc1_X[y-2]),
                                      "loc6" :  (self.loc1_X[y]),
                                      "loc6+1" : (None),
                                      "loc6+2" : (None),
                                      "loc6-1" : (self.loc1_X[y-1]),
                                      "loc6-2" : (self.loc1_X[y-2]),
                                      "quote1" : (self.inq_X[y]),
                                      "quote1+1" : (None),
                                      "quote1+2" : (None),
                                      "quote1-1" : (self.inq_X[y-1]),
                                      "quote1-2" : (self.inq_X[y-2]),
                                      "quote2" :  (self.qb_X[y]),
                                      "quote2+1" : (None),
                                      "quote2+2" : (None),
                                      "quote2-1" : (self.qb_X[y-1]),
                                      "quote2-2" : (self.inq_X[y-2]),
                                      "asmo1" : (self.agree_X[y]),
                                      "asmo1+1" : (None),
                                      "asmo1+2" : (None),
                                      "asmo1-1" : (self.agree_X[y-1]),
                                      "asmo1-2" : (self.agree_X[y-2]),
                                      "asmo2" : (self.outcome_X[y]),
                                      "asmo2+1" : (None),
                                      "asmo2+2" : (None),
                                      "asmo2-1" : (self.outcome_X[y-1]),
                                      "asmo2-2" : (self.outcome_X[y-2]),
# =============================================================================
#                                       "cue1" : (self.asp_X[y]),
#                                       "cue1+1" : (None),
#                                       "cue1+2" : (None),
#                                       "cue1-1" : (self.asp_X[y-1]),
#                                       "cue1-2" : (self.asp_X[y-2]),
#                                       "cue2" : (self.modal_X[y]),
#                                       "cue2+1" : (None),
#                                       "cue2+2" : (None),
#                                       "cue2-1" : (self.modal_X[y-1]),
#                                       "cue2-2" : (self.modal_X[y-2]),
#                                       "cue3" : (self.voice_X[y]),
#                                       "cue3+1" : (None),
#                                       "cue3+2" : (None),
#                                       "cue3-1" : (self.voice_X[y-1]),
#                                       "cue3-2" : (self.voice_X[y-2]),
#                                       "cue4" : (self.negcue_X[y]),
#                                       "cue4+1" : (None),
#                                       "cue4+2" : (None),
#                                       "cue4-1" : (self.negcue_X[y-1]),
#                                       "cue4-2" : (self.negcue_X[y-2]),
#                                       "cue5" : (self.tense_X[y]),
#                                       "cue5+1" : (None),
#                                       "cue5+2" : (None),
#                                       "cue5-1" : (self.tense_X[y-1]),
#                                       "cue5-2" : (self.tense_X[y-2]),
# =============================================================================
                                      "bl1" : (self.provision_blackstone[y]),
                                      "bl1+1" : (None),
                                      "bl1+2" : (None),
                                      "bl1-1" : (self.provision_blackstone[y-1]),
                                      "bl1-2" : (self.provision_blackstone[y-2]),
                                      "bl2" : (self.instrument_blackstone[y]),
                                      "bl2+1" : (None),
                                      "bl2+2" : (None),
                                      "bl2-1" : (self.instrument_blackstone[y-1]),
                                      "bl2-2" : (self.instrument_blackstone[y-2]),
                                      "bl3" : (self.court_blackstone [y]),
                                      "bl3+1" : (None),
                                      "bl3+2" : (None),
                                      "bl3-1" : (self.court_blackstone [y-1]),
                                      "bl3-2" : (self.court_blackstone [y-2]),
                                      "bl4" : (self.case_blackstone[y]),
                                      "bl4+1" : (None),
                                      "bl4+2" : (None),
                                      "bl4-1" : (self.case_blackstone[y-1]),
                                      "bl4-2" : (self.case_blackstone[y-2]),
                                      "bl5" :  (self.citation_blackstone[y]),
                                      "bl5+1" : (None),
                                      "bl5+2" : (None),
                                      "bl5-1" : (self.citation_blackstone[y-1]),
                                      "bl5-2" : (self.citation_blackstone[y-2]),
                                      "bl6" : (self.judge_blackstone[y]),
                                      "bl6+1" : (None),
                                      "bl6+2" : (None),
                                      "bl6-1" : (self.judge_blackstone[y-1]),
                                      "bl6-2" : (self.judge_blackstone[y-2]),
# =============================================================================
#                                       "spacy1" : (self.loc_ent_X[y]),
#                                       "spacy1+1" : (None),
#                                       "spacy1+2" : (None),
#                                       "spacy1-1" : (self.loc_ent_X[y-1]),
#                                       "spacy1-2" : (self.loc_ent_X[y-2]),
#                                       "spacy2" : (self.org_ent_X[y]),
#                                       "spacy2+1" : (None),
#                                       "spacy2+2" : (None),
#                                       "spacy2-1" : (self.org_ent_X[y-1]),
#                                       "spacy2-2" : (self.org_ent_X[y-2]),
#                                       "spacy3" : (self.date_ent_X[y]),
#                                       "spacy3+1" : (None),
#                                       "spacy3+2" : (None),
#                                       "spacy3-1" : (self.date_ent_X[y-1]),
#                                       "spacy3-2" : (self.date_ent_X[y-2]),
#                                       "spacy4" : (self.person_ent_X[y]),
#                                       "spacy4+1" : (None),
#                                       "spacy4+2" : (None),
#                                       "spacy4-1" : (self.person_ent_X[y-1]),
#                                       "spacy4-2" : (self.person_ent_X[y-2]),
#                                       "spacy5" : (self.time_ent_X[y]),
#                                       "spacy5+1" : (None),
#                                       "spacy5+2" : (None),
#                                       "spacy5-1" : (self.time_ent_X[y-1]),
#                                       "spacy5-2" : (self.time_ent_X[y-2]),
#                                       "spacy6" : (self.gpe_ent_X[y]),
#                                       "spacy6+1" : (None),
#                                       "spacy6+2" : (None),
#                                       "spacy6-1" : (self.gpe_ent_X[y-1]),
#                                       "spacy6-2" : (self.gpe_ent_X[y-2]),
#                                       "spacy7" : (self.ordinal_ent_X[y]),
#                                       "spacy7+1" : (None),
#                                       "spacy7+2" : (None),
#                                       "spacy7-1" : (self.ordinal_ent_X[y-1]),
#                                       "spacy7-2" : (self.fac_ent_X[y-2]),
#                                       "spacy8" : (self.ordinal_ent_X[y]),
#                                       "spacy8+1" : (None),
#                                       "spacy8+2" : (None),
#                                       "spacy8-1" : (self.ordinal_ent_X[y-1]),
#                                       "spacy8-2" : (self.ordinal_ent_X[y-2])
# =============================================================================
                                      })
            elif newSpeechLookAheadBy2:
                sentence_features.update({"r-1" : tag_history[sentence_id-2],
                                      "r-2 r-1" : "%s %s" % (tag_history[sentence_id-3], tag_history[sentence_id-2]),
                                      "length" : (self.sent_length[y]),
                                      "length+1" : (self.sent_length[y+1]),
                                      "length+2" :  (None),
                                      "length-1" : (self.sent_length[y-1]),
                                      "length-2" : (self.sent_length[y-2]),
# =============================================================================
#                                       "tfdif" : (self.tfidf_top20[y]),
#                                       "tfdif+1" : (self.tfidf_top20[y+1]),
#                                       "tfdif+2" : (None),
#                                       "tfdif-1" : (self.tfidf_top20[y-1]),
#                                       "tfdif-2" : (self.tfidf_top20[y-2]),
# =============================================================================
                                      "loc1" : (self.loc1_X[y]),
                                      "loc1+1" : (self.loc1_X[y+1]),
                                      "loc1+2" : (None),
                                      "loc1-1" : (self.loc1_X[y-1]),
                                      "loc1-2" : (self.loc1_X[y-2]),
                                      "loc2" : (self.loc2_X[y]),
                                      "loc2+1" : (self.loc2_X[y+1]),
                                      "loc2+2" : (None),
                                      "loc2-1" : (self.loc1_X[y-1]),
                                      "loc2-2" : (self.loc1_X[y-2]),
                                      "loc3" : (self.loc3_X[y]),
                                      "loc3+1" : (self.loc3_X[y+1]),
                                      "loc3+2" : (None),
                                      "loc3-1" : (self.loc1_X[y-1]),
                                      "loc3-2" : (self.loc1_X[y-2]),
                                      "loc4" : (self.loc4_X[y]),
                                      "loc4+1" : (self.loc4_X[y+1]),
                                      "loc4+2" : (None),
                                      "loc4-1" : (self.loc1_X[y-1]),
                                      "loc4-2" : (self.loc1_X[y-2]),
                                      "loc5" : (self.loc5_X[y]),
                                      "loc5+1" : (self.loc5_X[y+1]),
                                      "loc5+2" : (None),
                                      "loc5-1" : (self.loc1_X[y-1]),
                                      "loc5-2" : (self.loc1_X[y-2]),
                                      "loc6" : (self.loc6_X[y]),
                                      "loc6+1" : (self.loc6_X[y+1]),
                                      "loc6+2" : (None),
                                      "loc6-1" : (self.loc1_X[y-1]),
                                      "loc6-2" : (self.loc1_X[y-2]),
                                      "quote1" : (self.inq_X[y]),
                                      "quote1+1" : (self.inq_X[y+1]),
                                      "quote1+2" : (None),
                                      "quote1-1" : (self.inq_X[y-1]),
                                      "quote1-2" : (self.inq_X[y-2]),
                                      "quote2" : (self.qb_X[y]),
                                      "quote2+1" : (self.qb_X[y+1]),
                                      "quote2+2" : (None),
                                      "quote2-1" : (self.qb_X[y-1]),
                                      "quote2-2" : (self.inq_X[y-2]),
                                      "asmo1" : (self.agree_X[y]),
                                      "asmo1+1" : (self.agree_X[y+1]),
                                      "asmo1+2" : (None),
                                      "asmo1-1" : (self.agree_X[y-1]),
                                      "asmo1-2" : (self.agree_X[y-2]),
                                      "asmo2" : (self.outcome_X[y]),
                                      "asmo2+1" : (self.outcome_X[y+1]),
                                      "asmo2+2" : (None),
                                      "asmo2-1" : (self.outcome_X[y-1]),
                                      "asmo2-2" : (self.outcome_X[y-2]),
# =============================================================================
#                                       "cue1" : (self.asp_X[y]),
#                                       "cue1+1" : (self.asp_X[y+1]),
#                                       "cue1+2" : (None),
#                                       "cue1-1" : (self.asp_X[y-1]),
#                                       "cue1-2" : (self.asp_X[y-2]),
#                                       "cue2" : (self.modal_X[y]),
#                                       "cue2+1" : (self.modal_X[y+1]),
#                                       "cue2+2" : (None),
#                                       "cue2-1" : (self.modal_X[y-1]),
#                                       "cue2-2" : (self.modal_X[y-2]),
#                                       "cue3" : (self.voice_X[y]),
#                                       "cue3+1" : (self.voice_X[y+1]),
#                                       "cue3+2" : (None),
#                                       "cue3-1" : (self.voice_X[y-1]),
#                                       "cue3-2" : (self.voice_X[y-2]),
#                                       "cue4" : (self.negcue_X[y]),
#                                       "cue4+1" : (self.negcue_X[y+1]),
#                                       "cue4+2" : (None),
#                                       "cue4-1" : (self.negcue_X[y-1]),
#                                       "cue4-2" : (self.negcue_X[y-2]),
#                                       "cue5" : (self.tense_X[y]),
#                                       "cue5+1" : (self.tense_X[y+1]),
#                                       "cue5+2" : (None),
#                                       "cue5-1" : (self.tense_X[y-1]),
#                                       "cue5-2" : (self.tense_X[y-2]),
# =============================================================================
                                      "bl1" : (self.provision_blackstone[y]),
                                      "bl1+1" : (self.provision_blackstone[y+1]),
                                      "bl1+2" : (None),
                                      "bl1-1" : (self.provision_blackstone[y-1]),
                                      "bl1-2" : (self.provision_blackstone[y-2]),
                                      "bl2" : (self.instrument_blackstone[y]),
                                      "bl2+1" : (self.instrument_blackstone[y+1]),
                                      "bl2+2" : (None),
                                      "bl2-1" : (self.instrument_blackstone[y-1]),
                                      "bl2-2" : (self.instrument_blackstone[y-2]),
                                      "bl3" : (self.court_blackstone [y]),
                                      "bl3+1" : (self.court_blackstone [y+1]),
                                      "bl3+2" : (None),
                                      "bl3-1" : (self.court_blackstone [y-1]),
                                      "bl3-2" : (self.court_blackstone [y-2]),
                                      "bl4" : (self.case_blackstone[y]),
                                      "bl4+1" : (self.case_blackstone[y+1]),
                                      "bl4+2" : (None),
                                      "bl4-1" : (self.case_blackstone[y-1]),
                                      "bl4-2" : (self.case_blackstone[y-2]),
                                      "bl5" : (self.citation_blackstone[y]),
                                      "bl5+1" : (self.citation_blackstone[y+1]),
                                      "bl5+2" : (None),
                                      "bl5-1" : (self.citation_blackstone[y-1]),
                                      "bl5-2" : (self.citation_blackstone[y-2]),
                                      "bl6" : (self.judge_blackstone[y]),
                                      "bl6+1" : (self.judge_blackstone[y+1]),
                                      "bl6+2" : (None),
                                      "bl6-1" : (self.judge_blackstone[y-1]),
                                      "bl6-2" : (self.judge_blackstone[y-2]),
# =============================================================================
#                                       "spacy1" : (self.loc_ent_X[y]),
#                                       "spacy1+1" : (self.loc_ent_X[y+1]),
#                                       "spacy1+2" : (None),
#                                       "spacy1-1" : (self.loc_ent_X[y-1]),
#                                       "spacy1-2" : (self.loc_ent_X[y-2]),
#                                       "spacy2" : (self.org_ent_X[y]),
#                                       "spacy2+1" : (self.org_ent_X[y+1]),
#                                       "spacy2+2" : (None),
#                                       "spacy2-1" : (self.org_ent_X[y-1]),
#                                       "spacy2-2" : (self.org_ent_X[y-2]),
#                                       "spacy3" : (self.date_ent_X[y]),
#                                       "spacy3+1" : (self.date_ent_X[y+1]),
#                                       "spacy3+2" : (None),
#                                       "spacy3-1" : (self.date_ent_X[y-1]),
#                                       "spacy3-2" : (self.date_ent_X[y-2]),
#                                       "spacy4" : (self.person_ent_X[y]),
#                                       "spacy4+1" : (self.person_ent_X[y+1]),
#                                       "spacy4+2" : (None),
#                                       "spacy4-1" : (self.person_ent_X[y-1]),
#                                       "spacy4-2" : (self.person_ent_X[y-2]),
#                                       "spacy5" : (self.time_ent_X[y]),
#                                       "spacy5+1" : (self.time_ent_X[y+1]),
#                                       "spacy5+2" : (None),
#                                       "spacy5-1" : (self.time_ent_X[y-1]),
#                                       "spacy5-2" : (self.time_ent_X[y-2]),
#                                       "spacy6" : (self.gpe_ent_X[y]),
#                                       "spacy6+1" : (self.gpe_ent_X[y+1]),
#                                       "spacy6+2" : (None),
#                                       "spacy6-1" : (self.gpe_ent_X[y-1]),
#                                       "spacy6-2" : (self.gpe_ent_X[y-2]),
#                                       "spacy7" : (self.fac_ent_X[y]),
#                                       "spacy7+1" : (self.fac_ent_X[y+1]),
#                                       "spacy7+2" : (None),
#                                       "spacy7-1" : (self.fac_ent_X[y-1]),
#                                       "spacy7-2" : (self.fac_ent_X[y-2]),
#                                       "spacy8" : (self.ordinal_ent_X[y]),
#                                       "spacy8+1" : (self.ordinal_ent_X[y+1]),
#                                       "spacy8+2" : (None),
#                                       "spacy8-1" : (self.ordinal_ent_X[y-1]),
#                                       "spacy8-2" : (self.ordinal_ent_X[y-2])
# =============================================================================
                                      })

            else:
                sentence_features.update({"r-1" : tag_history[sentence_id-2],
                                      "r-2 r-1" : "%s %s" % (tag_history[sentence_id-3], tag_history[sentence_id-2]),
                                      "length" : (self.sent_length[y]),
                                      "length+1" : (self.sent_length[y+1]),
                                      "length+2" : (self.sent_length[y+2]),
                                      "length-1" : (self.sent_length[y-1]),
                                      "length-2" : (self.sent_length[y-2]),
# =============================================================================
#                                       "tfdif" : (self.tfidf_top20[y]),
#                                       "tfdif+1" : (self.tfidf_top20[y+1]),
#                                       "tfdif+2" : (self.tfidf_top20[y+2]),
#                                       "tfdif-1" : (self.tfidf_top20[y-1]),
#                                       "tfdif-2" : (self.tfidf_top20[y-2]),
# =============================================================================
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
# =============================================================================
#                                       "cue1" : (self.asp_X[y]),
#                                       "cue1+1" : (self.asp_X[y+1]),
#                                       "cue1+2" : (self.asp_X[y+2]),
#                                       "cue1-1" : (self.asp_X[y-1]),
#                                       "cue1-2" : (self.asp_X[y-2]),
#                                       "cue2" : (self.modal_X[y]),
#                                       "cue2+1" : (self.modal_X[y+1]),
#                                       "cue2+2" : (self.modal_X[y+2]),
#                                       "cue2-1" : (self.modal_X[y-1]),
#                                       "cue2-2" : (self.modal_X[y-2]),
#                                       "cue3" : (self.voice_X[y]),
#                                       "cue3+1" : (self.voice_X[y+1]),
#                                       "cue3+2" : (self.voice_X[y+2]),
#                                       "cue3-1" : (self.voice_X[y-1]),
#                                       "cue3-2" : (self.voice_X[y-2]),
#                                       "cue4" : (self.negcue_X[y]),
#                                       "cue4+1" : (self.negcue_X[y+1]),
#                                       "cue4+2" : (self.negcue_X[y+2]),
#                                       "cue4-1" : (self.negcue_X[y-1]),
#                                       "cue4-2" : (self.negcue_X[y-2]),
#                                       "cue5" : (self.tense_X[y]),
#                                       "cue5+1" : (self.tense_X[y+1]),
#                                       "cue5+2" : (self.tense_X[y+2]),
#                                       "cue5-1" : (self.tense_X[y-1]),
#                                       "cue5-2" : (self.tense_X[y-2]),
# =============================================================================
                                      "bl1" : (self.provision_blackstone[y]),
                                      "bl1+1" : (self.provision_blackstone[y+1]),
                                      "bl1+2" : (self.provision_blackstone[y+2]),
                                      "bl1-1" : (self.provision_blackstone[y-1]),
                                      "bl1-2" : (self.provision_blackstone[y-2]),
                                      "bl2" : (self.instrument_blackstone[y]),
                                      "bl2+1" : (self.instrument_blackstone[y+1]),
                                      "bl2+2" : (self.instrument_blackstone[y+2]),
                                      "bl2-1" : (self.instrument_blackstone[y-1]),
                                      "bl2-2" : (self.instrument_blackstone[y-2]),
                                      "bl3" : (self.court_blackstone [y]),
                                      "bl3+1" : (self.court_blackstone [y+1]),
                                      "bl3+2" : (self.court_blackstone [y+2]),
                                      "bl3-1" : (self.court_blackstone [y-1]),
                                      "bl3-2" : (self.court_blackstone [y-2]),
                                      "bl4" : (self.case_blackstone[y]),
                                      "bl4+1" : (self.case_blackstone[y+1]),
                                      "bl4+2" : (self.case_blackstone[y+2]),
                                      "bl4-1" : (self.case_blackstone[y-1]),
                                      "bl4-2" : (self.case_blackstone[y-2]),
                                      "bl5" : (self.citation_blackstone[y]),
                                      "bl5+1" : (self.citation_blackstone[y+1]),
                                      "bl5+2" : (self.citation_blackstone[y+2]),
                                      "bl5-1" : (self.citation_blackstone[y-1]),
                                      "bl5-2" : (self.citation_blackstone[y-2]),
                                      "bl6" : (self.judge_blackstone[y]),
                                      "bl6+1" : (self.judge_blackstone[y+1]),
                                      "bl6+2" : (self.judge_blackstone[y+2]),
                                      "bl6-1" : (self.judge_blackstone[y-1]),
                                      "bl6-2" : (self.judge_blackstone[y-2]),
# =============================================================================
#                                       "spacy1" : (self.loc_ent_X[y]),
#                                       "spacy1+1" : (self.loc_ent_X[y+1]),
#                                       "spacy1+2" : (self.loc_ent_X[y+2]),
#                                       "spacy1-1" : (self.loc_ent_X[y-1]),
#                                       "spacy1-2" : (self.loc_ent_X[y-2]),
#                                       "spacy2" : (self.org_ent_X[y]),
#                                       "spacy2+1" : (self.org_ent_X[y+1]),
#                                       "spacy2+2" : (self.org_ent_X[y+2]),
#                                       "spacy2-1" : (self.org_ent_X[y-1]),
#                                       "spacy2-2" : (self.org_ent_X[y-2]),
#                                       "spacy3" : (self.date_ent_X[y]),
#                                       "spacy3+1" : (self.date_ent_X[y+1]),
#                                       "spacy3+2" : (self.date_ent_X[y+2]),
#                                       "spacy3-1" : (self.date_ent_X[y-1]),
#                                       "spacy3-2" : (self.date_ent_X[y-2]),
#                                       "spacy4" : (self.person_ent_X[y]),
#                                       "spacy4+1" : (self.person_ent_X[y+1]),
#                                       "spacy4+2" : (self.person_ent_X[y+2]),
#                                       "spacy4-1" : (self.person_ent_X[y-1]),
#                                       "spacy4-2" : (self.person_ent_X[y-2]),
#                                       "spacy5" : (self.time_ent_X[y]),
#                                       "spacy5+1" : (self.time_ent_X[y+1]),
#                                       "spacy5+2" : (self.time_ent_X[y+2]),
#                                       "spacy5-1" : (self.time_ent_X[y-1]),
#                                       "spacy5-2" : (self.time_ent_X[y-2]),
#                                       "spacy6" : (self.gpe_ent_X[y]),
#                                       "spacy6+1" : (self.gpe_ent_X[y+1]),
#                                       "spacy6+2" : (self.gpe_ent_X[y+2]),
#                                       "spacy6-1" : (self.gpe_ent_X[y-1]),
#                                       "spacy6-2" : (self.gpe_ent_X[y-2]),
#                                       "spacy7" : (self.fac_ent_X[y]),
#                                       "spacy7+1" : (self.fac_ent_X[y+1]),
#                                       "spacy7+2" : (self.fac_ent_X[y+2]),
#                                       "spacy7-1" : (self.fac_ent_X[y-1]),
#                                       "spacy7-2" : (self.fac_ent_X[y-2]),
#                                       "spacy8" : (self.ordinal_ent_X[y]),
#                                       "spacy8+1" : (self.ordinal_ent_X[y+1]),
#                                       "spacy8+2" : (self.ordinal_ent_X[y+2]),
#                                       "spacy8-1" : (self.ordinal_ent_X[y-1]),
#                                       "spacy8-2" : (self.ordinal_ent_X[y-2])
# =============================================================================
                                      })



            print(sentence_features)
            return sentence_features


    # here it assigns relevant training data to the np.array, then can use the index to pull out the right rows
    def pull_training_data(self):
        # open up the MLdata
        # create the sentences arrays at the same time? including relevant
        # role is the issue - create an object (dictionary) that has all the
        # relevant features - as above, plus s-1, and s-2 ...
        with open('./data/MLdata_train_seq.csv', 'r') as infile:
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
                self.tfidf_top20_X = np.append(self.tfidf_top20_X, [float(row['tfidf_top20'])])
                self.asp_X = np.append(self.asp_X, [float(row['aspect'])])
                self.modal_X = np.append(self.modal_X, [float(row['modal'])])
                self.voice_X = np.append(self.voice_X, [float(row['voice'])])
                self.negcue_X = np.append(self.negcue_X, [float(row['negation'])])
                self.tense_X = np.append(self.tense_X, [float(row['tense'])])
                self.provision_blackstone = np.append(self.provision_blackstone, [float(row['provision ent'])])
                self.instrument_blackstone = np.append(self.instrument_blackstone, [float(row['instrument ent'])])
                self.court_blackstone = np.append(self.court_blackstone, [float(row['court ent'])])
                self.case_blackstone = np.append(self.case_blackstone, [float(row['case name ent'])])
                self.citation_blackstone = np.append(self.citation_blackstone, [float(row['citation bl ent'])])
                self.judge_blackstone = np.append(self.judge_blackstone, [float(row['judge ent'])])
                self.loc_ent_X = np.append(self.loc_ent_X, [float(row['loc ent'])])
                self.org_ent_X = np.append(self.org_ent_X, [float(row['org ent'])])
                self.date_ent_X = np.append(self.date_ent_X, [float(row['date ent'])])
                self.person_ent_X = np.append(self.person_ent_X, [float(row['person ent'])])
                self.time_ent_X = np.append(self.time_ent_X, [float(row['time ent'])])
                self.gpe_ent_X = np.append(self.gpe_ent_X, [float(row['gpe ent'])])
                self.fac_ent_X = np.append(self.fac_ent_X, [float(row['fac ent'])])
                self.ordinal_ent_X = np.append(self.ordinal_ent_X, [float(row['ordinal ent'])])
                self.judgename.append(row['judgename'])
                self.rhetlabel.append(row['rhet label'])

        self.location = self.loc1_X, self.loc2_X, self.loc3_X, self.loc4_X, self.loc5_X, self.loc6_X
        self.quote = self.inq_X, self.qb_X
        self.asmo = self.agree_X, self.outcome_X
        self.cue_phrase = self.asp_X, self.modal_X, self.voice_X, self.negcue_X, self.tense_X
        self.sent_length =  self.sentlen_X
        self.tfidf_top20 = self.tfidf_top20_X
        self.blackstone = self.provision_blackstone, self.instrument_blackstone, self.court_blackstone, self.case_blackstone,
        self.citation_blackstone, self.judge_blackstone
        self.spacy = self.loc_ent_X, self.org_ent_X, self.date_ent_X, self.person_ent_X, self.time_ent_X, self.gpe_ent_X, self.fac_ent_X, self.ordinal_ent_X

    def pull_testing_data(self):
        # open up the MLdata
        # create the sentences arrays at the same time? including relevant
        # role is the issue - create an object (dictionary) that has all the
        # relevant features - as above, plus s-1, and s-2 ...
        with open('./data/MLdata_test_seq.csv', 'r') as infile:
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
                self.tfidf_top20_X = np.append(self.tfidf_top20_X, [float(row['tfidf_top20'])])
                self.asp_X = np.append(self.asp_X, [float(row['aspect'])])
                self.modal_X = np.append(self.modal_X, [float(row['modal'])])
                self.voice_X = np.append(self.voice_X, [float(row['voice'])])
                self.negcue_X = np.append(self.negcue_X, [float(row['negation'])])
                self.tense_X = np.append(self.tense_X, [float(row['tense'])])
                self.provision_blackstone = np.append(self.provision_blackstone, [float(row['provision ent'])])
                self.instrument_blackstone = np.append(self.instrument_blackstone, [float(row['instrument ent'])])
                self.court_blackstone = np.append(self.court_blackstone, [float(row['court ent'])])
                self.case_blackstone = np.append(self.case_blackstone, [float(row['case name ent'])])
                self.citation_blackstone = np.append(self.citation_blackstone, [float(row['citation bl ent'])])
                self.judge_blackstone = np.append(self.judge_blackstone, [float(row['judge ent'])])
                self.loc_ent_X = np.append(self.loc_ent_X, [float(row['loc ent'])])
                self.org_ent_X = np.append(self.org_ent_X, [float(row['org ent'])])
                self.date_ent_X = np.append(self.date_ent_X, [float(row['date ent'])])
                self.person_ent_X = np.append(self.person_ent_X, [float(row['person ent'])])
                self.time_ent_X = np.append(self.time_ent_X, [float(row['time ent'])])
                self.gpe_ent_X = np.append(self.gpe_ent_X, [float(row['gpe ent'])])
                self.fac_ent_X = np.append(self.fac_ent_X, [float(row['fac ent'])])
                self.ordinal_ent_X = np.append(self.ordinal_ent_X, [float(row['ordinal ent'])])
                self.judgename.append(row['judgename'])
                self.rhetlabel.append(row['rhet label'])

        self.location = self.loc1_X, self.loc2_X, self.loc3_X, self.loc4_X, self.loc5_X, self.loc6_X
        self.quote = self.inq_X, self.qb_X
        self.asmo = self.agree_X, self.outcome_X
        self.cue_phrase = self.asp_X, self.modal_X, self.voice_X, self.negcue_X, self.tense_X
        self.sent_length =  self.sentlen_X
        self.tfidf_top20 = self.tfidf_top20_X
        self.blackstone = self.provision_blackstone, self.instrument_blackstone, self.court_blackstone, self.case_blackstone,
        self.citation_blackstone, self.judge_blackstone
        self.spacy = self.loc_ent_X, self.org_ent_X, self.date_ent_X, self.person_ent_X, self.time_ent_X, self.gpe_ent_X, self.fac_ent_X, self.ordinal_ent_X
seq = seq()
seq.train_seq()
