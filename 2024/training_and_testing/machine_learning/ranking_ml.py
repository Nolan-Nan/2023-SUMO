
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
similar code as running ml, however ml.py tests the data (allowing F-1 scores, and
other metrics) - this file creates corpus_ranking.csv to attach a ranking to each
sentence in the corpus based on the predict_proba score which is then converted
in to a ranking out of 100 as to how confident the system is that the sentence is
relevant or not

this is mainly used to test the ranking system, prior to applying it on new data

@author: amyconroy
"""

import csv
import numpy as np

class ranking_ml:
  def __init__(self):
        self.case_id = np.array([])
        self.sent_id = np.array([])
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

        # all values are 0, thus non-beneficial in ml
        # self.caseent_X = np.array([])
        ##for cue phrase feature-set
        self.asp_X = np.array([])
        self.modal_X = np.array([])
        self.voice_X = np.array([])
        self.negcue_X = np.array([])
        self.tense_X = np.array([])

  def createRankingFile(case_flag, sent_flag, y, rank_flag):
    with open('./data/corpus_ranking.csv', 'w', newline='') as outfile:
        fieldnames = ['case_id', 'sent_id', 'rank']

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for v in range(len(y)):
            writer.writerow({'case_id': case_flag[v], 'sent_id': sent_flag[v], 'rank': rank_flag[v]})

  def supervised_ml(self, X, Y, label, feat_names, target_names, mode):
        print("Using train_test_split ")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size = .1)

        classifier, clf_name = mode.select_classifier(label)
        classifier.fit(X_train, y_train)
        rank = classifier.predict_proba(X)[:, 1]
        for v in enumerate(rank):
            print(v) # first value is the sentence number
            print(rank[v[0]]) # this gives the appropriate ranking
        print(rank)

    #Extract all data and prepare it for ML
  def prep_data(self, filename):
        with open('./data/' + filename + '.csv', 'r') as infile:
            reader = csv.DictReader(infile)

        # for each row in the MLDATA cv file, get the corresponding result - add to array
            for row in reader:
                if not isinstance(row['case_id'], str):
                    self.case_id = np.append(self.case_id, [float(row['case_id'])])
                self.sent_id = np.append(self.sent_id, [float(row['sent_id'])])
                self.rel_y = np.append(self.rel_y, [float(row['align'])])
                self.agree_X = np.append(self.agree_X, [float(row['agree'])])
                self.outcome_X = np.append(self.outcome_X, [float(row['outcome'])])
                self.loc1_X = np.append(self.loc1_X, [float(row['loc1'])])
                self.loc2_X = np.append(self.loc2_X, [float(row['loc2'])])
                self.loc3_X = np.append(self.loc3_X, [float(row['loc3'])])
                self.loc4_X = np.append(self.loc4_X, [float(row['loc4'])])
                self.loc5_X = np.append(self.loc5_X, [float(row['loc5'])])
                self.loc6_X = np.append(self.loc6_X, [float(row['loc6'])])
                self.HGloc1_X = np.append(self.HGloc1_X, [float(row['HGloc1'])])
                self.HGloc2_X = np.append(self.HGloc2_X, [float(row['HGloc2'])])
                self.HGloc3_X = np.append(self.HGloc3_X, [float(row['HGloc3'])])
                self.HGloc4_X = np.append(self.HGloc4_X, [float(row['HGloc4'])])
                self.HGloc5_X = np.append(self.HGloc5_X, [float(row['HGloc5'])])
                self.HGloc6_X = np.append(self.HGloc6_X, [float(row['HGloc6'])])
                self.sentlen_X = np.append(self.sentlen_X, [float(row['sentlen'])])
                self.HGsentlen_X = np.append(self.HGsentlen_X, [float(row['HGsentlen'])])
                self.qb_X = np.append(self.qb_X, [float(row['quoteblock'])])
                self.inq_X = np.append(self.inq_X, [float(row['inline_q'])])
                self.rhet_X = np.append(self.rhet_X, [float(row['rhet'])])
                self.tfidf_max_X = np.append(self.tfidf_max_X, [float(row['tfidf_max'])])
                self.tfidf_top20_X = np.append(self.tfidf_top20_X, [float(row['tfidf_top20'])])
                self.tfidf_HGavg_X = np.append(self.tfidf_HGavg_X, [float(row['tfidf_HGavg'])])
                self.asp_X = np.append(self.asp_X, [float(row['aspect'])])
                self.modal_X = np.append(self.modal_X, [float(row['modal'])])
                self.voice_X = np.append(self.voice_X, [float(row['voice'])])
                self.negcue_X = np.append(self.negcue_X, [float(row['negation'])])
                self.tense_X = np.append(self.tense_X, [float(row['tense'])])
                self.legalent_X = np.append(self.legalent_X, [float(row['legal entities'])])
                self.enamex_X = np.append(self.enamex_X, [float(row['enamex'])])
                self.rhet_y = np.append(self.rhet_y, [float(row['rhet_target'])])
                self.wordlist_X = np.append(self.wordlist_X, [float(row['wordlist'])])
                self.pasttense_X = np.append(self.pasttense_X, [float(row['past tense'])])

  def exec(self):
        location = self.loc1_X, self.loc2_X, self.loc3_X, self.loc4_X, self.loc5_X, self.loc6_X
        HGlocation = self.HGloc1_X, self.HGloc2_X, self.HGloc3_X, self.HGloc4_X, self.HGloc5_X, self.HGloc6_X
        quotation = self.inq_X, self.qb_X
        entities = self.legalent_X, self.enamex_X
        asmo = self.agree_X, self.outcome_X
        cue_phrase = self.asp_X, self.modal_X, self.voice_X, self.negcue_X, self.tense_X
        sent_length = self.sentlen_X
        HGsent_length = self.HGsentlen_X
        tfidf_max = self.tfidf_max_X
        tfidf_top20 = self.tfidf_top20_X
        tfidf_HGavg = self.tfidf_HGavg_X
        rhet_role = self.rhet_X
        wordlist = self.wordlist_X
        pasttense = self.pasttense_X
        rhet_y = self.rhet_y
        rel_y = self.rel_y

        # TO DO - change this so it uses the final feature set,
        print("THIS IS TO RANK THE SENTENCES - PLEASE SELECT FINAL FEATURE SET")
        import mode_selector
        mode = mode_selector.mode_selector(location, HGlocation, quotation, entities, asmo,
        cue_phrase, sent_length, HGsent_length, tfidf_max, tfidf_top20, tfidf_HGavg, rhet_role,
        wordlist, pasttense, rhet_y, rel_y)
        num_of_features = input("how many features? ")
        X, feat_names = mode.select_features(num_of_features)
        print("NB - please select RELEVANCE to get the correct sentence ranking")
        Y, label, target_names = mode.select_target()

        self.supervised_ml(X, Y, label, feat_names, target_names, mode)

pipeline = ranking_ml()
pipeline.prep_data('MLdata')
# pipeline.prep_data('MLdata_train')
pipeline.exec()
