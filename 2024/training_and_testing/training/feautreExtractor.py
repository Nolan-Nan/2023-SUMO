
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature extractor for a new case to summarise.

@author: amyconroy
"""
import csv
import numpy as np
import math

class featureExtractor():
    def __init__ (self, casenum):
        #List of features
        ##for asmo feature-set
        self.agree_X = np.array([])
        self.outcome_X = np.array([])
        ##for location feature-set
        self.loc1_X = np.array([]); self.loc2_X = np.array([]); self.loc3_X = np.array([])
        self.loc4_X = np.array([]); self.loc5_X = np.array([]); self.loc6_X = np.array([])
        self.sentlen_X = np.array([])
        self.rhet_X = np.array([])
        self.wordlist_X = np.array([])
        self.pasttense_X = np.array([])

        self.tfidf_max_X = np.array([])
        self.tfidf_top20_X = np.array([])

        #Hachey and Grover's original features
        ##for location feature-set
        self.tfidf_HGavg_X = np.array([])
        self.HGsentlen_X = np.array([])
        self.qb_X = np.array([])
        self.inq_X = np.array([])

        # for updated entity feature-set
        self.citationent_X = np.array([])
        self.casenameent_X = np.array([])

        # using blackstones feature-set
        self.provision_ent_X = np.array([])
        self.instrument_ent_X = np.array([])
        self.court_ent_X = np.array([])
        self.casename_ent_X = np.array([])
        self.citation_ent_X = np.array([])
        self.judge_ent_X = np.array([])

        # updated NER from spacy
        self.loc_ent_X = np.array([])
        self.org_ent_X = np.array([])
        self.date_ent_X = np.array([])
        self.person_ent_X = np.array([])

        ##for cue phrase feature-set
        self.asp_X = np.array([])
        self.modal_X = np.array([])
        self.voice_X = np.array([])
        self.negcue_X = np.array([])
        self.tense_X = np.array([])

        #for storing Xs values
        self.case_flag = []
        self.sent_flag = []

        # for seq modelling
        self.judgename = []
        self.rhetlabel = []

        # actual code
        print("Fetching entities")
        self.getEntities(casenum)
        print("Entities fetched")
        print("Creating features")
        self.createFeatures(casenum)
        print("Features created")
        print("Outputting data")
        self.outputData(casenum)
        print("Data outputed")

    def get_end_par_in_lord(self, judge, case, current_paragraph):
        filename = 'UKHL_' + case + '.csv'
        if case == 'N/A':
            filename = 'UKHL_' + 'NA' + '.csv'
        with open('./data/UKHL_corpus2/' + filename, 'r') as infile:
                reader = csv.DictReader(infile)


                ret = ''
                for row in reader:
                    if row['judge'] == judge:
                        if row['para_id'] == '0.5':
                            rowpar = '0'
                        elif '.5' in row['para_id']:
                            rowpar = row['para_id'].replace('.5', '')
                        else:
                            rowpar = row['para_id']
                        if int(rowpar) >= int(current_paragraph):
                            ret = rowpar

                return ret

    def get_end_sent_in_lord(self, judge, case, current_sentence):
        filename = 'UKHL_' + case + '.csv'
        if case == 'N/A':
           filename = 'UKHL_' + 'NA' + '.csv'
        with open('./data/UKHL_corpus2/' + filename, 'r') as infile:
               reader = csv.DictReader(infile)

               ret = ''
               for row in reader:
                   if row['agree'] == 'no match' or row['role'] == '<prep-date>'\
                   or row['role'] == '<sub-heading>' or row['role'] == '<separator>' or row['role'] == '<new-case>':
                          continue
                   if row['judge'] == judge:
                       if int(row['sentence_id']) >= int(current_sentence):
                           ret = row['sentence_id']
               return ret

    def get_end_sent_in_par(self, par, case, current_sentence):
        filename = 'UKHL_' + case + '.csv'
        if case == 'N/A':
            filename = 'UKHL_' + 'NA' + '.csv'
        with open('./data/UKHL_corpus2/' + filename, 'r') as infile:
            reader = csv.DictReader(infile)

            ret = ''
            for row in reader:
                if row['agree'] == 'no match' or row['role'] == '<prep-date>'\
                or row['role'] == '<sub-heading>' or row['role'] == '<separator>' or row['role'] == '<new-case>':
                    continue

                if row['para_id'] == '0.5':
                    rowpar = '0'
                elif '.5' in row['para_id']:
                    rowpar = row['para_id'].replace('.5', '')
                else:
                    rowpar = row['para_id']
                if rowpar == par:
                    if int(row['sentence_id']) >= int(current_sentence):
                        ret = row['sentence_id']

            return ret

    def get_wordlist(self, text):
        with open('./data/wordlist.csv', 'r') as infile:
            reader = csv.DictReader(infile)
            score = 0

            for row in reader:
                if row['0'] in text and row['0'] != '':
                    pass
                if row['1'] in text and row['1'] != '':
                    score += 1
                if row['2'] in text and row['2'] != '':
                    score += 2
                if row['3'] in text and row['3'] != '':
                    score += 3
                if row['4'] in text and row['4'] != '':
                    score += 4

            return math.log(score+1, 16)

    def getEntities(self, casenum):
        entitiesFile = 'summarydata/UKHL_'+casenum+'.csv'

        with open(entitiesFile, 'r') as infile:
            reader = csv.DictReader(infile)

            for row in reader:
                self.provision_ent_X = np.append(self.provision_ent_X, row['provision ent'])
                self.court_ent_X = np.append(self.court_ent_X, row['court ent'])
                self.casename_ent_X = np.append(self.casename_ent_X, row['case name ent'])
                self.judge_ent_X = np.append(self.judge_ent_X, row['judge ent'])
                self.instrument_ent_X = np.append(self.instrument_ent_X, row['instrument ent'])
                self.citation_ent_X = np.append(self.citation_ent_X, row['citation bl ent'])
                self.loc_ent_X = np.append(self.loc_ent_X, row['loc ent'])
                self.org_ent_X = np.append(self.org_ent_X, row['org ent'])
                self.date_ent_X = np.append(self.date_ent_X, row['date ent'])
                self.person_ent_X =np.append(self.person_ent_X, row['person ent'])

    def createFeatures(self, casenum):
        caseFile = 'data/UKHL_corpus/UKHL_'+casenum+'.csv'

        judge = ''
        case = ''
        par = ''
        loc1 = 0; loc2 = 0; loc3 = 0; loc4 = 0; loc5 = 0; loc6 = 0
        import tfidf_feature
        tfidf = tfidf_feature.tfidf_calc()
        #for quotations
        qb_bool = False
        quoteblock = 0
        inquotes = False
        word_inq = 0
        with open(caseFile, 'r') as infile:
          reader = csv.DictReader(infile)
          for row in reader:
                if row['agree'] == 'no match' or row['role'] == '<prep-date>'\
                or row['role'] == '<sub-heading>' or row['role'] == '<separator>' or row['role'] == '<new-case>':
                    continue


                self.case_flag.append(casenum)
                self.sent_flag.append(row['sentence_id'])
                self.judgename.append(row['judge'])

                import nvGroups
                asp, modal, voice, negation, tense, pasttense = nvGroups.get_verb_features(casenum, row['sentence_id'])
                self.pasttense_X = np.append(self.pasttense_X, [pasttense])
                if asp == 'SIMPLE':
                    self.asp_X = np.append(self.asp_X, [1])
                elif asp == 'PERF':
                    self.asp_X = np.append(self.asp_X, [2/3])
                elif asp == 'PROG':
                    self.asp_X = np.append(self.asp_X, [1/3])
                else:
                    self.asp_X = np.append(self.asp_X, [0])
                if modal == 'NO':
                    self.modal_X = np.append(self.modal_X, [1])
                elif modal == 'YES':
                    self.modal_X = np.append(self.modal_X, [1/2])
                else:
                    self.modal_X = np.append(self.modal_X, [0])
                if voice == 'ACT':
                    self.voice_X = np.append(self.voice_X, [1])
                elif voice == 'PASS':
                    self.voice_X = np.append(self.voice_X, [1/2])
                else:
                    self.voice_X = np.append(self.voice_X, [0])
                if negation == 'yes':
                    self.negcue_X = np.append(self.negcue_X, [1])
                else:
                    self.negcue_X = np.append(self.negcue_X, [0])
                if tense == 'PRES':
                    self.tense_X = np.append(self.tense_X, [1])
                elif tense == 'PRESorBASE':
                    self.tense_X = np.append(self.tense_X, [3/4])
                elif tense == 'PAST':
                    self.tense_X = np.append(self.tense_X, [2/4])
                elif tense == 'INF':
                    self.tense_X = np.append(self.tense_X, [1/4])
                else:
                    self.tense_X = np.append(self.tense_X, [0])

                # ENTITIES FEATURE SET

                tfidf.get_doc(casenum)
                sent_max_tfidf, sent_intop20_tfidf, sent_avg_tfidf = tfidf.get_sent_features(row['text'])
                self.tfidf_max_X = np.append(self.tfidf_max_X, [sent_max_tfidf])
                self.tfidf_top20_X = np.append(self.tfidf_top20_X, [sent_intop20_tfidf])
                self.tfidf_HGavg_X = np.append(self.tfidf_HGavg_X, [sent_avg_tfidf])

                wordlist = self.get_wordlist(row['text'])
                self.wordlist_X = np.append(self.wordlist_X, [wordlist])

                tmptxt = row['text'].split()
                if '.5' in row['para_id']:
                    if tmptxt[0] == '"':
                        quoteblock = 1
                        qb_bool = True
                    if qb_bool == True:
                        quoteblock = 1
                    if tmptxt[-1] == '"' and qb_bool == True or '" .' in row['text'] and qb_bool == True:
                        quoteblock = 1
                        qb_bool = False
                else:
                    quoteblock = 0
                    qb_bool = False
                self.qb_X = np.append(self.qb_X, [quoteblock])

                if tmptxt[0] == '"':
                    tmptxt[0] = ''
                    if tmptxt[-1] == '"':
                        tmptxt[-1] == ''
                    if '" .' in row['text']:
                        tmptxt[:-3]
                word_inq = 0
                for v in range(len(tmptxt)):
                    if tmptxt[v] == '"' and inquotes == False:
                        inquotes = True
                    elif tmptxt[v] == '"' and inquotes == True:
                        inquotes = False
                    if inquotes == True and tmptxt[v] != '"':
                        word_inq += 1
                if word_inq == 0:
                    inq = 0
                else:
                    inq = word_inq / len(row['text'])
                self.inq_X = np.append(self.inq_X, [inq])

                sent_len = len(row['text'].split())
                self.HGsentlen_X = np.append(self.HGsentlen_X, [sent_len])
                sent_len = math.log(sent_len,625)
                self.sentlen_X = np.append(self.sentlen_X, [sent_len])

                current_case = casenum
                current_judge = row['judge']
                current_paragraph = row['para_id']
                if current_paragraph == '0.5':
                    current_paragraph = '0'
                elif '.5' in current_paragraph:
                    current_paragraph = current_paragraph.replace('.5', '')
                current_sentence = row['sentence_id']


                if case != current_case:
                    case = current_case
                    start_lord_par = 0 #for every new lord, get current paragraph
                    start_lord_sent = 0 #for every new lord, get current sentence
                    start_par_sent = 0 #for every new paragraph, get current sentence
                    end_lord_paragraphs = 0 #total paragraphs in a lord
                    end_lord_sentences = 0 #total sentences in a lord
                    end_par_sentences = 0 #total sentences in a paragraph

                if judge != current_judge:
                    judge = current_judge
                    start_lord_par = int(current_paragraph)
                    start_lord_sent = int(current_sentence)
                    end_lord_paragraphs = int(self.get_end_par_in_lord(judge, case, current_paragraph))
                    end_lord_sentences = int(self.get_end_sent_in_lord(judge, case, current_sentence))

                if par != current_paragraph:
                    par = current_paragraph
                    start_par_sent = int(current_sentence)
                    end_par_sentences = int(self.get_end_sent_in_par(par, case, current_sentence))

                loc1 = 1 + int(current_paragraph) - start_lord_par
                if loc1 <= 0:
                    loc1 = 0
                else:
                    loc1 = math.log(loc1, 81)
                loc2 = 1 + end_lord_paragraphs - int(current_paragraph)
                if loc2 <= 0:
                    loc2 = 0
                else:
                    loc2 = math.log(loc2, 81)
                loc3 = 1 + int(current_sentence) - start_lord_sent
                if loc3 <= 0:
                    loc3 = 0
                else:
                    loc3 = math.log(loc3, 625)
                loc4 = 1 + end_lord_sentences - int(current_sentence)
                if loc4 <= 0:
                    loc4 = 0
                else:
                    loc4 = math.log(loc4, 625)
                loc5 = 1 + int(current_sentence) - start_par_sent
                if loc5 <= 0:
                    loc5 = 0
                else:
                    loc5 = math.log(loc5, 16)
                loc6 = 1 + end_par_sentences - int(current_sentence)
                if loc6 <= 0:
                    loc6 = 0
                else:
                    loc6 = math.log(loc6, 16)

                self.loc1_X = np.append(self.loc1_X, [loc1]); self.loc2_X = np.append(self.loc2_X, [loc2]); self.loc3_X = np.append(self.loc3_X, [loc3])
                self.loc4_X = np.append(self.loc4_X, [loc4]); self.loc5_X = np.append(self.loc5_X, [loc5]); self.loc6_X = np.append(self.loc6_X, [loc6])


                if row['agree'] == 'NONE':
                    if row['ackn'] != 'NONE':
                        self.agree_X = np.append(self.agree_X, [0.5])
                    else:
                        self.agree_X = np.append(self.agree_X, [0])
                else:
                    self.agree_X = np.append(self.agree_X, [1])
                if row['outcome'] == 'NONE':
                    self.outcome_X = np.append(self.outcome_X, [0])
                else:
                    self.outcome_X = np.append(self.outcome_X, [1])

    def outputData(self, casenum):
        with open('summarydata/UKHL_'+casenum+'_features.csv', 'w', newline='') as outfile:
            fieldnames = ['case_id', 'sent_id', 'align', 'agree', 'outcome', 'loc1', 'loc2', 'loc3',
            'loc4', 'loc5', 'loc6', 'sentlen',
            'HGsentlen', 'quoteblock', 'inline_q', 'tfidf_max', 'tfidf_top20', 'tfidf_HGavg', 'aspect', 'modal',
            'voice', 'negation', 'tense', 'wordlist', 'past tense',
            'provision ent', 'instrument ent', 'court ent', 'case name ent', 'citation bl ent', 'judge ent',
            'loc ent', 'org ent', 'date ent', 'person ent','judgename']

            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for v in range(len(self.sent_flag)):
                writer.writerow({'case_id': self.case_flag[v], 'sent_id': self.sent_flag[v], 'agree': self.agree_X[v],
                'outcome': self.outcome_X[v], 'loc1': self.loc1_X[v], 'loc2': self.loc2_X[v], 'loc3': self.loc3_X[v], 'loc4': self.loc4_X[v],
                'loc5': self.loc5_X[v], 'loc6': self.loc6_X[v], 'sentlen': self.sentlen_X[v], 'HGsentlen': self.HGsentlen_X[v], 'quoteblock': self.qb_X[v], 'inline_q': self.inq_X[v],
                 'tfidf_max': self.tfidf_max_X[v], 'tfidf_top20': self.tfidf_top20_X[v], 'tfidf_HGavg': self.tfidf_HGavg_X[v],'aspect': self.asp_X[v],
                'modal': self.modal_X[v], 'voice': self.voice_X[v], 'negation': self.negcue_X[v], 'tense': self.tense_X[v],  'wordlist': self.wordlist_X[v],
                'past tense': self.pasttense_X[v], 'provision ent' : self.provision_ent_X[v], 'instrument ent' : self.instrument_ent_X[v], 'court ent' : self.court_ent_X[v],
                'case name ent' : self.casename_ent_X[v], 'citation bl ent' : self.citation_ent_X[v], 'judge ent' : self.judge_ent_X[v], 'loc ent' : self.loc_ent_X[v], 'org ent' : self.org_ent_X[v], 'date ent' :
                self.date_ent_X[v], 'person ent' : self.person_ent_X[v], 'judgename' : self.judgename[v]})
