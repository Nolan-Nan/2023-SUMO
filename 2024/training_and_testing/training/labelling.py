
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
to label entities- this will create a new folder called /case and put the case in there
correctly labelled

NB - need spaCy version 2.1.8 to work for blackstone, backdate en-core-web-sm version
running python -m spacy validation downgrades the en-core-web-sm version to the appropriate version
down graded spacy to spacy==2.1.8

@author: amyconroy
"""
import csv
import numpy as np


class labelling():
    def __init__(self, casenum):
        print("NER Labelling")
        filename = self.NER(casenum)
        print("Finished labelling")

    def NER(self, casenum):
        import spacy
        # Load the model
        blackstoneNLP = spacy.load("en_blackstone_proto")
        spacyNLP = spacy.load("en_core_web_sm")

        sentid_X = np.array([])
        # BLACKSTONE ENTITIES
        provision_ent_X = np.array([])
        instrument_ent_X = np.array([])
        court_ent_X = np.array([])
        casename_ent_X = np.array([])
        citation_ent_X = np.array([])
        judge_ent_X = np.array([])
        #SPACY ENTITIES
        loc_ent_X = np.array([])
        org_ent_X = np.array([])
        date_ent_X = np.array([])
        person_ent_X = np.array([])

        y = 0

        # Iterate through the entities identified by the model
        with open('data/UKHL_corpus/UKHL_'+casenum+'.csv', 'r') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                if row['agree'] == 'no match' or row['role'] == '<prep-date>'\
                or row['role'] == '<sub-heading>' or row['role'] == '<separator>' or row['role'] == '<new-case>':
                        continue
                y += 1
                sentid_X = np.append(sentid_X, row['sentence_id'])
                text = row['text']
                doc = blackstoneNLP(text)
                label = [(ent.label_) for ent in doc.ents]

                provision_flag = False
                court_flag = False
                case_flag = False
                judge_flag = False
                instrument_flag = False
                citation_flag = False

                for v in range(len(label)):
                    lbl = label[v]
                    if lbl == 'PROVISION':
                        provision_flag = True
                    if lbl == 'COURT':
                        court_flag = True
                    if lbl == 'CASENAME':
                        case_flag = True
                    if lbl == 'JUDGE':
                        judge_flag = True
                    if lbl == 'INSTRUMENT':
                        instrument_flag = True
                    if lbl == 'CITATION':
                        citation_flag = True

                if provision_flag:
                    provision_ent_X = np.append(provision_ent_X, [1])
                else:
                    provision_ent_X = np.append(provision_ent_X, [0])
                if court_flag:
                    court_ent_X = np.append(court_ent_X, [1])
                else:
                   court_ent_X = np.append(court_ent_X, [0])
                if case_flag:
                    casename_ent_X = np.append(casename_ent_X, [1])
                else:
                    casename_ent_X = np.append(casename_ent_X, [0])
                if judge_flag:
                    judge_ent_X = np.append(judge_ent_X, [1])
                else:
                    judge_ent_X = np.append(judge_ent_X, [0])
                if instrument_flag:
                    instrument_ent_X = np.append(instrument_ent_X, [1])
                else:
                    instrument_ent_X = np.append(instrument_ent_X, [0])
                if citation_flag:
                    citation_ent_X = np.append(citation_ent_X, [1])
                else:
                    citation_ent_X = np.append(citation_ent_X, [0])

                text = row['text']
                doc = spacyNLP(text)
                label = [(ent.label_) for ent in doc.ents]

                loc_flag = False
                org_flag = False
                date_flag = False
                person_flag = False

                for v in range(len(label)):
                    lbl = label[v]
                    if lbl == 'LOC':
                        loc_flag = True
                    if lbl == 'ORG':
                        org_flag = True
                    if lbl == 'DATE':
                        date_flag = True
                    if lbl == 'PERSON':
                        person_flag = True


                if case_flag == False and judge_flag == False:
                    if person_flag:
                        person_ent_X = np.append(person_ent_X, [1])
                    else:
                        person_ent_X = np.append(person_ent_X, [0])
                else:
                    person_ent_X = np.append(person_ent_X, [0])

                if citation_flag == False:
                    if date_flag:
                         date_ent_X = np.append(date_ent_X, [1])
                    else:
                        date_ent_X = np.append(date_ent_X, [0])
                else:
                    date_ent_X = np.append(date_ent_X, [0])

                if court_flag == False:
                    if org_flag:
                        org_ent_X = np.append(org_ent_X, [1])
                    else:
                        org_ent_X = np.append(org_ent_X, [0])
                else:
                    org_ent_X = np.append(org_ent_X, [0])
                if loc_flag:
                    loc_ent_X = np.append(loc_ent_X, [1])
                else:
                    loc_ent_X = np.append(loc_ent_X, [0])







        with open('summarydata/UKHL_'+casenum+'.csv','w', newline='')as outfile:
            fieldnames = ['sent id', 'provision ent', 'instrument ent', 'court ent', 'case name ent', 'citation bl ent', 'judge ent',
                          'loc ent', 'org ent', 'date ent', 'person ent']

            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()


            for v in range(y):
                writer.writerow({'sent id' : sentid_X[v], 'provision ent' : provision_ent_X[v], 'instrument ent' : instrument_ent_X[v], 'court ent' : court_ent_X[v],
                                 'case name ent' : casename_ent_X[v], 'citation bl ent' : citation_ent_X[v], 'judge ent' : judge_ent_X[v],
                                 'loc ent' : loc_ent_X[v], 'org ent' : org_ent_X[v], 'date ent' : date_ent_X[v],
                                 'person ent' : person_ent_X[v]})

        filename = 'summarydata/UKHL_'+casenum+'.csv'

        return filename
