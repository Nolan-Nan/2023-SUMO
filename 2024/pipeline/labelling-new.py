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
import os

import numpy as np


class labelling():
    def __init__(self, casenum):
        print("NER Labelling")
        filename = self.NER(casenum)
        print("Finished labelling")
    
    def NER(self, casenum):
        import spacy
        # Load the model
        model_path = '../NER/model-best'
        nlp_ner = spacy.load(model_path)
        spacyNLP = spacy.load("en_core_web_sm")
    
        sentid_X = np.array([])
        # BLACKSTONE ENTITIES
        COURT = np.array([])
        PETITIONER = np.array([])
        RESPONDENT = np.array([])
        JUDGE = np.array([])
        LAWYER = np.array([])
        DATE = np.array([])
        ORG = np.array([])
        GPE = np.array([])
        STATUTE = np.array([])
        PROVISION = np.array([])
        PRECEDENT = np.array([])
        CASE_NUMBER = np.array([])
        WITNESS = np.array([])
        OTHER_PERSON = np.array([])
        #SPACY ENTITIES
        loc_ent_X = np.array([])
        org_ent_X = np.array([]) 
        date_ent_X = np.array([])
        person_ent_X = np.array([])

        y = 0
        flags = ["COURT", "PETITIONER", "RESPONDENT", "JUDGE", "LAWYER", "DATE", "GPE", "STATUTE",
                 "PROVISION", "PRECEDENT", "CASE_NUMBER", "WITNESS", "OTHER_PERSON"]
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
                doc = nlp_ner(text)
                label = [(ent.label_) for ent in doc.ents]
                print(label)
                court = False
                petitioner = False
                respondent = False
                judge = False
                lawyer = False
                date = False
                org = False
                gpe = False
                statute = False
                provision = False
                precedent = False
                case_number = False
                witness = False
                other_person = False

                for v in range(len(label)):
                    lbl = label[v]
                    if lbl == flags[0]:
                        court = True
                    if lbl == flags[1]:
                        petitioner = True
                    if lbl == flags[2]:
                        respondent = True
                    if lbl == flags[3]:
                        judge = True
                    if lbl == flags[4]:
                        lawyer = True
                    if lbl == flags[5]:
                        date = True
                    if lbl == flags[6]:
                        org = True
                    if lbl == flags[7]:
                        gpe = True
                    if lbl == flags[8]:
                        statute = True
                    if lbl == flags[9]:
                        provision = True
                    if lbl == flags[10]:
                        precedent = True
                    if lbl == flags[11]:
                        case_number = True
                    if lbl == flags[12]:
                        witness = True
                    if lbl == flags[13]:
                        other_person = True

                if court:
                    COURT = np.append(COURT,[1])
                else:
                    COURT = np.append(COURT,[0])
                if petitioner:
                    PETITIONER = np.append(PETITIONER,[1])
                else:
                    PETITIONER = np.append(PETITIONER,[0])
                if respondent:
                    RESPONDENT= np.append(RESPONDENT,[1])
                else:
                    RESPONDENT= np.append(RESPONDENT,[0])
                if judge:
                    JUDGE= np.append(JUDGE,[1])
                else:
                    JUDGE= np.append(JUDGE,[0])
                if lawyer:
                    LAWYER= np.append(LAWYER,[1])
                else:
                    LAWYER= np.append(LAWYER,[0])
                if date:
                    DATE= np.append(DATE,[1])
                else:
                    DATE= np.append(DATE,[0])
                if org:
                    ORG= np.append(ORG,[1])
                else:
                    ORG= np.append(ORG,[0])
                if gpe:
                    GPE= np.append(GPE,[1])
                else:
                    GPE= np.append(GPE,[0])
                if statute:
                    STATUTE= np.append(STATUTE,[1])
                else:
                    STATUTE= np.append(STATUTE,[0])
                if provision:
                    PROVISION= np.append(PROVISION,[1])
                else:
                    PROVISION= np.append(PROVISION,[0])
                if precedent:
                    PRECEDENT= np.append(PRECEDENT,[1])
                else:
                    PRECEDENT= np.append(PRECEDENT,[0])
                if case_number:
                    CASE_NUMBER= np.append(CASE_NUMBER,[1])
                else:
                    CASE_NUMBER= np.append(CASE_NUMBER,[0])
                if witness:
                    WITNESS= np.append(WITNESS,[1])
                else:
                    WITNESS= np.append(WITNESS,[0])
                if other_person:
                    OTHER_PERSON= np.append(OTHER_PERSON,[1])
                else:
                    OTHER_PERSON= np.append(OTHER_PERSON,[0])

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
          
                        
                if case_number == False and judge == False:
                    if person_flag: 
                        person_ent_X = np.append(person_ent_X, [1])
                    else: 
                        person_ent_X = np.append(person_ent_X, [0])
                else:
                    person_ent_X = np.append(person_ent_X, [0]) 
                
                if precedent == False:
                    if date_flag: 
                         date_ent_X = np.append(date_ent_X, [1])
                    else: 
                        date_ent_X = np.append(date_ent_X, [0])
                else: 
                    date_ent_X = np.append(date_ent_X, [0])  
                  
                if court == False:
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

        # Create the directory if it does not exist
        directory = './summarydata-withoutbs/'
        os.makedirs(directory, exist_ok=True)
            
        with open('./summarydata-withoutbs/UKHL_'+casenum+'-wtb.csv','w', newline='')as outfile:
            fieldnames = ['sent id','court', 'petitioner', 'respondent', 'judge', 'lawyer', 'date', 'org', 'gpe', 'statute', 'provision', 'precedent', 'case_number', 'witness', 'other_person',
                          'loc ent', 'org ent', 'date ent', 'person ent']        

            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
        

            for v in range(y):
                writer.writerow({'sent id' : sentid_X[v], 'court': COURT[v],'petitioner': PETITIONER[v],'respondent': RESPONDENT[v],
                                 'judge': JUDGE[v],'lawyer': LAWYER[v],'date': DATE[v],'org': ORG[v],'gpe': GPE[v],
                                 'statute': STATUTE[v],'provision': PROVISION[v],'precedent': PRECEDENT[v],'case_number': CASE_NUMBER[v],
                                 'witness': WITNESS[v],'other_person': OTHER_PERSON[v],
                                 'loc ent' : loc_ent_X[v], 'org ent' : org_ent_X[v], 'date ent' : date_ent_X[v], 'person ent' : person_ent_X[v]})
        
        filename = './summarydata-withoutbs/UKHL_'+casenum+'-wtb.csv'
        
        return filename
         
             
                        
             
                     
   


          
                     