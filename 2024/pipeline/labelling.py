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
        spacyNLP = spacy.load("en_core_web_sm")
    
        sentid_X = np.array([])

        #SPACY ENTITIES
        loc_ent_X = np.array([])
        org_ent_X = np.array([]) 
        date_ent_X = np.array([])
        person_ent_X = np.array([])
        fac_ent_X = np.array([])
        norp_ent_X = np.array([])
        gpe_ent_X = np.array([])
        event_ent_X = np.array([])
        law_ent_X = np.array([])
        time_ent_X = np.array([])
        work_of_art_ent_X = np.array([])
        ordinal_ent_X = np.array([])
        cardinal_ent_X = np.array([])
        money_ent_X = np.array([])
        percent_ent_X = np.array([])
        product_ent_X = np.array([])
        quantity_ent_X = np.array([])


        y = 0
    
        # Iterate through the entities identified by the model
        if casenum.startswith("UK"):
            path = 'data/UKHL_corpus/'+casenum+'.csv'
        else:
            path = 'data/UKHL_corpus/UKHL_'+casenum+'.csv'
        with open(path, 'r', encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                if row['agree'] == 'no match' or row['role'] == '<prep-date>'\
                or row['role'] == '<sub-heading>' or row['role'] == '<separator>' or row['role'] == '<new-case>':
                        continue
                y += 1
                sentid_X = np.append(sentid_X, row['sentence_id'])
                    
                text = row['text']
                doc = spacyNLP(text)
                label = [(ent.label_) for ent in doc.ents]
            
                loc_flag = False
                org_flag = False
                date_flag = False
                person_flag = False
                fac_flag = False
                norp_flag = False
                gpe_flag = False
                event_flag = False
                law_flag = False
                time_flag = False
                work_of_art_flag = False
                ordinal_flag = False
                cardinal_flag = False
                money_flag = False
                percent_flag = False
                product_flag = False
                quantity_flag = False
        
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
                    if lbl == 'FAC':
                        fac_flag = True
                    if lbl == 'NORP':
                        norp_flag = True
                    if lbl == 'GPE':
                        gpe_flag = True
                    if lbl == 'EVENT':
                        event_flag = True
                    if lbl == 'LAW':
                        law_flag = True
                    if lbl == 'TIME':
                        time_flag = True
                    if lbl == 'WORK_OF_ART':
                        work_of_art_flag = True
                    if lbl == 'ORDINAL':
                        ordinal_flag = True
                    if lbl == 'CARDINAL':
                        cardinal_flag = True
                    if lbl == 'MONEY':
                        money_flag = True
                    if lbl == 'PERCENT':
                        percent_flag = True
                    if lbl == 'PRODUCT':
                        product_flag = True
                    if lbl == 'QUANTITY':
                        quantity_flag = True
          
                        
                if person_flag:
                    person_ent_X = np.append(person_ent_X, [1])
                else:
                    person_ent_X = np.append(person_ent_X, [0])
                
                if date_flag:
                    date_ent_X = np.append(date_ent_X, [1])
                else:
                    date_ent_X = np.append(date_ent_X, [0])
                  
                if org_flag:
                    org_ent_X = np.append(org_ent_X, [1])
                else:
                    org_ent_X = np.append(org_ent_X, [0])

                if loc_flag: 
                    loc_ent_X = np.append(loc_ent_X, [1])
                else: 
                    loc_ent_X = np.append(loc_ent_X, [0])

                if fac_flag:
                    fac_ent_X = np.append(fac_ent_X,[1])
                else:
                    fac_ent_X = np.append(fac_ent_X,[0])
                if norp_flag:
                    norp_ent_X = np.append(norp_ent_X,[1])
                else:
                    norp_ent_X = np.append(norp_ent_X,[0])
                if gpe_flag:
                    gpe_ent_X= np.append(gpe_ent_X,[1])
                else:
                    gpe_ent_X= np.append(gpe_ent_X,[0])
                if event_flag:
                    event_ent_X= np.append(event_ent_X,[1])
                else:
                    event_ent_X= np.append(event_ent_X,[0])
                if law_flag:
                    law_ent_X= np.append(law_ent_X,[1])
                else:
                    law_ent_X= np.append(law_ent_X,[0])
                if time_flag:
                    time_ent_X= np.append(time_ent_X,[1])
                else:
                    time_ent_X= np.append(time_ent_X,[0])
                if work_of_art_flag:
                    work_of_art_ent_X= np.append(work_of_art_ent_X,[1])
                else:
                    work_of_art_ent_X= np.append(work_of_art_ent_X,[0])
                if ordinal_flag:
                    ordinal_ent_X= np.append(ordinal_ent_X,[1])
                else:
                    ordinal_ent_X= np.append(ordinal_ent_X,[0])
                if cardinal_flag:
                    cardinal_ent_X= np.append(cardinal_ent_X,[1])
                else:
                    cardinal_ent_X= np.append(cardinal_ent_X,[0])
                if money_flag:
                    money_ent_X= np.append(money_ent_X,[1])
                else:
                    money_ent_X= np.append(money_ent_X,[0])
                if percent_flag:
                    percent_ent_X= np.append(percent_ent_X,[1])
                else:
                    percent_ent_X= np.append(percent_ent_X,[0])
                if product_flag:
                    product_ent_X= np.append(product_ent_X,[1])
                else:
                    product_ent_X= np.append(product_ent_X,[0])
                if quantity_flag:
                    quantity_ent_X= np.append(quantity_ent_X,[1])
                else:
                    quantity_ent_X= np.append(quantity_ent_X,[0])

        directory = './summarydata-spacy/'
        os.makedirs(directory, exist_ok=True)
            
        with open('./summarydata-spacy/UKHL_'+casenum+'.csv','w', newline='')as outfile:
            fieldnames = ['sent id','loc ent', 'org ent', 'date ent', 'person ent',
                          'fac_ent','norp_ent','gpe_ent','event_ent', 'law_ent', 'time_ent',
                          'work_of_art_ent','ordinal_ent','cardinal_ent','money_ent','percent_ent',
                          'product_ent','quantity_ent']

            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
        

            for v in range(y):
                writer.writerow({'sent id' : sentid_X[v],'loc ent' : loc_ent_X[v], 'org ent' : org_ent_X[v], 'date ent' : date_ent_X[v],
                                 'person ent' : person_ent_X[v] , 'fac_ent' : fac_ent_X[v],'norp_ent' : norp_ent_X[v],'gpe_ent' : gpe_ent_X[v],
                                 'event_ent' : event_ent_X[v], 'law_ent' : law_ent_X[v], 'time_ent' : time_ent_X[v],
                                 'work_of_art_ent' : work_of_art_ent_X[v],'ordinal_ent' : ordinal_ent_X[v],'cardinal_ent' : cardinal_ent_X[v],
                                 'money_ent' : money_ent_X[v],'percent_ent' : percent_ent_X[v], 'product_ent' : product_ent_X[v],'quantity_ent' : quantity_ent_X[v]})
        
        filename = './summarydata-spacy/UKHL_'+casenum+'.csv'
        
        return filename
         
             
                        
             
                     
   


          
                     