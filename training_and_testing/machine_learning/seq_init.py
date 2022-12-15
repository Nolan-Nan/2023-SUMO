
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this goes through the corpus and creates lists of the speeches,
sentences, and rhetorical tags - relevant features are received from the featureExtraction.py()

@author: amyconroy
"""

import csv
# import pickle


''' this function creates a list of each speech in the corpus
tagged (numbered) '''
def create_tagged_speeches_list():
        with open('./data/UKHL_corpus.csv', 'r') as infile:
            reader = csv.DictReader(infile)

            # init
            speechindex = 0
            speeches_list = []
            previous_judgename = ''

            for row in reader:

                judgename = row['judge']
                # not a previous judge
                if judgename != 'NONE' and judgename != previous_judgename:
                        speechindex += 1
                        speech = {
                            'case' : row['case_id'],
                            'judge' : judgename,
                            'index' : speechindex
                        }
                        print(speech)
                        speeches_list.append(speech)

                previous_judgename = judgename
        return speeches_list


''' this function creates a list of each tagged sentence
it's sentence_id number, case_num, and speech num (generated
in line with create_tagged_speechs_list())

need to create the list with appropriate sentence number
included to exctract the context from the MLdata file'''
def create_tagged_sentences_list():
        print("hello? sent")
        with open('./data/UKHL_corpus.csv', 'r') as infile:
            reader = csv.DictReader(infile)

            # init
            sentence_list = []

            for row in reader:
                sentence_id = row['sentence_id']
                judgename = row['judge']

                if sentence_id != 'N/A' and judgename != 'NONE':
                        case = row['case_id']
                        role = row['role']

                        sentence = {
                            'sentence_id' : sentence_id,
                            'role' : role,
                            'case' : case,
                            'judge' : judgename
                            }
                        print(sentence)
                        sentence_list.append(sentence)

        return sentence_list



"""
    def seq_start(self):
        sentence_list = []
        speeches_list = []

        speeches_list = self.create_tagged_speeches_list()
        sentence_list = self.create_tagged_sentences_list()

        speeches_file = "./seq/speeches"
        sentence_file = "./seq/sentences"

        self.save_to_file(speeches_list, speeches_file)
        self.save_to_file(sentence_list, sentence_file)


data = seq_init()
data.seq_start()     """
