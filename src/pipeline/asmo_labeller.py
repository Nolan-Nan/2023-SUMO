#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 21:18:04 2022

@author: amyconroy
"""


    def parse_offline(self, annotator, file):
        all = []
        with open("corpus/annodata/" + annotator + "/" + file, 'r') as f:
            lines = [i.strip("\n") for i in f.readlines()]

            mj = self.ext_mj(lines)
            ref_sent = self.ext_ref(lines)
            line_num, body, max_line = self.ext_sent(file)
            case_name = int(file.strip(".txt"))

            for line_num, body in zip(line_num, body):
                try: # Each annotation must have from, to and relation filled in
                    ref = ref_sent[line_num]
                    ref_from = ref["from"]
                    ref_to = ref["to"]
                    relation = ref["rel"]
                    position = round(line_num/max_line, 1)
                    for r_f, r_t, rel in zip(ref_from, ref_to, relation):
                        # print(annotator, case_name, line_num, body, r_f, r_t, rel, position, mj)
                        all.append([annotator, case_name, line_num, body, r_f, r_t, rel, position, mj])

                except: # If there is no annotation, fills the blanks
                    ref_from, ref_to, relation = "NAN", "NAN", "NAN"
                    pos = round(line_num/max_line, 1)
                    all.append([annotator, case_name, line_num, body, ref_from, ref_to, relation, pos, mj])

            return all
        
        
        use the above to create the parser, pass in the file -> get this data 
        
        
    def get_prediction(self, MJ_corpus):
        if self.train:
            classifier = self.best_classifier()
#            save_data("classifier", classifier) #save
        else:
            classifier = load_data("classifier") #NOTE write test
            
            this will need to be added in with the rest of the info from the classifier - I think you will just
            need to remove the stuff that trains, only allow to pass the classifier in 
            
            
        load the classifier after getting the above data (can we do this with sumo ... probably)
        
        