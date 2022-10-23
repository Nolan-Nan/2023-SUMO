"""
Pipeline for generating the summaries. 

how data is stored when the summary is created: 
    first level - /data/UKHLcorpus (cases)
                    /summarydata/UKHL_ - this is the relevant case name, 
                    updated with _feaetures at the end once the feature set is created

 to run with testing case numbers - need the UKHL corpus individual cases in a file called UKHL_corpus
 also need to duplicate and label it UKHL_corpus2 as it needs to be open twice at once - NEED TO FIX THIS
 
 nb - also need the 68txt_corpus and SUM_69_corpus folders in the /data folder as well as the corpus_list.csv 
 wordlist.csv and UKHL_corpus.csv - these files just stay in the /data file root
 
 also create a /summarydata folder, the files inside are generated by the code. 
 
 relevant dependencies are: 
    labelling.py
     tfidf_feature.py
     nvGroups.py
     ml.py
     summary.py 
     


   
    
    TODO: 
        HTML parser for cases not in the corpus, need to do the verb group and cue phrase matching first - which 
        would require retraining all the models based on the new feature set



@author: amyconroy
"""

class pipeline():
    def begin(self):
         print("Enter 1 to select a case number, anything else to input the link of a case.")  
         answer = input()
         if answer == '1':
             print("Enter the UKHL Corpus case number (1.19, 1.63, 1.35, 2.23, 2.34, 3.21, 3.14, 1.03, 1.32, 1.7, 1.27, 2.25,")
             print("2.16, 2.06, 2.02, 2.09, 2.03, 2.01, 2.08, 3.45, 3.36, 3.24, 3.17, 3.37, 3.03, 3.48, 3.42, 3.38 \n")
             casenum = input()
             self.prepareCase(casenum)
         else: 
             print("We do not yet support this feature.")
             # here we would go and make it to the similar csv file, label, get ASMO, then follow same pipeline
             
    def prepareCase(self, casenum):
        print("\n PREPARING THE DATA FOR SUMMARISATION\n")
        import labelling
        labelling.labelling(casenum)
        import featureExtractor
        featureExtractor.featureExtractor(casenum)
        # integrated cue phrases above, now need to add cue phrases below 
        import ml
        ml.ml(casenum, True)
        print("\n SUMO PIPELINE SUMMARIES: \n")
        import summary
        summary.summary(casenum)
        print("\n")
        print("\n SUMO Summary Pipeline Complete.")
        
        
pipeline = pipeline()
pipeline.begin()

    
    
    
