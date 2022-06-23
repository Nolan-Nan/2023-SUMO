# SUMO-Training
Contains current WIP training files and the pipeline. 

# To Do
- [x] Switch cue phrases feature-set in the feature extractor 
- [X] Add cue phrases feature-set in the ml pipeline - relevance 
- [X] Add new classifier - relevance
- [X] Add cue phrases feature-set in the ml pipeline - rhetorical
- [x] Train the DTC classifier for the relevance feature set
- [X] Remove the relevance CRF code
- [X] Integrate the DTC classifier into the pipeline
- [ ] Integrate ASMO into the pipeline
- [X] Re-train CRF classifier using best performing normal ML classifier for rhet labels
- [ ] Updated pipepline to use re-trained CRF classifier
- [ ] Test with the different classifiers to see performance on CRF - ensure there is more support for Fact, etc


current classifiers: 
RHETORICAL - DTC -> CRF classifier 

              precision    recall  f1-score   support

         1.0      0.831     0.884     0.857      4736
         4.0      0.898     0.849     0.873       873
         5.0      0.847     0.732     0.785       272
         6.0      0.818     0.737     0.775      3260
         3.0      0.943     0.929     0.936       196
         2.0      0.857     0.857     0.857         7

   micro avg      0.836     0.826     0.831      9344
   macro avg      0.866     0.831     0.847      9344
weighted avg      0.836     0.826     0.829      9344

RELEVANCE - RF classifier (41.2% F-Score)
PRECISION MEAN
0.3272512120188103
RECALL MEAN
0.5725366432127635
F-SCORE MEAN
0.4123655391999811

