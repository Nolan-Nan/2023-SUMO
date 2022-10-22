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

         1.0      0.915     0.779     0.841       953
         2.0      0.860     0.865     0.863      1222
         3.0      0.796     0.817     0.807      2534
         5.0      0.702     0.766     0.733      3495
         4.0      0.834     0.867     0.850      3409
         6.0      0.729     0.481     0.580      1014

   micro avg      0.789     0.791     0.790     12627
   macro avg      0.806     0.762     0.779     12627
weighted avg      0.790     0.791     0.788     12627

RELEVANCE - RF classifier (41.2% F-Score)
PRECISION MEAN
0.3272512120188103
RECALL MEAN
0.5725366432127635
F-SCORE MEAN
0.4123655391999811

