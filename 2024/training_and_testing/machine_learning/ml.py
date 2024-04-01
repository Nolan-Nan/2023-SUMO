
"""
machine learning (select classifiers / features / etc when running this file)

necessary to run feature extractor first to generate relevant csv file

@author: rozano
"""

import numpy as np
import csv
import sys
# import lime

class ml:
    def __init__(self):
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

        ## updated entities feature-set
        self.citationent_X = np.array([])



        # spacy entities
        self.loc_ent_X = np.array([])
        self.org_ent_X = np.array([])
        self.date_ent_X = np.array([])
        self.person_ent_X = np.array([])
        self.fac_ent_X = np.array([])
        self.norp_ent_X = np.array([])
        self.gpe_ent_X = np.array([])
        self.event_ent_X = np.array([])
        self.law_ent_X = np.array([])
        self.time_ent_X = np.array([])
        self.work_of_art_ent_X = np.array([])
        self.ordinal_ent_X = np.array([])
        self.cardinal_ent_X = np.array([])
        self.money_ent_X = np.array([])
        self.percent_ent_X = np.array([])
        self.product_ent_X = np.array([])
        self.quantity_ent_X = np.array([])
        self.spacy = np.array([])
        self.total_spacy_X  = np.array([])

        self.HGents = np.array([])

        # all values are 0, thus non-beneficial in ml
        # self.caseent_X = np.array([])
        ##for cue phrase feature-set
        self.asp_X = np.array([])
        self.modal_X = np.array([])
        self.voice_X = np.array([])
        self.negcue_X = np.array([])
        self.tense_X = np.array([])


        # new cue phrases
        # modal data on the entire sentence (count and boolean values)
        self.modal_pos_bool_X = np.array([])
        self. modal_dep_bool_X = np.array([])
        self.modal_dep_count_X = np.array([])
        self.modal_pos_count_X = np.array([])

        # verb data on the first verb
        self.new_modal_X = np.array([])
        self.new_tense_X = np.array([])
        self.new_dep_X = np.array([])
        self.new_tag_X = np.array([])
        self.new_negative_X = np.array([])
        self.new_stop_X = np.array([])
        self.new_voice_X = np.array([])

        # data on the token after the verb
        self.second_pos_X = np.array([])
        self.second_dep_X = np.array([])
        self.second_tag_X = np.array([])
        self.second_stop_X = np.array([])


    def classifier_performance(self, X_train, y_train, X_test, y_test, classifier, label, feat_names):
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        y_true = y_test
        y_pred = predictions


       # if classifier.__class__.__name__ != 'SVC' :
           # probability = classifier.probability
            # can't do it when probability is false
           # if probability == True:

     #       y_scores = classifier.predict_proba(X_test)
     #       print("PREDICT PROBA")
    #        print(y_scores)

     #       from sklearn.metrics import roc_curve
    #        from sklearn.metrics import roc_auc_score

            #for ROC score - keeping positive predictions
     #       probs = y_scores[:, 1]
                #get AUC curve - compute the score
      #      auc = roc_auc_score(y_test, probs)
      #      print("AUC: %.2f" % auc)
      #      fpr, tpr, thresholds = roc_curve(y_test, probs)
     #       self.plot_roc_curve(fpr, tpr, classifier, feat_names)


        # print score report
        from sklearn.metrics import classification_report
        clf_report = classification_report(y_true, y_pred, labels=np.unique(y_pred))
        print("CLF REPORT:")
        print(clf_report)

        from sklearn.metrics import f1_score
        print("MACRO F-SCORE:")
        mac_fscore = f1_score(y_true, y_pred, average = 'macro')
        print(mac_fscore)
        print("MICRO F-SCORE:")
        mic_fscore = f1_score(y_true, y_pred, average = 'micro')
        print(mic_fscore)
        print("WEIGHTED F-SCORE (multinominal):")
        weigh_fscore = f1_score(y_true, y_pred, average = 'weighted')
        print(weigh_fscore)

        print(label)
        #for relevance classifier
        if label == "binary":
            print("BINARY F-SCORE:")
            bin_fscore = f1_score(y_true, y_pred, average='binary')
            print(bin_fscore)

        # # Visualize Decision Tree for DTC only
        #from sklearn.tree import export_graphviz

        # # Creates dot file named tree.dot
        # export_graphviz(
            #    classifier,
             #   out_file =  "output.dot",
             #   feature_names = ['loc3_X', 'loc4_X', 'loc1_X', 'loc2_X', 'loc5_X', 'loc6_X'],
             #   class_names = ['rhet_0', 'rhet_1', 'rhet_2', 'rhet_3', 'rhet_4', 'rhet_5', 'rhet_6'],
             #   filled = True,
             #   rounded = True)

        # save as PNG
       # from subprocess import call
       # call(['dot', '-Tpng', 'output.dot', '-o', 'output.png', '-Gdpi=600'])

             #lime code to explain classifier work

    #    import lime
    #    import lime.lime_tabular
    #    import lime.explanation.Explanation as expl

    #    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feat_names, discretize_continuous=True)
    #    i = np.random.randint(0, X_test.shape[0])
    #    exp = explainer.explain_instance(X_test[i], classifier.predict_proba, num_features=2, top_labels=1)

    #    import matplotlib.pyplot as plt

        #listt = exp.as_list(label=1)
        #1print(listt)

     #   fig = expl.as_pyplot_figure(exp)
     #   figure = plt.fig()
    #    plt.plot(range(10))
     #   plt.show()


    # train trest split ROC curve
    def plot_roc_curve(self, fpr, tpr, classifier, feat_names):
        import matplotlib.pyplot as plt

        plt.plot(fpr, tpr, color='blue', label='ROC')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of ' + classifier.__class__.__name__ + ': ' + feat_names)
        plt.legend()
        plt.show()

    # roc curve ploted for 10 fold cross validation
    def plot_cross_roc_curve(self, X, Y, clf, feat_names, cv):
        from sklearn.metrics import auc
        from sklearn.metrics import plot_roc_curve
        import matplotlib.pyplot as plt

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots()

        for i, (train, test) in enumerate (cv.split(X, Y)):
                 clf.fit(X[train], Y[train])
                 viz = plot_roc_curve(clf, X[test], Y[test],
                                  name='ROC fold {}'.format(i),
                                  alpha=0.3, lw=1, ax=ax)
                 interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                 interp_tpr[0] = 0.0
                 tprs.append(interp_tpr)
                 aucs.append(viz.roc_auc)

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                             label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                             lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                                     label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
                            title="ROC of " + clf.__class__.__name__ + ":" + feat_names)
        ax.legend(loc="lower right", fontsize=6.5)
        plt.show()



            #lime code to explain classifier work

           # from lime import lime_text
           # from lime.lime_text import LimeTextExplainer
           # explainer = LimeTextExplainer(class_names = feat_names)
           # exp = explainer.explain_instance()
           # print("Probability =", clf.predict_proba(X[test]), [0, 1])

           # exp.as_list()


    def classification_report_with_f1_score(self, y_true, y_pred, label):
        from sklearn.metrics import classification_report, f1_score
        print(classification_report(y_true, y_pred, labels=np.unique(y_pred)))
        if label == "multinomial":
            f1 = f1_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))
            print("F-SCORE multi")
            print(f1)
          #  print("MICRO F-SCORE:")
         #   mic_fscore = f1_score(y_true, y_pred, average = 'micro')
          #  print(mic_fscore)
            return f1
        elif label == "binary":
            f1 = f1_score(y_true, y_pred, average='binary', labels=np.unique(y_pred))
            print("F-SCORE bin")
            print(f1)

            return f1
        else:
            print("Could not detect target type")
            sys.exit()


    def supervised_ml(self, X, Y, label, feat_names, target_names, mode):
        model = input("Enter anything to use train_test_split, or '1' to use cross_val_score ")
        if model == '1':
            print("Using cross_val_score ")
            from sklearn.model_selection import cross_val_score
        else:
            print("Using train_test_split ")
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size = .1)


        rep = ""
        while rep != "done":
            clf, clf_name = mode.select_classifier(label)
            #cross val
            if model == '1':
                from sklearn.metrics import make_scorer
                from sklearn.model_selection import KFold

                #cross val fold
                cv = KFold(n_splits=10, shuffle=True)

                #can't do ROC for SVC
                if label == "binary":
                    if clf.__class__.__name__ != 'SVC' :
                       self.plot_cross_roc_curve(X, Y, clf, feat_names, cv)

                score = cross_val_score(clf, X=X, y=Y, cv=cv,
                    scoring=make_scorer(self.classification_report_with_f1_score, label=label))


                clf = clf.fit(X, Y)

                import pickle
                # CHANGE THIS NAME from 'test' to the classifier name you are writing, this will need to be updated in the other files / pipeline files
                f = open('rel-new.pickle', 'wb')
                pickle.dump(clf, f)
                f.close()
                print(score)
                # to print averaged recall and precision

                if label == "binary":
                    recall = cross_val_score(clf, X=X, y=Y, cv=cv, scoring='recall')
                    precision = cross_val_score(clf, X=X, y=Y, cv=cv, scoring='precision')
                    # not optimised for weighted average, will need to implement a different method here
                    print("PRECISION MEAN")
                    print(np.mean(precision))
                    print("RECALL MEAN")
                    print(np.mean(recall))
                print("F-SCORE MEAN")
                print(np.mean(score))





            #train test split
            else:
                self.classifier_performance(X_train, y_train, X_test, y_test, clf, label, feat_names)
            print("Classifier:", clf_name, "features:", feat_names, "target:", target_names)

            rep = input("Enter anything to try again, or 'done' to quit ")
        print('+++++++++++', 'DONE', '+++++++++++')



    #Extract all data and prepare it for ML
    def prep_data(self, filename):
        with open('./data/' + filename + '.csv', 'r') as infile:
            reader = csv.DictReader(infile)

        # for each row in the MLDATA cv file, get the corresponding result - add to array
            for row in reader:
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
              #  self.citationent_X = np.append(self.citationent_X, [float(row['citation entities'])])

                self.loc_ent_X = np.append(self.loc_ent_X, [float(row['loc ent'])])
                self.org_ent_X = np.append(self.org_ent_X, [float(row['org ent'])])
                self.date_ent_X = np.append(self.date_ent_X, [float(row['date ent'])])
                self.person_ent_X = np.append(self.person_ent_X, [float(row['person ent'])])
                self.fac_ent_X = np.append(self.fac_ent_X, [float(row['fac_ent'])])
                self.norp_ent_X = np.append(self.norp_ent_X, [float(row['norp_ent'])])
                self.gpe_ent_X = np.append(self.gpe_ent_X, [float(row['gpe_ent'])])
                self.event_ent_X = np.append(self.event_ent_X, [float(row['event_ent'])])
                self.law_ent_X = np.append(self.law_ent_X, [float(row['law_ent'])])
                self.time_ent_X = np.append(self.time_ent_X, [float(row['time_ent'])])
                self.work_of_art_ent_X = np.append(self.work_of_art_ent_X, [float(row['work_of_art_ent'])])
                self.ordinal_ent_X = np.append(self.ordinal_ent_X, [float(row['ordinal_ent'])])
                self.cardinal_ent_X = np.append(self.cardinal_ent_X, [float(row['cardinal_ent'])])
                self.money_ent_X = np.append(self.money_ent_X, [float(row['money_ent'])])
                self.percent_ent_X = np.append(self.percent_ent_X, [float(row['percent_ent'])])
                self.product_ent_X = np.append(self.product_ent_X, [float(row['product_ent'])])
                self.quantity_ent_X = np.append(self.quantity_ent_X, [float(row['quantity_ent'])])

                #self.total_spacy_X = np.append(self.total_spacy_X, [float(row['all ent'])])
                self.modal_pos_bool_X =  np.append(self.modal_pos_bool_X, [float(row['cp pos bool'])])
                self.modal_dep_bool_X = np.append(self.modal_dep_bool_X, [float(row['cp dep bool'])])
                self.modal_dep_count_X = np.append(self.modal_dep_count_X, [float(row['cp dep count'])])
                self.modal_pos_count_X = np.append(self.modal_pos_count_X, [float(row['cp pos count'])])
                self.new_modal_X = np.append(self.new_modal_X, [float(row['cp modal'])])
                self.new_tense_X = np.append(self.new_tense_X, [float(row['cp tense'])])
                self.new_dep_X = np.append(self.new_dep_X, [float(row['cp dep'])])
                self.new_tag_X = np.append(self.new_tag_X, [float(row['cp tag'])])
                self.new_negative_X = np.append(self.new_negative_X, [float(row['cp negative'])])
                self.new_stop_X = np.append(self.new_stop_X, [float(row['cp stop'])])
                self.new_voice_X = np.append(self.new_voice_X, [float(row['cp voice'])])

                self.second_pos_X = np.append(self.second_pos_X, [float(row['cp second pos'])])
                self.second_dep_X = np.append(self.second_dep_X, [float(row['cp second dep'])])
                self.second_tag_X = np.append(self.second_tag_X, [float(row['cp second tag'])])
                self.second_stop_X = np.append(self.second_stop_X, [float(row['cp second stop'])])



    def exec(self):
        location = self.loc1_X, self.loc2_X, self.loc3_X, self.loc4_X, self.loc5_X, self.loc6_X
        HGlocation = self.HGloc1_X, self.HGloc2_X, self.HGloc3_X, self.HGloc4_X, self.HGloc5_X, self.HGloc6_X
        quotation = self.inq_X, self.qb_X
        entities = self.legalent_X, self.enamex_X # ALSO WANT TO ATTEMPT ADDING THE CITATION ENTITY HERE
        asmo = self.agree_X, self.outcome_X
        cue_phrase = self.asp_X, self.modal_X, self.voice_X, self.negcue_X, self.tense_X
        sent_length = self.sentlen_X
        HGsent_length = self.HGsentlen_X
        tfidf_max = self.tfidf_max_X
        tfidf_HGavg = self.tfidf_HGavg_X
        tfidf_top20 = self.tfidf_top20_X
        rhet_role = self.rhet_X
        wordlist = self.wordlist_X
        pasttense = self.pasttense_X
        rhet_y = self.rhet_y
        rel_y = self.rel_y
       # citationent_X = self.citationent_X
        #   total_spacy = self.total_spacy_X
        HGents = self.loc_ent_X, self.org_ent_X, self.date_ent_X, self.person_ent_X, self.fac_ent_X, self.norp_ent_X, \
                      self.gpe_ent_X, self.event_ent_X, self.law_ent_X, self.time_ent_X, self.work_of_art_ent_X, self.ordinal_ent_X, \
                      self.cardinal_ent_X, self.money_ent_X, self.percent_ent_X, self.product_ent_X, self.quantity_ent_X

        modal = self.modal_dep_bool_X,  self.modal_dep_count_X   #currently without the POS data
        verb =  self.new_tense_X, self.new_tag_X, self.new_negative_X, self.new_stop_X, self.new_voice_X, self.new_modal_X # currently without dep and modality
        secondToken = self.second_pos_X, self.second_dep_X, self.second_tag_X, self.second_stop_X
        new_cue_phrases = self.modal_dep_bool_X, self.modal_dep_count_X, self.new_modal_X, self.new_tense_X, self.new_dep_X, self.new_tag_X, self.new_negative_X, self.new_stop_X, self.new_voice_X, self.second_pos_X, self.second_dep_X, self.second_tag_X, self.second_stop_X

        import mode_selector
        mode = mode_selector.mode_selector(location, quotation, asmo,
        sent_length, tfidf_max, rhet_role, wordlist, rhet_y, rel_y, HGents, new_cue_phrases)
        num_of_features = input("how many features? ")
        X, feat_names = mode.select_features(num_of_features)
        Y, label, target_names = mode.select_target()



        print(X)
        print(feat_names)
        print(Y)
        print(label)
        print(target_names)

        self.supervised_ml(X, Y, label, feat_names, target_names, mode)

pipeline = ml()
pipeline.prep_data('MLdata')
# pipeline.prep_data('MLdata_train')
pipeline.exec()
