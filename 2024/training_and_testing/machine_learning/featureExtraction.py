
"""
extracts relevant features

@author: rozano / amyconroy to create ranking file
"""

import csv
import numpy as np
import math

#this file creates the mldata.csv

def createRankingFile(case_flag, sent_flag, y, rank_flag):
    with open('./data/corpus_ranking.csv', 'w', newline='') as outfile:
        fieldnames = ['case_id', 'sent_id', 'rank']

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for v in range(len(y)):
            writer.writerow({'case_id': case_flag[v], 'sent_id': sent_flag[v], 'rank': rank_flag[v]})

def get_end_par_in_lord(judge, case, current_paragraph):
    filename = 'comsum_' + case + '.csv'
    if case == 'N/A':
        filename = 'comsum_' + 'NA' + '.csv'
    with open('./data/comsum_corpus/' + filename, 'r') as infile:
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

def get_end_sent_in_lord(judge, case, current_sentence):
    filename = 'comsum_' + case + '.csv'
    if case == 'N/A':
        filename = 'comsum_' + 'NA' + '.csv'
    with open('./data/comsum_corpus/' + filename, 'r') as infile:
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

def get_end_sent_in_par(par, case, current_sentence):
    filename = 'comsum_' + case + '.csv'
    if case == 'N/A':
        filename = 'comsum_' + 'NA' + '.csv'
    with open('./data/comsum_corpus/' + filename, 'r') as infile:
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

def storeFeatures(case_flag, sent_flag, y, agree_X, outcome_X, loc1_X, loc2_X, loc3_X, loc4_X, loc5_X, loc6_X,
HGloc1_X, HGloc2_X, HGloc3_X, HGloc4_X, HGloc5_X, HGloc6_X, sentlen_X, HGsentlen_X, qb_X, inq_X, rhet_X, tfidf_max_X, tfidf_top20_X,
tfidf_HGavg_X, asp_X, modal_X, voice_X, negcue_X, tense_X, caseent_X, legalent_X, enamex_X, rhet_y, wordlist_X, pasttense_X,
provision_ent_X, instrument_ent_X, court_ent_X, casename_ent_X, citation_ent_X, judge_ent_X, loc_ent_X, org_ent_X, date_ent_X,
person_ent_X, time_ent_X, gpe_ent_X, fac_ent_X, ordinal_ent_X, totalner_ent_X, judgename, rhetlabel, new_tense_X, new_modal_X, modal_pos_bool_X,
modal_dep_bool_X, modal_dep_count_X, modal_pos_count_X, new_dep_X, new_tag_X, negtoken_X, verbstop_X, newvoice_X,
second_pos_X, second_dep_X, second_tag_X, second_stop_X):
    with open('./data/MLdata.csv', 'w', newline='') as outfile:
        fieldnames = ['case_id', 'sent_id', 'align', 'agree', 'outcome', 'loc1', 'loc2', 'loc3',
        'loc4', 'loc5', 'loc6', 'HGloc1', 'HGloc2', 'HGloc3', 'HGloc4', 'HGloc5', 'HGloc6', 'sentlen',
        'HGsentlen', 'quoteblock', 'inline_q', 'rhet', 'tfidf_max', 'tfidf_top20', 'tfidf_HGavg', 'aspect', 'modal',
        'voice', 'negation', 'tense', 'case entities', 'legal entities', 'enamex','rhet_target', 'wordlist', 'past tense',
        'case name entity', 'provision ent', 'instrument ent', 'court ent', 'case name ent', 'citation bl ent', 'judge ent',
        'loc ent', 'org ent', 'date ent', 'person ent', 'time ent', 'gpe ent', 'fac ent', 'ordinal ent', 'all ent', 'judgename', 'rhet label',
        'cp tense', 'cp modal', 'cp pos bool', 'cp dep bool', 'cp dep count', 'cp pos count', 'cp dep', 'cp tag', 'cp negative',
        'cp stop', 'cp voice', 'cp second pos', 'cp second dep', 'cp second tag', 'cp second stop']

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()


        for v in range(len(y)):
            writer.writerow({'case_id': case_flag[v], 'sent_id': sent_flag[v], 'align': y[v], 'agree': agree_X[v],
            'outcome': outcome_X[v], 'loc1': loc1_X[v], 'loc2': loc2_X[v], 'loc3': loc3_X[v], 'loc4': loc4_X[v],
            'loc5': loc5_X[v], 'loc6': loc6_X[v], 'HGloc1': HGloc1_X[v], 'HGloc2': HGloc2_X[v], 'HGloc3': HGloc3_X[v], 'HGloc4': HGloc4_X[v],
            'HGloc5': HGloc5_X[v], 'HGloc6': HGloc6_X[v], 'sentlen': sentlen_X[v], 'HGsentlen': HGsentlen_X[v], 'quoteblock': qb_X[v], 'inline_q': inq_X[v],
            'rhet': rhet_X[v], 'tfidf_max': tfidf_max_X[v], 'tfidf_top20': tfidf_top20_X[v], 'tfidf_HGavg': tfidf_HGavg_X[v],'aspect': asp_X[v],
            'modal': modal_X[v], 'voice': voice_X[v], 'negation': negcue_X[v], 'tense': tense_X[v], 'case entities': caseent_X[v],
            'legal entities': legalent_X[v], 'enamex': enamex_X[v], 'rhet_target': rhet_y[v], 'wordlist': wordlist_X[v],
            'past tense': pasttense_X[v], 'provision ent' : provision_ent_X[v], 'instrument ent' : instrument_ent_X[v], 'court ent' : court_ent_X[v],
            'case name ent' : casename_ent_X[v], 'citation bl ent' : citation_ent_X[v], 'judge ent' : judge_ent_X[v], 'loc ent' : loc_ent_X[v], 'org ent' : org_ent_X[v], 'date ent' :
            date_ent_X[v], 'person ent' : person_ent_X[v], 'time ent' : time_ent_X[v], 'gpe ent' : gpe_ent_X[v],
            'fac ent' : fac_ent_X[v], 'ordinal ent' : ordinal_ent_X[v], 'all ent' : totalner_ent_X[v], 'judgename' : judgename[v], 'rhet label' : rhetlabel[v],
            'cp tense': new_tense_X[v], 'cp modal': new_modal_X[v], 'cp pos bool' :  modal_pos_bool_X[v], 'cp dep bool': modal_dep_bool_X[v],
            'cp dep count':  modal_dep_count_X[v], 'cp pos count': modal_pos_count_X[v], 'cp dep': new_dep_X[v], 'cp tag': new_tag_X[v], 'cp negative': negtoken_X[v],
            'cp stop': verbstop_X[v], 'cp voice' : newvoice_X[v], 'cp second pos': second_pos_X[v], 'cp second dep' : second_dep_X[v],
            'cp second tag' : second_tag_X[v], 'cp second stop' : second_stop_X[v]})

def get_wordlist(text):
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

#Target/label
#relevance target
y = np.array([])
#rhetorical role target
rhet_y = np.array([])

#List of features
##for asmo feature-set
agree_X = np.array([])
outcome_X = np.array([])
##for location feature-set
loc1_X = np.array([]); loc2_X = np.array([]); loc3_X = np.array([])
loc4_X = np.array([]); loc5_X = np.array([]); loc6_X = np.array([])
sentlen_X = np.array([])
rhet_X = np.array([])
wordlist_X = np.array([])
pasttense_X = np.array([])

import tfidf_feature
tfidf = tfidf_feature.tfidf_calc()
tfidf_max_X = np.array([])
tfidf_top20_X = np.array([])

#Hachey and Grover's original features
##for location feature-set
HGloc1_X = np.array([]); HGloc2_X = np.array([]); HGloc3_X = np.array([])
HGloc4_X = np.array([]); HGloc5_X = np.array([]); HGloc6_X = np.array([])
tfidf_HGavg_X = np.array([])
HGsentlen_X = np.array([])
qb_X = np.array([])
inq_X = np.array([])
##for entities feature-set
caseent_X = np.array([])
legalent_X = np.array([])
enamex_X = np.array([])

# for updated entity feature-set
citationent_X = np.array([])
casenameent_X = np.array([])

# updated NER from spacy
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
spacy = np.array([])
total_spacy_X  = np.array([])
totalner_ent_X = np.array([])

##for cue phrase feature-set
asp_X = np.array([])
modal_X = np.array([])
voice_X = np.array([])
negcue_X = np.array([])
tense_X = np.array([])

#for storing Xs values
case_flag = []
sent_flag = []

# for seq modelling
judgename = []
rhetlabel = []

# for amy's new cue phrase feature-set
new_tense_X = np.array([])
new_modal_X = np.array([])
modal_pos_bool_X = np.array([])
modal_dep_bool_X = np.array([])
modal_dep_count_X = np.array([])
modal_pos_count_X = np.array([])
new_dep_X = np.array([])
new_tag_X = np.array([])
negtoken_X = np.array([])
verbstop_X = np.array([])
newvoice_X = np.array([])
second_pos_X = np.array([])
second_dep_X = np.array([])
second_tag_X = np.array([])
second_stop_X = np.array([])

import cuephrases
new_tense_X, new_modal_X, modal_pos_bool_X, modal_dep_bool_X, modal_dep_count_X, modal_pos_count_X, new_dep_X, new_tag_X, negtoken_X, verbstop_X, newvoice_X, second_pos_X, second_dep_X, second_tag_X, second_stop_X = cuephrases.cuePhrases()


with open('data/UKHL_corpus.csv', 'r') as infile:
    reader = csv.DictReader(infile)

    #for location features
    judge = ''
    case = ''
    par = ''
    loc1 = 0; loc2 = 0; loc3 = 0; loc4 = 0; loc5 = 0; loc6 = 0

    #for quotations
    qb_bool = False
    quoteblock = 0
    inquotes = False
    word_inq = 0

   # cnt = 0 #for quick test
    for row in reader:
        # if row['case_id'] != '1.63': #quick test on first case
        #     continue
        # if row['case_id'] == 'N/A': #quick test on first 3 cases
        #     break

        # THIS IS HOW THE CREATION OF THE CORPUS SKIPS THE FIRST LINE ETC ETC
        if row['agree'] == 'no match' or row['role'] == '<prep-date>'\
        or row['role'] == '<sub-heading>' or row['role'] == '<separator>' or row['role'] == '<new-case>':
            continue

        case_flag.append(row['case_id'])
        sent_flag.append(row['sentence_id'])
        judgename.append(row['judge'])
        rhetlabel.append(row['role'])

        import nvGroups
    #    import entities
        print("nounin")
        asp, modal, voice, negation, tense, pasttense = nvGroups.get_verb_features(row['case_id'], row['sentence_id'])
        pasttense_X = np.append(pasttense_X, [pasttense])
        if asp == 'SIMPLE':
            asp_X = np.append(asp_X, [1])
        elif asp == 'PERF':
            asp_X = np.append(asp_X, [2/3])
        elif asp == 'PROG':
            asp_X = np.append(asp_X, [1/3])
        else:
            asp_X = np.append(asp_X, [0])
        if modal == 'NO':
            modal_X = np.append(modal_X, [1])
        elif modal == 'YES':
            modal_X = np.append(modal_X, [1/2])
        else:
            modal_X = np.append(modal_X, [0])
        if voice == 'ACT':
            voice_X = np.append(voice_X, [1])
        elif voice == 'PASS':
            voice_X = np.append(voice_X, [1/2])
        else:
            voice_X = np.append(voice_X, [0])
        if negation == 'yes':
            negcue_X = np.append(negcue_X, [1])
        else:
            negcue_X = np.append(negcue_X, [0])
        if tense == 'PRES':
            tense_X = np.append(tense_X, [1])
        elif tense == 'PRESorBASE':
            tense_X = np.append(tense_X, [3/4])
        elif tense == 'PAST':
            tense_X = np.append(tense_X, [2/4])
        elif tense == 'INF':
            tense_X = np.append(tense_X, [1/4])
        else:
            tense_X = np.append(tense_X, [0])

        # ENTITIES FEATURE SET
        caseent, legalent, enamex = nvGroups.get_noun_features(row['case_id'], row['sentence_id'])
        caseent_X = np.append(caseent_X, [caseent])
        legalent_X = np.append(legalent_X, [legalent])
        enamex_X = np.append(enamex_X, [enamex])

        tfidf.get_doc(row['case_id'])
        sent_max_tfidf, sent_intop20_tfidf, sent_avg_tfidf = tfidf.get_sent_features(row['text'])
        tfidf_max_X = np.append(tfidf_max_X, [sent_max_tfidf])
        tfidf_top20_X = np.append(tfidf_top20_X, [sent_intop20_tfidf])
        tfidf_HGavg_X = np.append(tfidf_HGavg_X, [sent_avg_tfidf])

        wordlist = get_wordlist(row['text'])
        wordlist_X = np.append(wordlist_X, [wordlist])

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
        qb_X = np.append(qb_X, [quoteblock])

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
        inq_X = np.append(inq_X, [inq])

        sent_len = len(row['text'].split())
        HGsentlen_X = np.append(HGsentlen_X, [sent_len])
        sent_len = math.log(sent_len,625)
        sentlen_X = np.append(sentlen_X, [sent_len])

        current_case = row['case_id']
        current_judge = row['judge']
        current_paragraph = row['para_id']
        if current_paragraph == '0.5':
            current_paragraph = '0'
        elif '.5' in current_paragraph:
            current_paragraph = current_paragraph.replace('.5', '')
        current_sentence = row['sentence_id']
        print("TEST SENTENCE")
        print(current_sentence)

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
            end_lord_paragraphs = int(get_end_par_in_lord(judge, case, current_paragraph))
            end_lord_sentences = int(get_end_sent_in_lord(judge, case, current_sentence))

        if par != current_paragraph:
            par = current_paragraph
            start_par_sent = int(current_sentence)
            end_par_sentences = int(get_end_sent_in_par(par, case, current_sentence))

        loc1 = 1 + int(current_paragraph) - start_lord_par
        if loc1 <= 0:
            loc1 = 0
            print(current_case)
            print(current_sentence)
        else:
            loc1 = math.log(loc1, 81)
        loc2 = 1 + end_lord_paragraphs - int(current_paragraph)
        if loc2 <= 0:
            loc2 = 0
            print(current_case)
            print(current_sentence)
        else:
            loc2 = math.log(loc2, 81)
        loc3 = 1 + int(current_sentence) - start_lord_sent
        if loc3 <= 0:
            loc3 = 0
            print(current_case)
            print(current_sentence)
        else:
            loc3 = math.log(loc3, 625)
        loc4 = 1 + end_lord_sentences - int(current_sentence)
        if loc4 <= 0:
            loc4 = 0
            print(current_case)
            print(current_sentence)
        else:
            loc4 = math.log(loc4, 625)
        loc5 = 1 + int(current_sentence) - start_par_sent
        if loc5 <= 0:
            loc5 = 0
            print(current_case)
            print(current_sentence)
        else:
            loc5 = math.log(loc5, 16)
        loc6 = 1 + end_par_sentences - int(current_sentence)
        if loc6 <= 0:
            loc6 = 0
            print(current_case)
            print(current_sentence)
        else:
            loc6 = math.log(loc6, 16)

        #norm attempt-1
        # loc1 = (int(current_paragraph) - start_lord_par)/(end_lord_paragraphs - start_lord_par + 1)
        # loc2 = (end_lord_paragraphs - int(current_paragraph))/(end_lord_paragraphs - start_lord_par + 1)
        # loc3 = (int(current_sentence) - start_lord_sent)/(end_lord_sentences - start_lord_sent + 1)
        # loc4 = (end_lord_sentences - int(current_sentence))/(end_lord_sentences - start_lord_sent + 1)
        # loc5 = (int(current_sentence) - start_par_sent)/(end_par_sentences - start_par_sent + 1)
        # loc6 = (end_par_sentences - int(current_sentence))/(end_par_sentences - start_par_sent + 1)

        #un-normalised
        HGloc1 = int(current_paragraph) - start_lord_par
        HGloc2 = end_lord_paragraphs - int(current_paragraph)
        HGloc3 = int(current_sentence) - start_lord_sent
        HGloc4 = end_lord_sentences - int(current_sentence)
        HGloc5 = int(current_sentence) - start_par_sent
        HGloc6 = end_par_sentences - int(current_sentence)

        HGloc1_X = np.append(HGloc1_X, [HGloc1]); HGloc2_X = np.append(HGloc2_X, [HGloc2]); HGloc3_X = np.append(HGloc3_X, [HGloc3])
        HGloc4_X = np.append(HGloc4_X, [HGloc4]); HGloc5_X = np.append(HGloc5_X, [HGloc5]); HGloc6_X = np.append(HGloc6_X, [HGloc6])

        loc1_X = np.append(loc1_X, [loc1]); loc2_X = np.append(loc2_X, [loc2]); loc3_X = np.append(loc3_X, [loc3])
        loc4_X = np.append(loc4_X, [loc4]); loc5_X = np.append(loc5_X, [loc5]); loc6_X = np.append(loc6_X, [loc6])

        if row['align'] == 'NONE':
            y = np.append(y, [0])
        else:
            y = np.append(y, [1])

        if row['agree'] == 'NONE':
            if row['ackn'] != 'NONE':
                agree_X = np.append(agree_X, [0.5])
            else:
                agree_X = np.append(agree_X, [0])
        else:
            agree_X = np.append(agree_X, [1])
        if row['outcome'] == 'NONE':
            outcome_X = np.append(outcome_X, [0])
        else:
            outcome_X = np.append(outcome_X, [1])

        if row['role'] == 'FACT':
           # rhet_y = np.append(rhet_y, [0])
            rhet_y = np.append(rhet_y, [2])
            rhet_X = np.append(rhet_X, [2/6])
        if row['role'] == 'PROCEEDINGS':
          #  rhet_y = np.append(rhet_y, [0])
            rhet_y = np.append(rhet_y, [3])
            rhet_X = np.append(rhet_X, [3/6])
        if row['role'] == 'BACKGROUND':
          #  rhet_y = np.append(rhet_y, [0])
            rhet_y = np.append(rhet_y, [4])
            rhet_X = np.append(rhet_X, [4/6])
        if row['role'] == 'FRAMING':
            rhet_y = np.append(rhet_y, [5])
          #  rhet_y = np.append(rhet_y, [1])
            rhet_X = np.append(rhet_X, [5/6])
        if row['role'] == 'DISPOSAL':
            rhet_y = np.append(rhet_y, [6])
           # rhet_y = np.append(rhet_y, [1])
            rhet_X = np.append(rhet_X, [1])
        if row['role'] == 'TEXTUAL':
           # rhet_y = np.append(rhet_y, [0])
            rhet_y = np.append(rhet_y, [1])
            rhet_X = np.append(rhet_X, [1/6])
        if row['role'] == 'NONE':
            rhet_y = np.append(rhet_y, [0])
            rhet_X = np.append(rhet_X, [0])



       # citation = entities.new_citation_feature(row['text'])
       # if citation is True:
       #     citationent_X = np.append(citationent_X, [1])
       # else:
       #     citationent_X = np.append(citationent_X, [0])

       # casename = entities.new_casename_feature(row['text'])
       # if casename is True:
       #     citationent_X = np.append(citationent_X, [1])
       # else:
       #     citationent_X = np.append(citationent_X, [0])

        loc_ent_X = np.append(loc_ent_X, row['loc ent'])
        org_ent_X = np.append(org_ent_X, row['org ent'])
        date_ent_X = np.append(date_ent_X, row['date ent'])
        person_ent_X = np.append(person_ent_X, row['person ent'])
        fac_ent_X = np.append(fac_ent_X, row['fac_ent'])
        norp_ent_X = np.append(norp_ent_X, row['norp_ent'])
        gpe_ent_X = np.append(gpe_ent_X, row['gpe_ent'])
        event_ent_X = np.append(event_ent_X, row['event_ent'])
        law_ent_X = np.append(law_ent_X, row['law_ent'])
        time_ent_X = np.append(time_ent_X, row['time_ent'])
        work_of_art_ent_X = np.append(work_of_art_ent_X, row['work_of_art_ent'])
        ordinal_ent_X = np.append(ordinal_ent_X, row['ordinal_ent'])
        cardinal_ent_X = np.append(cardinal_ent_X, row['cardinal_ent'])
        money_ent_X = np.append(money_ent_X, row['money_ent'])
        percent_ent_X = np.append(percent_ent_X, row['percent_ent'])
        product_ent_X = np.append(product_ent_X, row['product_ent'])
        quantity_ent_X = np.append(quantity_ent_X, row['quantity_ent'])
        totalner_ent_X = np.append(totalner_ent_X, row['all ent'])



        # #for quick test
       # cnt +=1
       # if cnt == 3:
         #    break


storeFeatures(case_flag, sent_flag, y, agree_X, outcome_X, loc1_X, loc2_X, loc3_X, loc4_X, loc5_X, loc6_X,
HGloc1_X, HGloc2_X, HGloc3_X, HGloc4_X, HGloc5_X, HGloc6_X, sentlen_X, HGsentlen_X, qb_X, inq_X, rhet_X, tfidf_max_X, tfidf_top20_X,
tfidf_HGavg_X, asp_X, modal_X, voice_X, negcue_X, tense_X, caseent_X, legalent_X, enamex_X, rhet_y, wordlist_X, pasttense_X,
loc_ent_X, org_ent_X, date_ent_X, person_ent_X, fac_ent_X, norp_ent_X, gpe_ent_X, event_ent_X, law_ent_X, time_ent_X,
work_of_art_ent_X, ordinal_ent_X, cardinal_ent_X, money_ent_X, percent_ent_X, product_ent_X, quantity_ent_X,
totalner_ent_X, judgename, rhetlabel, new_tense_X, new_modal_X, modal_pos_bool_X, modal_dep_bool_X,
modal_dep_count_X, modal_pos_count_X, new_dep_X, new_tag_X, negtoken_X, verbstop_X, newvoice_X, second_pos_X, second_dep_X, second_tag_X, second_stop_X)

#rank_flag = 0
#createRankingFile(case_flag, sent_flag, y, rank_flag)
