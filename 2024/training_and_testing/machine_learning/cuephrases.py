
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replicating Hachey and Grover's cue phrases feature set


H&G:

The term ‘cue phrase’ covers the kinds of stock phrases which are frequently
good indicators of rhetorical status (e.g. phrases such as The aim of this study
in the scientific article domain and It seems to me that in the HOLJ domain).
 Teufel and Moens invested a considerable amount of effort in building hand-crafted
 lexicons where the cue phrases are assigned to one of a number of fixed categories.
 A primary aim of the current research is to investigate whether this information
 can be encoded using automatically computable linguistic features. If they can,
 then this helps to relieve the burden involved in porting systems such as these to new domains.
 Our preliminary cue phrase feature set includes syntactic features of the main verb (voice,
                                                                                    tense, aspect, modality, negation),
 which we have shown in previous work to be correlated with rhetorical status (Grover et al. 2003).
 We also use sentence initial part- of-speech and sentence initial word features to roughly
 approximate for- mulaic expressions which are sentence-level adverbial or prepositional phrases.
 Subject features include the head lemma, entity type, and entity subtype.
 These features approximate the hand-coded agent features of Teufel and Moens.
 A main verb lemma feature simulates Teufel and Moens’s type of action and a feature
 encoding the part-of-speech after the main verb is meant to capture basic subcategorisation
 information.

 [step 1]:

 * VOICE
 * TENSE
 * ASPECT
 * MODALITY
 * NEGATION

 [step 2]:
 sentence Part of Speech& sentence initial word features
     approximate formulaic expressions - sentence-level adverbial or prepositional phrases

     verb subject -
         head lemma
         entity type
         entity subtype

For each sentence,
we use part-of-speech-based heuristics to determine tense,
voice, and presence of modal auxiliaries. This algorithm is shared
with the metadiscourse features, and the details are described below.


THE NEW CUE PHRASES FEATURE WILL INCLUDE:

new cue phrases feature set ->

    count of dep = aux (per sentence)
    count of pos = AUX (per sentence)
    boolean of dep = aux (true or false per sentence)
    boolean of pos= AUX (true or false per sentence)

for the first verb in the sentence ->
    tense
    modality
    dependency
    tag

@author: amyconroy
"""

import csv
import numpy as np
import spacy

# now being called by featureExtraction.py


# this will be on a sentence by sentence basis that it is parsed and cut up
# negation = https://stackoverflow.com/questions/54849111/negation-and-dependency-parsing-with-spacy?

def cuePhrases():
    import spacy

    print("calling cue phrases")

    nlp = spacy.load("en_core_web_sm")

    y = 0

    with open('data/UKHL_corpus.csv', 'r') as infile:
        reader = csv.DictReader(infile)

        verbDepList = []
        verbTagList = []
        verbTenseList = []

        tense = None
        verbDep = None
        verbTag = None

        aspects = []

        # going to actually put each sentence's one in here

        verbModalsResult = []
        tenseResult = []
        verbDepResult = []
        verbTagResult = []
        negResult = []
        verbStopResult = []
        voiceResult = []

        # second token
        secTokenPosList = []
        secTokenDepList = []
        secTokenTagList = []

        secTokenStopResult = []
        secTokenPosResult = []
        secTokenDepResult = []
        secTokenTagResult = []



        modalPosBoolResult = []
        modalDepBoolResult = []
        modalDepCountResult = []
        modalPosCountResult = []

        for row in reader:
            y += 1 # keep count for writing to the file later

            text = row['text']
            doc = nlp(text)


            tokenCount = 0

            modalPosBool, modalDepBool, modalDepCount, modalPosCount, tokenCount = modal(doc, nlp) #modality of the entire sentence

            normalizedDepCount = modalDepCount / tokenCount
            normalizedDepCount = round(normalizedDepCount, 2)


            modalPosBoolResult.append(modalPosBool)
            modalDepBoolResult.append(modalDepBool)
            modalDepCountResult.append(normalizedDepCount)
            modalPosCountResult.append(modalPosCount)

            # verb info for the entire sentence
            verbDepList, verbTagList, verbTenseList, verbModal, tense, verbDep, verbTag, negToken, verbStop, voice, secTokenPosList, secTokenDepList, secTokenTagList, secTokenStop, secTag, secPos, secDep = verb(doc, nlp, verbDepList, verbTagList, verbTenseList, secTokenPosList, secTokenDepList, secTokenTagList)

            if verbModal is None:
                verbModalsResult.append(0)
            else:
                verbModalsResult.append(verbModal)

            tenseResult.append(tense)
            verbDepResult.append(verbDep)
            verbTagResult.append(verbTag)
            negResult.append(negToken)
            verbStopResult.append(verbStop)

            if voice == 0:
                # voice is active
                voiceResult.append(1/2)
            else:  # voice is passive
                voiceResult.append(voice)

            secTokenStopResult.append(secTokenStop)
            secTokenPosResult.append(secPos)
            secTokenDepResult.append(secDep)
            secTokenTagResult.append(secTag)

           # aspects = aspectsAnalytics(doc, nlp, aspects)

    with open('data/cuePhrasesCorpus.csv','w', newline='')as outfile:
            fieldnames = ['POS modal boolean', 'Dep modal boolean', 'Dep modal count', 'POS modal count', 'Verb modal', 'Tense', 'Verb dep', 'Verb Tag',
                          'Verb Negation', 'Passive voice']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for v in range(y):
                writer.writerow({'POS modal boolean' : modalPosBoolResult[v], 'Dep modal boolean' : modalDepBoolResult[v], 'Dep modal count' : modalDepCountResult[v],
                                 'POS modal count' : modalPosCountResult[v], 'Verb modal' : verbModalsResult[v], 'Tense' : tenseResult[v], 'Verb dep' : verbDepResult[v],
                                 'Verb Tag' : verbTagResult[v], 'Verb Negation': negResult[v], 'Passive voice' : voiceResult[v]})

    # data smoothing for training
    newTenseType = convertForTraining(verbTenseList, tenseResult)
    newDepType = convertForTraining(verbDepList, verbDepResult)
    newTagType = convertForTraining(verbTagList, verbTagResult)


    secPosFinal = convertForTraining(secTokenPosList, secTokenPosResult)
    secDepFinal = convertForTraining(secTokenDepList, secTokenDepResult)
    secTagFinal = convertForTraining(secTokenTagList, secTokenTagResult)


    # convert to numpy array for feature extractor
    modalFinal = np.array(verbModalsResult)
    modalPosBoolFinal = np.array(modalPosBoolResult)
    modalDepBoolFinal = np.array(modalDepBoolResult)
    modalDepCountFinal = np.array(modalDepCountResult)
    modalPosCountFinal = np.array(modalPosCountResult)

    # second iteration of cue phrases
    negTokenFinal = np.array(negResult)
    verbStopFinal = np.array(verbStopResult)
    voiceFinal = np.array(voiceResult)

    # third iteration with the data of the second word
    secStopFinal = np.array(secTokenStopResult)



    return newTenseType, modalFinal, modalPosBoolFinal, modalDepBoolFinal, modalDepCountFinal, modalPosCountFinal, newDepType, newTagType, negTokenFinal, verbStopFinal, voiceFinal, secPosFinal, secDepFinal, secTagFinal, secStopFinal



    # then add this data as new rows to the UKHL corpus


def convertForTraining(completeList, resultList):
    newType = np.array([])
    if None in completeList:
        completeList.remove(None)
    listLength = len(completeList)
    for verbType in resultList:
        if verbType is None:
            newType = np.append(newType, 0)
        else: #similar method as previous featureExtraction.py (result is index over number of results)
            i = completeList.index(verbType)
            i += 1 # because we append 0 above where tense type is none (avoiding 0 for valuable training data in ML)
            result = i/listLength
            newType = np.append(newType, result)
    print(newType)

    return newType


# each token in the sentence
def modal(doc, nlp):
    modalPosBool = 0
    modalDepBool = 0
    modalDepCount = 0
    modalPosCount = 0

    tokenCount = 0


    for token in doc:
        if token.pos_ == "AUX":
            modalPosBool = 1 # true if found
            modalPosCount += 1

        if token.dep_ == "aux":
            modalDepCount += 1
            modalDepBool = 1 # true if found

        tokenCount += 1

    return modalPosBool, modalDepBool, modalDepCount, modalPosCount, tokenCount


#specifically data on the first verb in the sentence
def verb(doc, nlp, verbDepList, verbTagList, verbTenseList, secTokenPosList, secTokenDepList, secTokenTagList):
    rootVerb = False
    nextToken = False
    tense = None
    verbModal = None
    verbDep = None
    verbTag = None
    negToken = 0
    verbStop = 0
    passiveSentence = 0 # will change to 1 if a passive dep is found

    # token following the verb
    secTokenStop = 0

    secPos = None
    secDep = None
    secTag = None
    # left to do : get the data of the token following the first verb
    # data will be: its tag, dependency, POS, if it is a stop word
    # remove the count from the one feature set (leave it as just the booleans)
    # and then also check to see what is passive


    #adjust the 0's in the training data

    for token in doc:
                # token directly after the root verb
                if rootVerb == True and nextToken == False:
                    nextToken = True

                    # collet the tags in an array
                    if token.pos_ not in secTokenPosList:
                        secTokenPosList.append(token.pos_)

                    secPos = token.pos_

                    # get dep
                    if token.dep_ not in secTokenDepList:
                        secTokenDepList.append(token.dep_)

                    secDep = token.dep_


                    if token.tag_ not in secTokenTagList:
                        secTokenTagList.append(token.tag_)

                    secTag = token.tag_

                    # check if stop
                    if token.is_stop is True:
                        secTokenStop = 1



                # data for token after the first verb
                if token.dep_ == "ROOT":
                    rootVerb = True
                    #tense = nlp.vocab.morphology.tag_map[token.tag_].get("Tense")
                    tense = token.morph.to_dict().get("Tense")

                    if tense != "pres" and tense != "past":
                        #verbForm = nlp.vocab.morphology.tag_map[token.tag_].get("VerbForm")
                        verbForm = token.morph.to_dict().get("VerbForm")
                        if verbForm == "inf":
                            tense = verbForm

                    if tense not in verbTenseList:
                        verbTenseList.append(tense)

                    verbDep = token.dep_
                    if verbDep not in verbDepList:
                        verbDepList.append(verbDep)

                    verbTag = token.tag_
                    if verbTag not in verbTagList:
                        verbTagList.append(verbTag)

                    # if modality
                    if token.tag_ == "MD":
                        print("MODALITY")
                        verbModal = 1

                    if token.is_stop is True:
                        verbStop = 1

                    if token.dep_ == "subjpass" or token.dep_ == "auxpass":
                        passiveSentence = 1


                # find negative verb tokens
                negation_tokens = [tok for tok in doc if tok.dep_ == 'neg']
                negation_head_tokens = [token.head for token in negation_tokens]

                for token in negation_head_tokens:
                    if token.head.pos_ == "VERB":
                        negToken = 1


    return verbDepList, verbTagList, verbTenseList, verbModal, tense, verbDep, verbTag, negToken, verbStop, passiveSentence, secTokenPosList, secTokenDepList, secTokenTagList, secTokenStop, secTag, secPos, secDep

# maybe use this logic for the aspect parsing? https://stackoverflow.com/questions/60967134/named-entity-recognition-in-aspect-opinion-extraction-using-dependency-rule-matc
# using the logic from https://towardsdatascience.com/aspect-based-sentiment-analysis-using-spacy-textblob-4c8de3e0d2b9
def aspectsAnalytics(doc, nlp, aspects):

    adjective = ''
    target = ''

    for token in doc:
        if token.dep_ == 'nsubj' and token.pos_ == 'NOUN':
          target = token.text
        if token.pos_ == 'ADJ':
          prepend = ''
          for child in token.children:
            if child.pos_ != 'ADV':
              continue
            prepend += child.text + ' '
          adjective = prepend + token.text
        aspects.append({'aspect': target,
                          'description': adjective})
    print(aspects)

    return aspects



cuePhrases()
