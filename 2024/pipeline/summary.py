#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code to actually generate the summaries, all summaries also saved in a .txt in 
/summarydata/summaries

multiple different approaches implemented here: first approach, just taking the top 15 ranked
sentences (15 is the number chosen in line with Hachey and Grover)

@author: amyconroy
"""

import csv
import xml.etree.ElementTree as ET

class summary():
    def __init__(self, casenum):
        sentences, judges, citation, majority = self.getSentences(casenum)
        MLData = self.getSummaryData(casenum)
        respondent, appellant = self.summaryHeaderData(casenum)
        print("\n SUMMARY BASED ON RELEVANCE PREDICTIONS: \n")
        print(citation)
        print(respondent + ", " + appellant)
        print("\n")
        self.createRelevanceOnlySummary(MLData, sentences)
        
        print("\n Now creating a summary based on top ranked sentences.")
        print("\n How many sentences would you like in your summary?")
        summaryLength = input()
        print("\n SUMMARY BASED ON TOP " + summaryLength + " RANKED SENTENCES: \n")
        print(citation)
        print(respondent + " " + appellant)
        summaryLength = int(summaryLength)
        rankedData = self.createRankingOnlySummary(MLData, sentences, summaryLength)
        self.createRankingandRhetSummary(rankedData, summaryLength, judges)
        
        print("\n STRUCTURED SUMMARY \n")
        print(citation)
        print(respondent + " " + appellant)
        self.printJudges(judges)
        agreeJudges = self.prepareASMOData(casenum)
        self.createICLRSummary(agreeJudges, judges, majority, rankedData, summaryLength)
        print("\n")
        print("UKSC SUMMARY ")
        print("\n")
        self.createUKSCSummary(agreeJudges, judges, majority, rankedData, summaryLength)
        
        
      
    
    def printJudges(self, judges):
        print("HL: ", end="")
        for judge in judges:
            if judge != "NONE":
                print(" LORD " + judge, end="")
        
    def prepareASMOData(self, casenum):
        ASMOagree = []
        with open("data/UKHL_corpus/UKHL_" +casenum+ ".csv", "r") as infile:
            reader = csv.DictReader(infile)
            
            for row in reader:
                if row["role"] != "<new-case>":
                    if row["agree"] != "NONE" and row["agree"] != "no match":
                        ASMOagree.append([row["judge"].upper(), row["agree"].upper().split("+")])
                        
        return ASMOagree
        
    # get the UKHL data from corpus
    def summaryHeaderData(self, casenum):
        corpusList = ["2001Apr04eastbrn-1.ling.xml", "2001Dec13aib-1.ling.xml", "2001Dec13smith-1.ling.xml", "2001Feb08kuwait-1.ling.xml",
            "2001Feb08presto-1.ling.xml", "2001Jan18intern-1.ling.xml", "2001Jan31card-1.ling.xml", "2001Jul05m-1.ling.xml", "2001Jul12mcgra-1.ling.xml",
            "2001Jul12news-1.ling.xml", "2001Jul25dan-1.ling.xml", "2001Jun28norris-1.ling.xml", "2001Mar08mehann-1.ling.xml", "2001Mar22hallam-1.ling.xml",
            "2001May23daly-1.ling.xml", "2001May23liver-1.ling.xml", "2001Nov01moham-1.ling.xml", "2001Oct11uratem-1.ling.xml", "2001Oct25dela-1.ling.xml",
            "2002Apr18gersn-1.ling.xml", "2002Apr25cave-1.ling.xml", "2002Jul04graham-1.ling.xml", "2002Jul25robert-1.ling.xml", "2002Jul25sten-1.ling.xml",
            "2002Jun20pope-1.ling.xml", "2002Jun20wngton-1.ling.xml", "2002Jun27ash-1.ling.xml", "2002May16morgan-1.ling.xml", "2002May23burket-1.ling.xml",
            "2002Nov14byrne-1.ling.xml", "2002Nov25lich-1.ling.xml", "2002Oct31regina-1.ling.xml", "2003Apr03green-1.ling.xml", "2003Apr10bellin-1.ling.xml",
            "2003Apr10sage-1.ling.xml", "2003Feb20glaz-1.ling.xml", "2003Feb27diets-1.ling.xml", "2003Feb27inrep-1.ling.xml", "2003Jan30kanar-1.ling.xml",
            "2003Jan30regina-1.ling.xml", "2003Jul31moyna-1.ling.xml", "2003Jul31mulkrn-1.ling.xml", "2003Jun12kuwa-1.ling.xml", "2003Jun12lyon-1.ling.xml",
            "2003Mar20sepet-1.ling.xml", "2003Mar20sivak-1.ling.xml", "2003May22john-1.ling.xml", "2001Jan25montgo-1.ling.xml", "2001Jun28optid-1-NS.ling.xml", 
            "2001Dec17inreb-1-NS.ling.xml", "2001May23newfi-1-NS.ling.xml", "2002Jun20ni-1-NS.ling.xml", "2002Apr25cape-1-NS.ling.xml", "2002Oct17westmi-1.ling.xml", 
            "2002Feb20kahn-1-NS.ling.xml", "2002Jan24benja-1.ling.xml", "2002Feb28amin-1.ling.xml", "2002Jan24zeqiri-1-NS.ling.xml", "2002Jan24rezvi-1.ling.xml", 
            "2002Feb28esusse-1-NS.ling.xml", "2003Jul31thomsn-1-NS.ling.xml", "2003Jun26anuf-1-NS.ling.xml", "2003May08russ-1.ling.xml", "2003Apr03action-1.ling.xml", 
            "2003Jun26aston-1-NS.ling.xml", "2003Feb06shield-1-NS.ling.xml", "2003Jul31lloyds-1-NS.ling.xml", "2003Jul31giles-1-NS.ling.xml", "2003Jun26rus-1-NS.ling.xml"]
        caseList = ['1.19', '1.63', '1.68', 'NA', 
            '1.05', '1.02', '1.04', '1.35', '1.39', 
            '1.38', '1.42', '1.34', '1.11', '1.15', 
            '1.26', '1.28', '1.57', '1.43', '1.55', 
            '2.13', '2.18', '2.3', '2.35', '2.34', 
            '2.26', '2.24', '2.29', '2.21', '2.23', 
            '2.45', '2.47', '2.41', '3.18', '3.21', 
            '3.22', '3.07', '3.1', '3.08', '3.02', 
            'N/A', '3.44', '3.41', '3.31', '3.32', 
            '3.15', '3.14', '3.28', '1.03', '1.32', 
            '1.7', '1.27', '2.25', '2.16', 'NA', '2.06', 
            '2.02', '2.09', '2.03', '2.01', '2.08', '3.45',
            '3.36', '3.24', '3.17', '3.37', '3.03', '3.48', 
            '3.42', '3.38']
        index = caseList.index(casenum)    
        tree = ET.parse('./data/SUM_69_corpus/' + corpusList[index])
        root = tree.getroot()
        
        respondents = []
        appellants = []
    
        for elem in root.iter("case"):
            if len(elem):
                for subelem in elem:
                    if subelem.attrib.get("subtype") == "respondent":
                        respondents.append(subelem.text.replace("\n", " "))
                    if subelem.attrib.get("subtype") == "appellant":
                        appellants.append(subelem.text.replace("\n", " "))
                   
        # where it cant get the respondents and appellants
        if len(respondents) > 0 and len(appellants) > 0:
            return respondents[0], appellants[0]
        else:
            return "", ""
    
        
    def getSummaryData(self, casenum):
        with open('summarydata-spacy/UKHL_'+casenum+'_classification.csv', 'r') as infile:
            reader = csv.DictReader(infile)
            
            sentences = []
            for row in reader:
                sentid = row['sent_id']

                sentenceData = {
                    'sent id' : sentid,
                    'role' : row['rhet label'],
                    'relevant' : row['relevant'], 
                    'rank' : row['yes confidence']
                }
                sentences.append(sentenceData)
            
            return sentences
            
        
    def getSentences(self, casenum):
        caseFile = 'data/UKHL_corpus/UKHL_'+casenum+'.csv'
        
        with open(caseFile, 'r') as infile:
          reader = csv.DictReader(infile)
          sentences = []
          judges = []
          citation = []
          majorityJudges = []
          majority = []
         
          for row in reader:
              if row['role'] == "<new-case>":
                  citation.append(row['text'])
                  majorityJudges = row["agree"].split("+")
                  for i in range(len(majorityJudges)):
                        if majorityJudges[i].upper() != "NONE":
                            majority.append(majorityJudges[i].upper())
              newJudge = row['judge']
              sentid = row['sentence_id']
            
              sentence = {
                 'sent id' : sentid, 
                 'judge' : row['judge'].upper(), 
                 'text' : row['text']
                }
              sentences.append(sentence)
              if newJudge.upper() not in judges: 
                  judges.append(newJudge.upper())
          
          return sentences, judges, citation[0], majority
      
    def createRelevanceOnlySummary(self, MLData, sentences):
        relevantSentences = []
        summarySentences = []
        for data in MLData:
            
            if data['relevant'] == 'yes':
                relevantSentences.append(data['sent id'])
                
        for sentence in sentences:
            for sentId in relevantSentences:
                if sentence['sent id'] == sentId:
                    print(sentence['text']) 


    # this creates a summary based on top ranked - regardless of rhetorical role
    def createRankingOnlySummary(self, MLData, sentences, summaryLength):
        finalData = []
            
        for sentence in sentences:
            for mlData in MLData:
                if sentence['sent id'] == mlData['sent id']:
                    new = {'text' : sentence['text'], 
                        'judge' : sentence['judge']}
                    mlData.update(new)
                    finalData.append(mlData)
                    continue
        
        rankedList = sorted(finalData, key=lambda k: k['rank'], reverse=True)
        
        i = 0
        for sentence in rankedList:
            if i <= summaryLength:
                print((sentence['sent id']) + " : " + sentence['role'] + " : " + sentence['judge'] + " : " + sentence['text']) 
            else: 
                break
            i += 1
            
        
        return rankedList
    
    def createRankingandRhetSummary(self, rankedData, summaryLength, judges):
        #  rhetorical distribution for these summaries are based on the distribution provided by Hachey and Grover
        backgroundDist = 10.2
        framingDist = 30.0
        proceedingsDist = 18.4
        factDist = 10.3
        disposalDist = 31.1
        
        backgroundSentences = []
        framingSentences = []
        proceedingsSentences = []
        factSentences = []
        disposalSentences = []
        
        for sentence in rankedData: 
            if sentence['role'] == '4.0': 
                backgroundSentences.append(sentence)
            elif sentence['role'] == '5.0':
                framingSentences.append(sentence)
            elif sentence['role'] == '3.0': 
                proceedingsSentences.append(sentence)
            elif sentence['role'] == '2.0': 
                factSentences.append(sentence)
            elif sentence['role'] == '6.0':
                disposalSentences.append(sentence)
                
        backgroundNum = self.prepareDistribution(backgroundDist, summaryLength)
        framingNum = self.prepareDistribution(framingDist, summaryLength)
        proceedingsNum = self.prepareDistribution(proceedingsDist, summaryLength)
        factNum = self.prepareDistribution(factDist, summaryLength)
        disposalNum = self.prepareDistribution(disposalDist, summaryLength)
        
        
        print("\n SUMMARY BASED ON TOP " + str(summaryLength) + " SENTENCES, ORDERED BY RHET ROLE.\n")
        
        factJudgeList = self.printRhetSentences(factSentences, factNum, "FACT")
        proceedingsJudgeList = self.printRhetSentences(proceedingsSentences, proceedingsNum, "PROCEEDINGS")
        backgroundJudgeList = self.printRhetSentences(backgroundSentences, backgroundNum, "BACKGROUND")
        framingJudgeList = self.printRhetSentences(framingSentences, framingNum, "FRAMING")
        disposalJudgeList = self.printRhetSentences(disposalSentences, disposalNum, "DISPOSAL")

        print("\n SUMMARY BASED ON TOP " + str(summaryLength) + " SENTENCES, ORDERED BY JUDGE UNDER RHET ROLE.\n")
        self.printRhetRankedbyJudge(factJudgeList, "FACT", judges)
        self.printRhetRankedbyJudge(proceedingsJudgeList, "PROCEEDINGS", judges)
        self.printRhetRankedbyJudge(backgroundJudgeList, "BACKGROUND", judges)
        self.printRhetRankedbyJudge(framingJudgeList, "FRAMING", judges)
        self.printRhetRankedbyJudge(disposalJudgeList, "DISPOSAL", judges)
        
    def prepareDistribution(self, dist, summaryLength):
        distribution = summaryLength / 100
        distribution = distribution * dist
        sentences = round(distribution)
        return sentences
    
    def printRhetSentences(self, sentences, dist, name):
        i = 0
        sentenceToPrint = []
        print(name)
        
        for sentence in sentences:
            if i <= dist:
                sentenceUpdate = {
                    'sent id' : sentence['sent id'], 
                    'judge' : sentence['judge'], 
                    'text' : sentence['text']
                }
                sentenceToPrint.append(sentenceUpdate)

            else: 
                break
            i += 1
            
        rankedList = sorted(sentenceToPrint, key=lambda k: k['judge'], reverse=True)
        return rankedList
    
    def printRhetRankedbyJudge(self, sentences, name, judges):
        print("\n" + name)
        for sentence in sentences: 
            print(sentence['text']) 
                    
        
    # print majority (these are the key judges), print the judges, agreeJudges have the jduges who agreed
    def createICLRSummary(self, agreeJudges, judges, majority, rankedData, sentencesNum):
        backgroundSentences, framingSentences, proceedingsSentences, factSentences, disposalSentences = self.getRankedSentbyRole(rankedData)
        # this will get the appropriate distribution, top ranked Fact from all judges
        fact, pro = self.getICLRFactandProceedingsDistribution(rankedData, sentencesNum, factSentences, proceedingsSentences, False)
    
        print("\n")
        for sentence in fact: 
            print(sentence['text'], end="")
        for sentence in pro: 
            print(sentence['text'], end="")
        print("\n")
        
        # this function just gets the Fact sentences from the majority judges
   #     self.writeFactsParagraph(majority, factSentences)
        self.writeJudgmentParagraph(majority, agreeJudges,  backgroundSentences, framingSentences, 
                                    disposalSentences, judges, sentencesNum, rankedData)
                
    def getICLRFactandProceedingsDistribution(self, rankedData, summaryLength, factSentences, proceedingsSentences, UKSC):
        #  rhetorical distribution for these summaries are based on the distribution provided by Hachey and Grover

        if UKSC == False: 
    
            factDist = 10.3
            proceedingsDist = 18.4
            i = 0 
            factSentenceToPrint = []
            proSentenceToPrint = []
                    
            distribution = summaryLength / 100
            distribution = distribution * factDist
            fact_Dist = round(distribution)
            
            distribution = summaryLength / 100
            distribution = distribution * proceedingsDist
            proceedings_Dist = round(distribution)
        
        if UKSC == True: 
            factDist = 15
            proceedingsDist = 15
            i = 0 
            factSentenceToPrint = []
            proSentenceToPrint = []
                    
            distribution = 30 / 100
            distribution = distribution * factDist
            fact_Dist = round(distribution)
            
            distribution = 30 / 100
            distribution = distribution * proceedingsDist
            proceedings_Dist = round(distribution)

     
        for sentence in factSentences:
            if i < fact_Dist:
                sentId = int(sentence['sent id'])
                sentenceUpdate = {
                    'sent id' : sentId, 
                    'judge' : sentence['judge'], 
                    'text' : sentence['text'],
                    'rank' : sentence['rank']
                }
                factSentenceToPrint.append(sentenceUpdate)
            else: 
                break
            i += 1
        i = 0
        for sentence in proceedingsSentences:
            if i < proceedings_Dist:
                sentId = int(sentence['sent id'])
                
                sentenceUpdate = {
                    'sent id' : sentId, 
                    'judge' : sentence['judge'], 
                    'text' : sentence['text'], 
                    'rank' : sentence['rank']
                }
                proSentenceToPrint.append(sentenceUpdate)
            else: 
                break
            i += 1
            

      #  rankedList = sentenceToPrint
     #   factSentenceToPrint = sorted(factSentenceToPrint, key=lambda k: k['sent id'], reverse=True)
     #   proSentenceToPrint = sorted(proSentenceToPrint, key=lambda k: k['sent id'], reverse=True)
        return factSentenceToPrint, proSentenceToPrint
    
    def getRankedSentbyRole(self, rankedData):
        backgroundSentences = []
        framingSentences = []
        proceedingsSentences = []
        factSentences = []
        disposalSentences = []
        
        for sentence in rankedData: 
            if sentence['role'] == '4.0': 
                backgroundSentences.append(sentence)
            elif sentence['role'] == '5.0':
                framingSentences.append(sentence)
            elif sentence['role'] == '3.0': 
                proceedingsSentences.append(sentence)
            elif sentence['role'] == '2.0': 
                factSentences.append(sentence)
            elif sentence['role'] == '6.0':
                disposalSentences.append(sentence)
                
        return backgroundSentences, framingSentences, proceedingsSentences, factSentences, disposalSentences
    
    def writeFactsParagraph(self, majority, factsSentences):
        newFacts = self.getJudgeSentences(majority, factsSentences)

        print("\n")
        for sentence in newFacts: 
      
            print(sentence['text'], end="")
        print("\n")
        
    def writeJudgmentParagraph(self, majority, agreeJudges,  backgroundSentences, framingSentences, disposalSentences, judges, length, rankedData):
        backgroundDist = 10.2
        distribution = length / 100
        distribution = distribution * backgroundDist
        backgroundDist = round(distribution)
        framingDist = 30.0
        distribution = length / 100
        distribution = distribution * framingDist
        framingDist = round(distribution)
        disposalDist = 31.1
        distribution = length / 100
        distribution = distribution * disposalDist
        disposalDist = round(distribution)
        
        
        self.getOutcome(judges, rankedData, majority, agreeJudges)
        
        
        # before going through line of reasoning b/w judges we print the outcome statements first
        if len(majority) > 0:
            print("The line of reasoning forming the majority opinion was delivered by ", end= "")
            majorityMultiple = False
            for judge in majority: 
                if majorityMultiple == True: 
                    print(" and ", end = "")
                print("LORD " + judge, end="")
                majorityMultiple = True
            print(". ", end="")

            for judge in majority: 
                sentences = self.createMajorityStatements(judge, backgroundSentences, framingSentences, disposalSentences,  
                                                          backgroundDist, framingDist, disposalDist)
                self.printMajorityOpinions(sentences, judge)

        # manually check for a majority first just in case then back up for no majority 
        else:
            possibleMajority = []
            for lord in agreeJudges: 
                multiple_agree = False
                for judge in judges: 
                    if judge != "NONE":
                        if judge not in majority:
                            if lord[0] in judge:
                                    if len(lord[1]) > 1: 
                                        for i in range(len(lord[1])): 
                                            for jdg in judges: 
                                                if lord[1][i] in jdg: 
                                                    possibleMajority.append(jdg)
                                    else:
                                        possibleMajority.append(lord[1][0])
            judges.remove("NONE")
            major = round(len(judges)/2)
          
            for jdg in judges:
                count = 0
                for judge in possibleMajority: 
                    if judge == jdg:
                        count += 1
                    elif judge == "ALL":
                        count += 1
                if count >= major:
                    if judge not in majority:
                        majority.append(judge)
            
            # majority not identified by the ASMO system, here check manually 
            if (len(majority)) > 0:
                print("The majority of judges agreed to the reasoning delivered by ", end= "")
                majorityMultiple = False
                for judge in majority: 
                    if majorityMultiple == True: 
                        print(" and ", end = "")
                    print("LORD " + judge, end="")
                    majorityMultiple = True
                print(". ", end="")
                
    
                for judge in majority: 
                    sentences = self.createMajorityStatements(judge, backgroundSentences, framingSentences, disposalSentences,  
                                                              backgroundDist, framingDist, disposalDist)
                    self.printMajorityOpinions(sentences, judge)
              
        
    
        # checks to see if any of the judges agreed with all of the fellow judges
        allJudges = []
        for lord in agreeJudges: 
            multiple_agree = False
            for judge in judges: 
                if judge != "NONE":
                    if judge not in majority:
                        if lord[0] in judge:
                            if(lord[1][0] == "ALL"):
                                allJudges.append(judge)
                                
        for judge in allJudges: 
            print("LORD " + judge + " delivered an opinion agreeing with the reasoning of all of his fellow Lords. ", end="")
                     
                
        
        agreements = {}
        for lord in agreeJudges: 
            multiple_agree = False
            for judge in judges: 
                if judge != "NONE":
                    if judge not in majority:
                        if judge not in allJudges:
                            if lord[0] in judge:
                                    currAgreements = []
                             
                                    if len(lord[1]) > 1: 
                                        for i in range(len(lord[1])): 
                                       
                                            if lord[1][i] == "SELF":
                                                print("LORD " + judge + " emphasised the importance of his own reasoning.", end="")
                                            if lord[1][0] == "ALL":
                                                print("LORD " + judge + " agreed with all of his fellow Lords", end="")
                                            else: 
                                                for jdg in judges: 
                                                    if lord[1][i] in jdg: 
                                                        currAgreements.append(lord[1][i])
                                                        if jdg == "ALL":
                                                            print("LORD " + judge +  " all of his fellow Lords", end="")
                                                   
                                            multiple_agree = True
                                    else:
                                        if(lord[1][0] == "ALL"):
                                             print("LORD " + judge + " all of his fellow Lords", end = "")
                                        else:
                                    
                                            newlord = lord[1][0]
                                            currAgreements.append(newlord)
                           
                                    sortedAgreements = sorted(currAgreements) # to sort the judges
                                    judgeStr = ""
                                    multiple = False
                                    if(len(sortedAgreements)) > 1: 
                                        multiple = True
                                    first = True
                                    for judgee in sortedAgreements: 
                                        if multiple == True and first == False: 
                                            newStr = " and LORD " + judgee
                                            judgeStr = judgeStr + newStr
                                        else: 
                                            newStr = "LORD " + judgee
                                            judgeStr = newStr
                                        first= False
                                    if judgeStr not in agreements: 
                                        val = "LORD " + judge
                                        agreements.update({judgeStr : val})
                                    elif judgeStr in agreements:
                                        oldStr = agreements.get(judgeStr)
                                        newStr = oldStr + " and LORD " + judge
                                        agreements.update({judgeStr : newStr})
        # to not duplicate the sentences
        for agreeTo, agreeFrom in agreements.items():
            print(agreeFrom + " delivered an opinion agreeing with " + agreeTo + ". ", end = "")
            
        agreeJdgs = []
        for lord in agreeJudges: 
            agreeJdgs.append(lord[0])
        
        disagreeJdgs = []
        for judge in judges: 
            if judge != "NONE":
                if judge not in agreeJdgs and judge not in majority: 
                    disagreeJdgs.append(judge)
           
        if len(disagreeJdgs) > 0:
            for disJudge in disagreeJdgs:
                print("LORD " + disJudge + " did not agree with the line of reasoning of his fellow Lords. He said: \" ", end="")
                self.printDisposalSentences(disJudge, disposalSentences,  disposalDist)
            print("\"")
            
    def printDisposalSentences(self, judge, sentences, disposalDist):
        i = 0 
        for sentence in sentences: 
            if sentence['judge'] == judge:
                if i < 2:
                    print(sentence['text'], end="")
                i+=1
            
            
    def printMajorityOpinions(self, sentences, judge):
        y = 0
        
        if(len(sentences) > 0):
            print("LORD " + judge + " said: \"", end="")
            for sentence in sentences: 
                print(sentence['text'], end="")
                y = y + 1
                if y == 5: 
                    print("\n")
                    y = 0
                
            print("\"", end="")
            print("\n")
            
     # get top ranked for each data type     
    def createMajorityStatements(self, judge, backgroundSentences, framingSentences, disposalSentences, 
                                 backgroundDist, framingDist, disposalDist):
        newData = []
        background = 0 
        framing = 0 
        disposal = 0

        for sentence in backgroundSentences: 
           if background < backgroundDist:
               if(sentence['judge'] == judge): 
                     sentId = sentence['sent id']
                     sentId = int(sentId)
                     currSent = {
                         'text' : sentence['text'], 
                         'sent id' : sentId
                         }
                     newData.append(currSent)

           else: 
               continue
           background += 1
        for sentence in framingSentences: 
            if framing < framingDist: 
               if(sentence['judge'] == judge): 
                     sentId = sentence['sent id']
                     sentId = int(sentId)
                     currSent = {
                         'text' : sentence['text'], 
                         'sent id' : sentId
                         }
                     newData.append(currSent)
            else: 
                continue
            framing += 1
        for sentence in disposalSentences: 
            if disposal < disposalDist:
               if(sentence['judge'] == judge): 
                     sentId = sentence['sent id']
                     sentId = int(sentId)
                     currSent = {
                         'text' : sentence['text'], 
                         'sent id' : sentId
                         }
                     newData.append(currSent)
            else: 
                 continue
            disposal += 1
            
        rankedList = newData
    #    rankedList = sorted(newData, key=lambda i: i['sent id'], reverse=True)
        return rankedList
    
    
    def getJudgeSentences(self, judges, sentences):
        i = 0 
        newData = []
        
        for sentence in sentences: 
            if i <= 5:
                for judge in judges:
                    if(sentence['judge'] == judge): 
                        sentId = sentence['sent id']
                        sentId = int(sentId)
                        currSent = {
                            'text' : sentence['text'], 
                            'sent id' : sentId
                            }
                        newData.append(currSent)
            i += 1
        
        
        rankedList = sorted(newData, key=lambda i: i['sent id'], reverse=True)
        return rankedList
    
    def getOutcome(self, judges, rankedData, majority, agreeJudges): 
        disposalSentences = []
        outcomeSentences = []
        
        for sentence in rankedData: 
            if sentence['role'] == '6.0': 
                disposalSentences.append(sentence)
                
        for disposal in disposalSentences: 
            for judge in majority:
                if disposal['judge'] == judge: 
                    outcomeSentences.append(disposal['text'])
        
# =============================================================================
#         
#         if not outcomeSentences: 
#             for disposal in disposalSentences: 
#                 for judge in agreeJudges:
#                     if disposal['judge'] == judge: 
#                         outcomeSentences.append(disposal['text'])
# =============================================================================
                        
 
       # outcomeSentences these are all top ranked disposal sentences ideally with the outcome

        self.parseOutcome(outcomeSentences)
        
        
        # need to account for this being empty next w/ using the agreement judges
        
    def parseOutcome(self, outcomeSentences): 
        dismissFragments = ['I would dismiss the appeal', 'should be dismissed', 'I would dismiss', 'would dismiss the appeal', 'would therefore dismiss the appeal', 'refuse the appeal', 'dismiss the appeal']
        allowFragments = ['I would allow the appeal', 'would allow the appeal', 'too would allow the appeal', 'appeal should be allowed', 'allow the appeal', 'would allow this appeal']
        
        
        for fragment in dismissFragments: 
            check = any(fragment in string for string in outcomeSentences)    

            if check is True: 
                print("The appeal was dismissed.")
                break

        
     
        for fragment in allowFragments: 
            check = any(fragment in string for string in outcomeSentences)  
        
            if check is True: 
                print("The appeal was allowed.")
                break
            
    def createUKSCSummary(self, agreeJudges, judges, majority, rankedData, sentencesNum):
        backgroundSentences, framingSentences, proceedingsSentences, factSentences, disposalSentences = self.getRankedSentbyRole(rankedData)
        # this will get the appropriate distribution, top ranked Fact from all judges
        fact, pro = self.getICLRFactandProceedingsDistribution(rankedData, sentencesNum, factSentences, proceedingsSentences, True)
    
        print("\n")
        
        print("BACKGROUND TO THE APPEAL")
    
        print("\n")
        for sentence in fact: 
            print(sentence['text'], end="")
            
        print("\n")
            
        for sentence in pro: 
            print(sentence['text'], end="")
        print("\n")
        
        # this function just gets the Fact sentences from the majority judges
   #     self.writeFactsParagraph(majority, factSentences)
        self.writeUKSCJudgmentParagraph(majority, agreeJudges,  backgroundSentences, framingSentences, 
                                    disposalSentences, judges, sentencesNum, rankedData)
        
        
    def writeUKSCJudgmentParagraph(self, majority, agreeJudges,  backgroundSentences, framingSentences, disposalSentences, judges, length, rankedData):
        length = 30
        backgroundDist = 10.2
        distribution = length / 100
        distribution = distribution * backgroundDist
        backgroundDist = round(distribution)
        framingDist = 30.0
        distribution = length / 100
        distribution = distribution * framingDist
        framingDist = round(distribution)
        disposalDist = 31.1
        distribution = length / 100
        distribution = distribution * disposalDist
        disposalDist = round(distribution)
        
        print("\n JUDGMENT \n")
        
        
        self.getOutcome(judges, rankedData, majority, agreeJudges)
        
        
        
        
        # before going through line of reasoning b/w judges we print the outcome statements first
        if len(majority) > 0:
            print("The line of reasoning forming the majority opinion was delivered by ", end= "")
            majorityMultiple = False
            for judge in majority: 
                if majorityMultiple == True: 
                    print(" and ", end = "")
                print("LORD " + judge, end="")
                majorityMultiple = True
            print(". ", end="")

            

        # manually check for a majority first just in case then back up for no majority 
        else:
            possibleMajority = []
            for lord in agreeJudges: 
                multiple_agree = False
                for judge in judges: 
                    if judge != "NONE":
                        if judge not in majority:
                            if lord[0] in judge:
                                    if len(lord[1]) > 1: 
                                        for i in range(len(lord[1])): 
                                            for jdg in judges: 
                                                if lord[1][i] in jdg: 
                                                    possibleMajority.append(jdg)
                                    else:
                                        possibleMajority.append(lord[1][0])
            judges.remove("NONE")
            major = round(len(judges)/2)
          
            for jdg in judges:
                count = 0
                for judge in possibleMajority: 
                    if judge == jdg:
                        count += 1
                    elif judge == "ALL":
                        count += 1
                if count >= major:
                    if judge not in majority:
                        majority.append(judge)
            
            # majority not identified by the ASMO system, here check manually 
            if (len(majority)) > 0:
                print("The majority of judges agreed to the reasoning delivered by ", end= "")
                majorityMultiple = False
                for judge in majority: 
                    if majorityMultiple == True: 
                        print(" and ", end = "")
                    print("LORD " + judge, end="")
                    majorityMultiple = True
                print(". ", end="")
                
    
              
        
    
        # checks to see if any of the judges agreed with all of the fellow judges
        allJudges = []
        for lord in agreeJudges: 
            multiple_agree = False
            for judge in judges: 
                if judge != "NONE":
                    if judge not in majority:
                        if lord[0] in judge:
                            if(lord[1][0] == "ALL"):
                                allJudges.append(judge)
                                
        for judge in allJudges: 
            print("LORD " + judge + " delivered an opinion agreeing with the reasoning of all of his fellow Lords. ", end="")
                     
                
        
        agreements = {}
        for lord in agreeJudges: 
            multiple_agree = False
            for judge in judges: 
                if judge != "NONE":
                    if judge not in majority:
                        if judge not in allJudges:
                            if lord[0] in judge:
                                    currAgreements = []
                             
                                    if len(lord[1]) > 1: 
                                        for i in range(len(lord[1])): 
                                       
                                            if lord[1][i] == "SELF":
                                                print("LORD " + judge + " emphasised the importance of his own reasoning.", end="")
                                            if lord[1][0] == "ALL":
                                                print("LORD " + judge + " agreed with all of his fellow Lords", end="")
                                            else: 
                                                for jdg in judges: 
                                                    if lord[1][i] in jdg: 
                                                        currAgreements.append(lord[1][i])
                                                        if jdg == "ALL":
                                                            print("LORD " + judge +  " all of his fellow Lords", end="")
                                                   
                                            multiple_agree = True
                                    else:
                                        if(lord[1][0] == "ALL"):
                                             print("LORD " + judge + " all of his fellow Lords", end = "")
                                        else:
                                    
                                            newlord = lord[1][0]
                                            currAgreements.append(newlord)
                           
                                    sortedAgreements = sorted(currAgreements) # to sort the judges
                                    judgeStr = ""
                                    multiple = False
                                    if(len(sortedAgreements)) > 1: 
                                        multiple = True
                                    first = True
                                    for judgee in sortedAgreements: 
                                        if multiple == True and first == False: 
                                            newStr = " and LORD " + judgee
                                            judgeStr = judgeStr + newStr
                                        else: 
                                            newStr = "LORD " + judgee
                                            judgeStr = newStr
                                        first= False
                                    if judgeStr not in agreements: 
                                        val = "LORD " + judge
                                        agreements.update({judgeStr : val})
                                    elif judgeStr in agreements:
                                        oldStr = agreements.get(judgeStr)
                                        newStr = oldStr + " and LORD " + judge
                                        agreements.update({judgeStr : newStr})
        # to not duplicate the sentences
        for agreeTo, agreeFrom in agreements.items():
            print(agreeFrom + " delivered an opinion agreeing with " + agreeTo + ". ", end = "")
            
        agreeJdgs = []
        for lord in agreeJudges: 
            agreeJdgs.append(lord[0])
        
        disagreeJdgs = []
        for judge in judges: 
            if judge != "NONE":
                if judge not in agreeJdgs and judge not in majority: 
                    disagreeJdgs.append(judge)
           
        if len(disagreeJdgs) > 0:
            for disJudge in disagreeJdgs:
                print("LORD " + disJudge + " did not agree with the line of reasoning of his fellow Lords.", end="")
            
        print("\n")
        
        print("REASONS FOR THE JUDGMENT")
        
        print("\n")
            
        for judge in majority: 
                    sentences = self.createMajorityStatements(judge, backgroundSentences, framingSentences, disposalSentences,  
                                                              backgroundDist, framingDist, disposalDist)
                    self.printMajorityOpinions(sentences, judge)

        
            
                
        