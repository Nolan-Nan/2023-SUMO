
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
updated entities file, for correct parsing of entities (can be used in automatic
text markup - for new and unseen data)

nb - just running this for the blackstone() function will generate a file that contains
the entities of each sentence in the corpus - combining this info with the corpus allows


@author: amyconroy
"""

import csv
import numpy as np
import spacy

# go through and open up the file, pass in the case id and sentence id, get the sentence from the file
            # check if the update contains any of the new entities recognized - new feature that way
# starting by adding the citation REGEX for UKHL judgments
def new_citation_feature(text):
   from spacy.matcher import Matcher

   nlp = spacy.load('en')
   matcher = Matcher(nlp.vocab)

   pattern = [{"IS_BRACKET": True, "OP": "?"}, {"SHAPE": "dddd"}, {"IS_BRACKET": True, "OP": "?"},
   {"TEXT": {"REGEX": "^[A-Z]"}, "OP": "?"}, {"LIKE_NUM": True}]

   matcher.add("citation", None, pattern)
   doc = nlp(text)



   matches = matcher(doc)
   #print(matches)
   if matches:
        # print("TRUE : " + text)
        return True # matching UKHL citation pattern - TRUE

   else:
        return False

# syntax for this bit will be:
        # casenames: pattern = TEXT v TEXT / In Re TEXT / in the TEXT case
        # {"TEXT": {"REGEX": "^[A-Z]"}} , {"TEXT": {"REGEX": "[Vv]"}} , {"TEXT": {"REGEX": "^[A-Z]"}}
        # {"TEXT": {"REGEX": "^[Ii](\.?|n)$"}}

def new_casename_feature(text):
   from spacy.matcher import Matcher

   nlp = spacy.load('en')
   matcher = Matcher(nlp.vocab)

   # for example conroy v conroy
   versus = [{"TEXT": {"REGEX": "[(([A-Z]('[A-Z]|[a-z][A-Z])?[a-z]+[A-Z]?|&)\s)+(v\s)(([A-Z]('[A-Z]|[a-z][A-Z])?[a-z]+[A-Z]?|&)\s)+]"}}]
   #inRe = [{"CASEINSENSITIVE": "in"} , {"CASEINSENSITIVE": "re"} , {"TEXT": {"REGEX": "[\w+]"}}]
   #inThe = [{"CASEINSENSITIVE": "in"} , {"CASEINSENSITIVE": "the"} , {"TEXT": {"REGEX": "[\w+]"}}, {"CASEINSENSITIVE": "case"}]

   matcher.add("caseVcase", None, versus)
   #matcher.add("inReCase", None, inRe)
   #matcher.add("inTheCase", None, inThe)

   doc = nlp(text)

   matches = matcher(doc)
   #print(matches)
   if matches:
        print("TRUE CASE NAME : " + text)
        return True # matching UKHL citation pattern - TRUE

   else:
        return False


# Entity Ruler
def new_legal_entities():
    from spacy.matcher import Matcher
    from spacy.pipeline import EntityRuler

    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)
    print("Match Legal Entities")

    ruler = EntityRuler(nlp)
    patterns = [{"label": "ORG", "pattern": "MyCorp Inc."}]
    ruler.add_patterns(patterns)
    nlp.add_pipe(ruler)

def new_normal_entities():
    from spacy.matcher import Matcher

    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)

    print("Match Normal Entities")

    normalentities_match = np.array([])

    with open('data/UKHL_corpus.csv', 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            doc = nlp(row['text'])
            matches = matcher(doc)
         #   print([(ent.text, ent.label_) for ent in doc.ents])
            if matches:
                normalentities_match = np.append(normalentities_match, [1])
            else:
                normalentities_match = np.append(normalentities_match, [0])

    print(normalentities_match)

    with open('UKHL_corpus_newnrment.csv', 'w', newline='') as outfile:
        fieldnames = ['data']
        writer = csv.DictWriter(outfile, fieldnames= fieldnames)
        writer.writeheader()

     #   print(len(normalentities_match))

        for v in range(len(normalentities_match)):
            writer.writerow({'data': normalentities_match[v]})


def blackstone():
    import spacy
    # Load the model
    nlp = spacy.load("en_blackstone_proto")

    provision_ent_X = np.array([])
    instrument_ent_X = np.array([])
    court_ent_X = np.array([])
    casename_ent_X = np.array([])
    citation_ent_X = np.array([])
    judge_ent_X = np.array([])
    casenum_X = np.array([])
    sentid_X = np.array([])
    y = 0

# Iterate through the entities identified by the model
    with open('data/UKHL_corpus.csv', 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            y += 1
       #     print("Y")
       #     print(y)
            sentid_X = np.append(sentid_X, row['sentence_id'])
            casenum_X = np.append(casenum_X, row['case_id'])
            text = row['text']
            doc = nlp(text)
       #     print([(ent.text, ent.label_) for ent in doc.ents])
            label = [(ent.label_) for ent in doc.ents]
       #     print("LABEL ARRAY")
       #     print(label)

            provision_flag = False
            court_flag = False
            case_flag = False
            judge_flag = False
            instrument_flag = False
            citation_flag = False

            for v in range(len(label)):
                lbl = label[v]
           #     print("LABELS")
          #      print(lbl)
                if lbl == 'PROVISION':
                    provision_flag = True
                   # provision_ent_X = np.append(provision_ent_X, [1])
               # else:
                   # provision_flag = False
                   # provision_ent_X = np.append(provision_ent_X, [0])
                if lbl == 'COURT':
                    court_flag = True
                   # court_ent_X = np.append(court_ent_X, [1])
               # else:
                   # court_flag = False
                   # court_ent_X = np.append(court_ent_X, [0])
                if lbl == 'CASENAME':
                    case_flag = True
                  #  casename_ent_X = np.append(casename_ent_X, [1])
                #else:
                  #  case_flag = False
                #    casename_ent_X = np.append(casename_ent_X, [0])
                if lbl == 'JUDGE':
                    judge_flag = True
                   # judge_ent_X = np.append(judge_ent_X, [1])
               # else:
                  #  judge_flag = False
                   # judge_ent_X = np.append(judge_ent_X, [0])
                if lbl == 'INSTRUMENT':
                    instrument_flag = True
                   # instrument_ent_X = np.append(instrument_ent_X, [1])
               # else:
                  #  instrument_flag = False
                  #  instrument_ent_X = np.append(instrument_ent_X, [0])
                if lbl == 'CITATION':
                    citation_flag = True
                   # citation_ent_X = np.append(citation_ent_X, [1])
               # else:
                  #  citation_flag = False
                   # citation_ent_X = np.append(citation_ent_X, [0])

            if provision_flag:
                provision_ent_X = np.append(provision_ent_X, [1])
            else:
                provision_ent_X = np.append(provision_ent_X, [0])
            if court_flag:
                court_ent_X = np.append(court_ent_X, [1])
            else:
                court_ent_X = np.append(court_ent_X, [0])
            if case_flag:
                casename_ent_X = np.append(casename_ent_X, [1])
            else:
                casename_ent_X = np.append(casename_ent_X, [0])
            if judge_flag:
                judge_ent_X = np.append(judge_ent_X, [1])
            else:
                judge_ent_X = np.append(judge_ent_X, [0])
            if instrument_flag:
                instrument_ent_X = np.append(instrument_ent_X, [1])
            else:
                instrument_ent_X = np.append(instrument_ent_X, [0])
            if citation_flag:
                citation_ent_X = np.append(citation_ent_X, [1])
            else:
                citation_ent_X = np.append(citation_ent_X, [0])
   # print(normalentities_match)
  #  return provision_ent_X, instrument_ent_X, court_ent_X, casename_ent_X, citation_ent_X, judge_ent_X
   # return provision_flag, instrument_flag, court_flag, case_flag, citation_flag, judge_flag

    with open('data/EntitiesCorpus.csv','w', newline='')as outfile:
        fieldnames = ['provision ent', 'instrument ent', 'court ent', 'case name ent', 'citation bl ent', 'judge ent']

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

       # print("Y TOTAL")
       # print(range(y))

        for v in range(y):
        #    print(provision_ent_X[v])
        #    print(instrument_ent_X[v])
        #    print(court_ent_X[v])
        #    print(casename_ent_X[v])
        #    print(citation_ent_X[v])
        #    print(judge_ent_X[v])
            writer.writerow({ 'provision ent' : provision_ent_X[v], 'instrument ent' : instrument_ent_X[v], 'court ent' : court_ent_X[v],
            'case name ent' : casename_ent_X[v], 'citation bl ent' : citation_ent_X[v], 'judge ent' : judge_ent_X[v]})

def ner():
    import spacy

    nlp = spacy.load("en_core_web_sm")

    loc_ent_X = np.array([])
    org_ent_X = np.array([])
    date_ent_X = np.array([])
    person_ent_X = np.array([])
    time_ent_X = np.array([])
    gpe_ent_X = np.array([])
    fac_ent_X = np.array([])
    ordinal_ent_X = np.array([])

    y = 0

# Iterate through the entities identified by the model
    with open('data/UKHL_corpus.csv', 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            y += 1
         #   print("Y")
        #    print(y)
            text = row['text']



            doc = nlp(text)
          #  print([(ent.text, ent.label_) for ent in doc.ents])
            label = [(ent.label_) for ent in doc.ents]
          #  print("LABEL ARRAY")
          #  print(label)

            loc_flag = False
            org_flag = False
            date_flag = False
            person_flag = False
            time_flag = False
            gpe_flag = False
            fac_flag = False
            ordinal_flag = False

            for v in range(len(label)):
                lbl = label[v]
              #  print("LABELS")
              #  print(lbl)
                if lbl == 'LOC':
                    loc_flag = True
                if lbl == 'ORG':
                    org_flag = True
                if lbl == 'DATE':
                    date_flag = True
                if lbl == 'PERSON':
                    person_flag = True
                if lbl == 'TIME':
                    time_flag = True
                if lbl == 'GPE':
                    gpe_flag = True
                if lbl == 'FAC':
                    fac_flag = True
                if lbl == 'ORDINAL':
                    ordinal_flag = True

            if loc_flag:
                loc_ent_X = np.append(loc_ent_X, [1])
            else:
                loc_ent_X = np.append(loc_ent_X, [0])
            if org_flag:
                org_ent_X = np.append(org_ent_X, [1])
            else:
                org_ent_X = np.append(org_ent_X, [0])
            if date_flag:
                date_ent_X = np.append(date_ent_X, [1])
            else:
                date_ent_X = np.append(date_ent_X, [0])
            if person_flag:
                person_ent_X = np.append(person_ent_X, [1])
            else:
                person_ent_X = np.append(person_ent_X, [0])
            if time_flag:
                time_ent_X = np.append(time_ent_X, [1])
            else:
                time_ent_X = np.append(time_ent_X, [0])
            if gpe_flag:
                gpe_ent_X = np.append(gpe_ent_X, [1])
            else:
                gpe_ent_X = np.append(gpe_ent_X, [0])
            if fac_flag:
                fac_ent_X = np.append(fac_ent_X, [1])
            else:
                fac_ent_X = np.append(fac_ent_X, [0])
            if ordinal_flag:
                ordinal_ent_X = np.append(ordinal_ent_X, [1])
            else:
                ordinal_ent_X = np.append(ordinal_ent_X, [0])



    with open('data/nerCorpus.csv','w', newline='')as outfile:
        fieldnames = ['loc ent', 'org ent', 'date ent', 'person ent', 'time ent', 'gpe ent', 'fac ent', 'ordinal ent']
        fieldnames = ['ent']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

     #   print("Y TOTAL")
    #    print(range(y))

        for v in range(y):
           writer.writerow({'loc ent' : loc_ent_X[v], 'org ent' : org_ent_X[v], 'date ent' : date_ent_X[v],
                             'person ent' : person_ent_X[v], 'time ent' : time_ent_X[v], 'gpe ent' : gpe_ent_X[v],
                             'fac ent' : fac_ent_X[v], 'ordinal ent' : ordinal_ent_X[v]})


def all_ner():
    import spacy

    nlp = spacy.load("en_core_web_sm")

    y = 0
    all_ner_X = np.array([])

# Iterate through the entities identified by the model
    with open('data/UKHL_corpus.csv', 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            y += 1
            text = row['text']

            doc = nlp(text)
          #  print([(ent.text, ent.label_) for ent in doc.ents])
            label = [(ent.label_) for ent in doc.ents]

            entitylbl = False

            for v in range(len(label)):
                lbl = label[v]
             #   print("LABELS")
            #    print(lbl)
                if lbl != 'LAW':
                    entitylbl = True

            if entitylbl:
                all_ner_X = np.append(all_ner_X, [1])
            else:
                all_ner_X = np.append(all_ner_X, [0])


    with open('data/total_nerCorpus.csv','w', newline='')as outfile:
        fieldnames = ['ent']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

     #   print("Y TOTAL")
     #   print(range(y))

        for v in range(y):
           writer.writerow({'ent' : all_ner_X[v]})


# do this first
# new_normal_entities()
    # for quick testing

blackstone()
ner()
all_ner()
