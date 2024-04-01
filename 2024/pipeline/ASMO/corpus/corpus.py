"""
Corpus class handles the annotated corpus stored at www.holj.ml
the corpus is accessed via SSH through paramiko library and stored
as pandas dataframe.
"""

import paramiko
import os
import ast
import random
import pandas as pd
from random import shuffle

from collections import Counter

from .storage import save_data, load_data
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

class Corpus:

    def __init__(self, user, MJ_size, update):
        self.user = user # SSH details
        self.MJ_size = MJ_size # Size of majority judgement corpus
        self.update = update # If True, downloads fresh data
        self.golden = None

    def ssh_download(self):
        """
        Connects to the server, downloads annotations.
        Returns them as df.
        """

        # Open FTP connection
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy()) #NOTE security risk!!!
        ssh.connect(self.user.get_ip(), username=self.user.get_user(), key_filename=self.user.get_key())

        ftp = ssh.open_sftp()

        # Get all the annotations
        all = []
        for annotator in self.user.get_anno():
            file_list = ftp.listdir(path = self.user.get_annPath() + annotator + "/")
            for file in file_list:
                all += self.parse_file(annotator, file, ftp)

        corpus = pd.DataFrame.from_records(all, index = "case", columns=["annotator", "case", "line", "body", "from", "to", "relation", "pos", "mj"])

        # Close FTP connetion
        ftp.close()
        ssh.close()

        return corpus

    def get_offline(self):
        all = []
        for annotator in self.user.get_anno():
            for file in list(range(1, 301)):
                file = str(file) + ".txt"
                all += self.parse_offline(annotator, file)

        corpus = pd.DataFrame.from_records(all, index = "case", columns=["annotator", "case", "line", "body", "from", "to", "relation", "pos", "mj"])
        return corpus

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

    def parse_file(self, annotator, file, ftp):
        """
        Parses .txt file storing annotation data.
        """
        all = []
        case = ftp.file(self.user.get_annPath() + annotator + "/" + file,'r')
        lines = [i.strip("\n") for i in case.readlines()]
        with open("corpus/annodata/" + annotator + "/" + file, 'w') as f:
            for l in lines:
                f.write(l + "\n")

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

    def ext_sent(self, file_name):
        """
        Extracts sentence numbers and sentence text from cases.
        Returns them in a list.
        """
        files = os.listdir(self.user.get_corPath()) # Full corpus text is stored locally at corpus/corpus (or .get_corPath)
        for file in files:
            if file.endswith(".txt") and file == file_name:
                with open(self.user.get_corPath() + "/" + file) as f:
                    lines = [i.strip("\n") for i in f.readlines()]
                    lines = [line for line in lines if line.strip()] # ignore empty lines
                    line_num = list(range(len(lines))) # list of line numbers
                    max_line = len(lines)

        return line_num, lines, max_line

    def ext_ref(self, anno):
        """
        Parses the .txt annotation recieved from the server.
        Returns dictionary: {from: x, to: y, rel: x}
        """
        ref = {}

        # Checks for empty file
        if len(anno) > 0: anno = anno[:-1]
        else: return {}

        for line in anno:
            line = ast.literal_eval(line)
            try: # Tries to add to existing dic element
                r = ref[int(line[0])]
                r["from"].append(line[1])
                r["rel"].append(line[2])
                r["to"].append(line[3])
            except: # Creates a new dic element
                d = {"from": [line[1]], "rel": [line[2]], "to": [line[3]]}
                ref[int(line[0])] = d

        return ref

    def ext_mj(self, anno):
        """
        Extracts the majority, sorts them into a list "[judge1, judge2]"
        """

        if len(anno) > 0: mj = anno[-1] # MJ is allways at the end of the file
        else: return "NAN" # if empty file

        mj = ast.literal_eval(mj) # multiple judges are in a list
        if not isinstance(mj, list): mj = [mj] # single judge as a list

        mj = sorted(mj) # sorts for consistency
        if not mj[0].isdigit(): # checks if anno is a mj (if not it will start with a line num)
            mj = ", ".join(mj)
            return mj
        return "NAN"


    def update_corpus(self):
        """
        Loads corpus from pickle or downloads a new one from holj.ml
        """

        if self.update:
            # corpus = self.ssh_download() # download fresh corpus data
            corpus = self.get_offline()
            # corpus = load_data("corpus")
            corpus = self.get_annoupdate(corpus) # report on the new anno state
            save_data("corpus", corpus)
            return corpus

        else:
            return load_data("corpus") # load existing data #NOTE write test

    def get_corpus(self, type):
        """
        Splits corpus into Machine Learning (ML) corpus used to train ML classifier,
        and a Majority Judgement (MJ) corpus used to evaluate the ML classifier.
        """
        corpus = self.update_corpus().sort_index() # full corpus sorted by case
        dirty = corpus
        x = corpus[corpus['annotator']==self.user.mainAnno] # select only one annotator
        x = x.reset_index(drop=True)

        # Split on cases (as opposed to lines etc.)
        # casenum = x.case.unique()
        # xnum = len(casenum)*self.MJ_size
        # casenum = 300
        # xnum = 200

        random.seed(42)
        all = list(range(1, 301))
        shuffle(all)
        a = all[:100]
        b = all[100:200]
        c = all[200:300]
        d = all


        # testnum = random.sample(range(1, 301), 200)
        # casenum = list(range(1, 301))
        # trainnum = list(set(casenum) - set(testnum))

        # random.seed(42)
        # testnum = random.sample(range(casenum), xnum)
        # testnum = [casenum[i] for i in testnum]
        # trainnum = list(set(casenum) - set(testnum))

        MJ_set = x.loc[x["case"].isin(a)].sort_values(by=['case', 'line'])
        ML_set = x.loc[x["case"].isin(b)].sort_values(by=['case', 'line'])
        ALL_set = x.loc[x["case"].isin(c)].sort_values(by=['case', 'line'])
        Count_set = x.loc[x["case"].isin(d)].sort_values(by=['case', 'line'])


        if type == "ml":
            return ML_set
        if type == "mj":
            return MJ_set
        if type == "all":
            return ALL_set
        if type == "count":
            return Count_set
        if type == "dirty":
            return dirty

    def get_annoupdate(self, corpus):
        """
        Prints progress report on annotators.
        """

        anno = corpus.reset_index()
        annotators = anno["annotator"].unique()
        progress = self.print_anno(anno, annotators)

        min_ann = annotators[progress.index(min(progress))]
        min_ann_cases = anno[anno.annotator == min_ann]["case"].unique()

        anno = self.print_mj(anno, min_ann_cases, annotators)
        gs = self.print_line_slow(anno, min_ann_cases, annotators)
        # self.print_line(anno, min_ann_cases, annotators)
        return gs

    def print_anno(self, corpus, annotators):
        """
        Prints how much the annotators have annotated out of corpus.
        """
        total = self.get_maxcase()
        completed = len(corpus["case"].unique())
        percentage = (completed/total) * 100
        print("\nProgress Report:\nTotal annotated %d out of %d (%d%%)" % (completed, total, percentage))

        progress = []
        for annotator in annotators:
            a = corpus[corpus.annotator == annotator]
            completed = len(a["case"].unique())
            progress.append(completed)
            print("     %s completed %d" % (annotator, completed))

        return progress

    def get_maxcase(self):
        """
        Returns the total number of cases in the corpus.
        """
        total = 0
        files = os.listdir(self.user.get_corPath())
        for file in files:
            if file.endswith(".txt"):
                total += 1

        return total

    def print_mj(self, corpus, min_ann_cases, annotators):
        """
        Prints how much the annotators agree on majority of judgement.
        """

        dfmj = corpus.sort_values(by=['case', 'line', 'relation'])

        agreement = 0
        disagreement = 0

        comp_agreement = 0
        comp_disagreement = 0

        golden = []

        anno_mstks = {"gr": 0, "alice": 0, "jasleen": 0}

        for case in min_ann_cases:
            most = None
            nomost = None
            mj = []
            truth = "NAN"
            for annotator in annotators:
                mj.append(str(dfmj[(dfmj.case == case) & (dfmj.annotator == annotator)]["mj"].unique()))


            if len(set(mj)) > 1 and len(set(mj)) < 3:
                most, cnt = Counter(mj).most_common()[0]
                disagreement += 1
                print("\nCase: %d" % (case))
                for a, c in zip(annotators, mj):
                    print(a, c)
            else:
                agreement += 1

            if len(set(mj)) > 2:
                nomost = True
                comp_disagreement += 1
                print("Complete Case Disagreement:")
                print("\nCase: %d" % (case))
                for a, c in zip(annotators, mj):
                    print(a, c)
            else:
                comp_agreement += 1

            if most != None and nomost == None:
                truth = most[2:-2]
                golden.append(most[2:-2])
            elif nomost == True:
                golden.append(mj[0][2:-2])
            else:
                truth = mj[0][2:-2]
                golden.append(mj[0][2:-2])



            for annotator in annotators:
                test = dfmj[(dfmj.case == case) & (dfmj.annotator == annotator)]["mj"].unique()
                if test[0] != truth:
                    anno_mstks[annotator] += 1
                    print(annotator, "Fail!!", anno_mstks[annotator])
                index = dfmj[(dfmj.case == case)].index.values.tolist()

            for i in index:
                dfmj.at[i, 'mj'] = truth
            print("truth", truth)

        self.golden = golden
        agr = agreement/(agreement + disagreement)
        print("\n--------------------------------------------\nAgreed: %d, Disagreed: %d" % (agreement, disagreement))
        print("Agreement:", agr)
        print("\n--------------------------------------------\nComp_Agreed: %d, Comp_Disagreed: %d" % (comp_agreement, comp_disagreement))

        return dfmj

    def print_line(self, corpus, min_ann_cases, annotators):
        """
        Prints how much the annotators agree on labels for individual lines.
        """

        df = corpus.sort_values(by=['case', 'line', 'relation']).drop(columns=["mj"])
        agreement = 0
        disagreement = 0

        comp_agreement = 0
        comp_disagreement = 0

        for case in min_ann_cases:
            lines = df[df.case == case]["line"].unique()
            for line in lines:
                compare = []
                for annotator in annotators:
                    annotation = df[(df.case == case) & (df.line == line) & (df.annotator == annotator)]
                    compare.append(str(annotation['relation'].unique().tolist()))
                if len(set(compare)) > 1:
                    disagreement += 1
                    print("\nCase: %d, Line %d" % (case, line))
                    for a, c in zip(annotators, compare):
                        print(a, c)
                else:
                    agreement += 1

                if len(set(compare)) > 2:
                    comp_disagreement += 1
                    print("Complete Case Disagreement:")
                    print("Case: %d, Line %d" % (case, line))
                    for a, c in zip(annotators, compare):
                        print(a, c)
                else:
                    comp_agreement += 1

        agr = agreement/(agreement + disagreement)
        print("\n--------------------------------------------\nAgreed: %d, Disagreed: %d" % (agreement, disagreement))
        print("Agreement:", agr, "\n")
        print("\n--------------------------------------------\nComp_Agreed: %d, Comp_Disagreed: %d" % (comp_agreement, comp_disagreement))


    def print_line_slow(self, corpus, min_ann_cases, annotators):
        """
        Prints how much the annotators agree on labels for individual lines.
        """

        df = corpus.sort_values(by=['case', 'line', 'relation']).drop(columns=["mj"])
        df = df[df.relation.isin(["fullagr", "ackn"])]
        gs = corpus

        complete_dis = 0
        complete_agr = 0
        agreement = 0
        disagreement = 0
        for case in min_ann_cases:
            lines = df[df.case == case]["line"].unique()
            for line in lines:
                compare = []
                # print("TESTLINE", line)
                for annotator in annotators:
                    annotation = df[(df.case == case) & (df.line == line) & (df.annotator == annotator)]
                    # print("ANNOTATION", annotation)
                    an = annotation['relation'].unique().tolist()
                    compare.append(str(annotation['relation'].unique().tolist()))

                if len(set(compare)) > 2:
                    complete_dis += 1
                    print("\n Complete Disagreement!")
                    print("Case: %d, Line %d" % (case, line))
                    for a, c in zip(annotators, compare):
                        print(a, c)
                else:
                    complete_agr += 1

                # print("COMPARE", compare)
                if len(set(compare)) > 1:
                    disagreement += 1
                    most, cnt = Counter(compare).most_common()[0]
                    print("\nCase: %d, Line %d" % (case, line))
                    for a, c in zip(annotators, compare):
                        print(a, c)
                        if c != most:
                            outcast = a

                    if "fullagr" in most:
                        most = "fullagr"

                    if "fullagr" not in most and "ackn" not in most:
                        most = "NAN"

                    if "fullagr" not in most and "ackn" in most:
                        most = "ackn"

                    index = gs[(gs.case == case) & (gs.line == line) & (gs.annotator == outcast)].index.values.tolist()

                    # print("before", gs[(gs.case == case) & (gs.line == line)])
                    for i in index:
                        gs.at[i, 'relation'] = most
                    # print("after", gs[(gs.case == case) & (gs.line == line)])

                else:
                    agreement += 1


        agr = agreement/(agreement + disagreement)
        print("\n--------------------------------------------\nOne_Agreed: %d, One_Disagreed: %d" % (agreement, disagreement))
        print("Agreement:", agr, "\n")

        print("\n--------------------------------------------\nComp_Agreed: %d, Comp_Disagreed: %d" % (complete_agr, complete_dis))
        return gs

    def print_line_fast(self, corpus):
        corpus = corpus.reset_index()
        # print(corpus)
        df1 = corpus[(corpus.annotator == "gr") & (corpus.relation.isin(["fullagr", "ackn"]))][["line", "relation", "case"]].sort_values(by=['case', 'line']).reset_index(drop=True)
        df2 = corpus[(corpus.annotator == "jasleen") & (corpus.relation.isin(["fullagr", "ackn"]))][["line", "relation", "case"]].sort_values(by=['case', 'line']).reset_index(drop=True)

        gj = pd.concat([df1,df2]).drop_duplicates(keep=False)
        ind = gj.index.tolist()
        print("THIS1", df1[df1.index.isin(ind)])
        # print("THIS2", df2[df2.index.isin(ind)])
        # print(df2[(df2["line"] == 22) & (df2["case"] == 5)])
        # print(df2[(df2["line"] == 194) & (df2["case"] == 7)])
