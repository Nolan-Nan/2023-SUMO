import collections
import os
import pandas as pd
import string

class Baseline():
    def __init__(self, threshold, path, corpus):
        self.threshold = threshold
        self.path = path
        self.corpus = corpus

    def find_AS(self):
        majorityA = []
        casenum = []
        allowed = self.corpus["case"].unique().tolist()
        for case in allowed:
            name = []
            sentences = self.corpus[(self.corpus["case"] == case) & (self.corpus["relation"].isin(["ackn", "fullagr"]))]
            sent = sentences["body"].values.tolist()
            for s in sent:
                if self.extractNER(s) != None:
                    name += self.extractNER(s)

            counter = collections.Counter(name)
            print(name)
            if name:
                majorityA.append(counter.most_common(1)[0][0])
            else:
                majorityA.append("wrong")
            casenum.append(case)

        mjA = pd.DataFrame({"mj": majorityA, "case": casenum})
        print("\nBASELINE A: Most Cited Judge")
        print(mjA)
        #self.print_results(mjA)

    def find_majority(self):
        allowed = self.corpus["case"].unique().tolist()
        majority = self.corpus[["case", "mj"]].drop_duplicates(subset = "case").sort_values("case")
        cases = os.listdir(self.path)
        majorityA = []
        majorityB = []
        majorityC = []
        majorityD = []
        casenum = []
        total_sent = 0
        total_sent_count = 0
        for case in allowed:
            case = str(case) + ".txt"
            loc = self.path + case # Path to a case
            judges = self.get_judges(loc) # Dictionary judge: body
            print(judges)
            casenum.append(case.strip(".txt"))
            name = []
            max_sent_cnt = 0
            max_word_cnt = 0
            max_sent_cnt_judge = 0
            max_word_cnt_judge = 0
            all_judges = []
            rejected_judges = []
            for judge in judges.keys():
                body = judges[judge]["body"]
                sent_cnt = 0
                word_cnt = 0
                for sentence in body:
                    total_sent += 1
                    sent_cnt += 1
                    sen_len = sentence.split(" ")
                    word_cnt += len(sen_len)
                    total_sent_count += len(sen_len)
                    if self.extractNER(sentence) != None:
                        name += self.extractNER(sentence)

                if word_cnt > max_word_cnt:
                    max_word_cnt = word_cnt
                    max_word_cnt_judge = judge

                if word_cnt > max_sent_cnt:
                    max_sent_cnt = word_cnt
                    max_sent_cnt_judge = judge

                if sent_cnt < self.threshold:
                    rejected_judges.append(judge)
                all_judges.append(judge)

            counter = collections.Counter(name)

            min_sent_cnt_judge = list(set(all_judges).symmetric_difference(set(rejected_judges)))
            min_sent_cnt_judge.sort()
            min_sent_cnt_judge = ", ".join(min_sent_cnt_judge)


            majorityA.append(counter.most_common(1)[0][0])
            majorityB.append(max_word_cnt_judge)
            majorityC.append(max_sent_cnt_judge)
            majorityD.append(min_sent_cnt_judge)

        print("total number of sentences:", total_sent, (total_sent_count/total_sent))

        mjA = pd.DataFrame({"mj": majorityA, "case": casenum})
        mjB = pd.DataFrame({"mj": majorityB, "case": casenum})
        mjC = pd.DataFrame({"mj": majorityC, "case": casenum})
        mjD = pd.DataFrame({"mj": majorityD, "case": casenum})

        print("\nBASELINE A: Most Cited Judge")
        print(mjA)
        #self.print_results(mjA)
        print("\nBASELINE B: Most Verbose Judge (Words)")
        print(mjB)
        # self.print_results(mjB)
        print("\nBASELINE C: Most Verbose Judge (Sentences)")
        print(mjC)
        #self.print_results(mjC)
        print("\nBASELINE D: Keep only judges with more than 15 sentences")
        print(mjD)
        #self.print_results(mjD)
        return mjA

    def print_results(self, mj):
        corpus = self.corpus.reset_index()
        majority = corpus[["case", "mj"]].drop_duplicates(subset = "case").sort_values("case")
        mj = mj.sort_values("case")
        positives = majority.reset_index()["mj"] == mj.reset_index()["mj"]
        print("\nMetrics: \n", positives.value_counts())
        counts = positives.value_counts().tolist()
        accuracy = counts[1]/100 * 100
        print("\nAccuracy: ", accuracy, "%")
        # print("\nWrongly decided: \n", majority.reset_index()[positives == False].reset_index()[["case", "mj"]])

    def extractNER(self, sentence):
        sentence = sentence.translate(str.maketrans("", "", string.punctuation))
        sentence = sentence.split(" ")

        indices = [i for i, x in enumerate(sentence) if x == "Lord" or x == "Lady" or x == "Baroness"]
        names = []
        for i in indices:
            if i+1 < len(sentence):
                name = sentence[i] + " " + sentence[i+1]
                names.append(name.lower())
        if names:
            return names
        else:
            return None

    def get_judges(self, loc):
        with open(loc, encoding='utf-8') as f:
            lines = [i.strip("\n") for i in f.readlines()]
            flag = False
            judge = None
            judges = {}
            for line in lines:
                if line.startswith("LORD"):
                    flag = True
                if flag == True:
                    judge = self.cln_judge(line)
                    judges[judge] = {}
                    judges[judge]["body"] = []
                    flag = False
                elif judge:
                    judges[judge]["body"].append(line)


            return judges

    def cln_judge(self, judge):
        """
        Judge name is lower-case first two words of his name.
        """
        judge = ' '.join(judge.split()[:2]).lower() # keeps two lower-case names i.e. lord x

        return judge
