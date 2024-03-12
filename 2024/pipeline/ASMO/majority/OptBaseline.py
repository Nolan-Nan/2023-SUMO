import collections
import os
import pandas as pd
import string

class Optimal():
    def __init__(self, path, corpus):
        self.threshold = range(10, 100, 5)
        self.path = path
        self.corpus = corpus

    def find_optimal(self):
        max_ac = 0
        max_thres = 0
        for thres in self.threshold:
            accuracy = self.find_majority(thres)
            if accuracy > max_ac:
                max_ac = accuracy
                max_thres = thres
            print(thres, accuracy)

        print("Best threshold:", max_thres, max_ac)
        return thres

    def find_majority(self, thres):
        allowed = self.corpus["case"].unique().tolist()
        cases = os.listdir(self.path)

        majorityD = []
        casenum = []
        for case in allowed:
            case = str(case) + ".txt"

            loc = self.path + case # Path to a case
            judges = self.get_judges(loc) # Dictionary judge: body
            casenum.append(int(case.strip(".txt")))

            name = []
            all_judges = []
            rejected_judges = []
            for judge in judges.keys():
                body = judges[judge]["body"]
                sent_cnt = 0
                for sentence in body:
                    sent_cnt += 1

                if sent_cnt < thres:
                    rejected_judges.append(judge)
                all_judges.append(judge)

            counter = collections.Counter(name)

            min_sent_cnt_judge = list(set(all_judges).symmetric_difference(set(rejected_judges)))
            min_sent_cnt_judge.sort()
            min_sent_cnt_judge = ", ".join(min_sent_cnt_judge)

            majorityD.append(min_sent_cnt_judge)

        mjD = pd.DataFrame({"mj": majorityD, "case": casenum})

        # print("\nBASELINE D: Keep only judges with more than ")
        accuracy = self.print_results(mjD)
        return accuracy

    def print_results(self, mj):
        corpus = self.corpus.reset_index()
        majority = corpus[["case", "mj"]].drop_duplicates(subset = "case").sort_values("case")
        mj = mj.sort_values("case")
        positives = majority.reset_index()["mj"] == mj.reset_index()["mj"]
        # print("\nMetrics: \n", positives.value_counts())
        counts = positives.value_counts().tolist()
        accuracy = counts[1]/100 * 100
        # print("\nAccuracy: ", "%.2f" % accuracy, "%")
        return accuracy
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
                if flag == True:
                    judge = self.cln_judge(line)
                    judges[judge] = {}
                    judges[judge]["body"] = []
                    flag = False
                elif judge:
                    judges[judge]["body"].append(line)
                if "NEW JUDGE" in line:
                    flag = True

            return judges

    def cln_judge(self, judge):
        """
        Judge name is lower-case first two words of his name.
        """
        judge = " ".join(judge.lower().split(" ")[:2]) # keeps two lower-case names i.e. lord x
        return judge
