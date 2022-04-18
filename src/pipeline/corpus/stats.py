"""
x Number of judges in case - DONE
x Number of explicit AS - DONE
x Number of judges in inferred MO (0 if there is no MO) - DONE
x Number of inferred agreements (transitive/reflexive) - DONE
x Number of inferred agreements necessary to establish MO (or just work of the number of cases where this happens) - DONE
x Number of stand-alone judgements (i.e. that have no outgoing agreements) - DONE
x Number of stand-alone judgements that are cited (i.e. that have some incoming but no outgoing agreements) - DONE
x Number of generic agreements - DONE
Number of generic agreements that are cited
"""

from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Stats:

    def __init__(self, corpus):
        self.corpus = corpus
        self.predicted = self.get_pred()

    def get_pred(self):
        fullagr = self.corpus[(self.corpus["relation"] == "fullagr") | (self.corpus["relation"] == "ackn")][["case", "line", "relation"]]
        return fullagr

    def count_AS(self):
        a_s = self.corpus[self.corpus["relation"] == "fullagr"].groupby(["case"]).size()
        a_s.plot.hist()
        plt.show()

        a_s = list(zip(a_s.index, a_s))
        cnt = 1
        index = 0

        for k,v in a_s:
            if k != cnt:
                a_s.insert(index, (cnt, 0))
                print(a_s[index][0], a_s[index][1])
            else:
                print(k, v)
            cnt += 1
            index += 1

    def count_MO(self):
        data = []
        cases = list(range(1,301))
        for case in cases:
            mj = self.corpus[self.corpus["case"] == case]["mj"].unique().item()
            if mj == "NAN":
                data.append(0)
                print(case, 0)
            else:
                MO_num = len(mj.split(","))
                data.append(MO_num)
                print(case, MO_num)

        his = pd.Series(data)
        his.plot.hist()
        plt.show()


    def count_GEN(self):
        a_s = self.corpus[(self.corpus["relation"] == "fullagr") & (self.corpus["to"] == "all")].groupby(["case"]).size()
        a_s = list(zip(a_s.index, a_s))

        cnt = 1
        index = 0
        test = []
        print(len(a_s))
        for k,v in a_s:
            if k != cnt:
                test.append(0)
                a_s.insert(index, (cnt, 0))
                print(a_s[index][0], a_s[index][1])
            else:
                test.append(v)
                print(k, v)
            cnt += 1
            index += 1

        while cnt != 301:
            print(cnt, 0)
            cnt += 1

        his = pd.Series(test)
        his.plot.hist()
        plt.show()

    def count_JUD(self):
        jdg = []
        cases = list(range(1,301))
        for case in cases:
            judges = self.get_jud(self.corpus[self.corpus['case'] == case], case)
            print(case, len(judges))
            jdg.append(len(judges))

        his = pd.Series(jdg)
        his.plot.hist()
        plt.show()

    def get_jud(self, case, number):
        """
        Finds names of the judges in the case.
        Finds line number corresponding to the judge.
        """
        break_pos = case[case["body"] == "------------- NEW JUDGE --------------- "]["line"].tolist()

        judges = []
        for brk in break_pos:
            judge = case[case["line"] == brk+1]["body"].values[0] # +1 Judge name is always one line after break.
            judges.append(judge)

        return judges

##------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------------------------

    def predict(self):
        """
        Creates a map (dictionary) of predicted agreements between judges for each case.
        Resolves this map to see who does the majority agree with.
        Compares the predicted MJ with the actual MJ.
        """

        map = self.map_agreement()
        pred_mj = self.resolve_map(map)
        true_mj = self.corpus[["case", "mj"]].drop_duplicates("case").reset_index(drop=True)
        # self.evaluate(pred_mj, true_mj)

    def evaluate(self, pred_mj, true_mj):
        """
        Prints accuracy and error report.
        """
        agree = np.where((true_mj['mj'] == pred_mj["mj"]), True, False)
        pred_mj["agr"] = agree
        pred_mj["truth"] = true_mj["mj"]
        print("\nError report:\n", pred_mj.to_string())

        stats = pred_mj.groupby('agr').count()["case"]
        total = stats[0] + stats[1]
        accuracy = stats[1]/total * 100
        print("\nAccuracy: ", accuracy)

    def map_agreement(self):
        """
        For each judge in each case of the corpus, finds who they fully agree with
        and store it in a dictionary.
        {'case1': {'judge1': 'x, y, z'}}
        """
        judge_map = {}
        cases = self.corpus['case'].unique().tolist() # numbers of cases in corpus

        for case in cases:
            judges, locations = self.get_judges(self.corpus[self.corpus['case'] == case], case)
            pred_lines = self.get_lines(case)
            for judge, loc in zip(judges, locations):
                agreed_judge = []
                lines = [i for i in range(loc[0], loc[1])]
                sentences = list(set(lines).intersection(pred_lines))
                agreed_judge += self.parse_sent(sentences, case, judges, judge)
                judge_map = self.add_map(judge_map, case, judge, set(agreed_judge)) #set(agreed_judge) removes duplicates

        return judge_map

    def get_lines(self, case):
        """
        Filter out the acknowledgement lines which don't have full agreement
        following immediately after. Return as a list.
        """

        line = self.predicted[self.predicted["case"] == case]["line"].tolist()
        relation = self.predicted[self.predicted["case"] == case]["relation"].tolist()
        lines = []
        i = 0

        for l in line:
            if relation[i] == "ackn":
                next = i+1
                while next < len(line) and (l == line[next] or relation[next] == "ackn"):
                    next += 1
                if next < len(line) and l+1 == line[next]:
                    lines.append(l)
                    lines.append(line[next])
            else:
                lines.append(l)

            i += 1

        return list(set(lines))

    def hasNumbers(self, inputString):
        return any(char.isdigit() for char in inputString)

    def parse_sent(self, sentences, case, judges, judge):
        """
        Finds names of judges present in the sentence.
        Returns them as a list.
        """
        if sentences:
            sent = self.corpus[(self.corpus["case"] == case) & (self.corpus["line"].isin(sentences))]["body"].unique().tolist() # Gets sentences
            ground_truth = self.corpus[(self.corpus["case"] == case) & (self.corpus["line"].isin(sentences)) & self.corpus["relation"].isin(["fullagr"])]["to"].unique().tolist()
            ordr_sent = []

            for s in sent:
                if not self.hasNumbers(s):
                    ordr_sent.append(self.check_ordr(s))
            sent = ordr_sent

            if self.check_ackagrees(sent, judges):
                sent = sent[1:]
                sent = " ".join(sent)
            else:
                sent = " ".join(sent) # joins acknowledgements and fullagreement sentences NOTE change here to check if full agreement exists.

            names = self.extr_name(sent, judges)
            if self.check_self(sent):
                names.append(judge)
            if names:
                # if sorted(ground_truth) != sorted(names):
                    # print(case, "truth", ground_truth, "pred", names, sent, "\n")
                return sorted(names)
        return [None]

    def check_ackagrees(self, sent, judges):
        if len(sent) >= 2:
            names1 = self.extr_name(sent[0], judges)
            names2 = self.extr_name(sent[1],judges)
            if names2:
                return True

    def check_ordr(self, sent):
        sent = sent.split("order")[0]
        return sent

    def check_self(self, sent):
        if "For these reasons" in sent or "in this opinion" in sent or "For these short reasons" in sent or "reasons I have given" in sent or "In agreement" in sent or "I wish also" in sent or "for the foregoing" in sent:
            return True
        return False

    def extr_name(self, sent, judges):
        """
        Finds which judge is cited in a sentence.
        """
        names = []
        for judge in judges:
            if judge in sent.lower():
                judge = self.name_changer(judge)
                names.append(judge)
        return names

    def get_judges(self, case, number):
        """
        Finds names of the judges in the case.
        Finds line number corresponding to the judge.
        """
        break_pos = case[case["body"] == "------------- NEW JUDGE --------------- "]["line"].tolist()


        judges = []
        for brk in break_pos:
            judge = case[case["line"] == brk+1]["body"].values[0] # +1 Judge name is always one line after break.
            judges.append(self.cln_judge(judge))

        break_pos = break_pos[1:] # remove first break, because case starts here
        break_pos.append(max(case['line'].tolist())) # add last line as last break

        start = 0
        locations = []
        for brk in break_pos:
            brk += 1 # adjusts for indexing from 0
            locations.append([start, brk])
            start = brk

        return judges, locations

    def cln_judge(self, judge):
        """
        Judge name is lower-case first two words of his name.
        """
        judge = " ".join(judge.lower().split(" ")[:2]) # keeps two lower-case names i.e. lord x
        return judge

    def add_map(self, judge_map, case, judge, agreed_judge):
        """
        Adds the judge relation to the dictionary.
        {'case': {'from_judge': 'to_judge_x, to_judge_y, to_judge_z'}}
        """
        judge = self.name_changer(judge)

        if not agreed_judge:
            agreed_judge = None

        if not case in judge_map:
            judge_map[case] = {}
            judge_map[case][judge] = agreed_judge
        else:
            judge_map[case][judge] = agreed_judge

        # print(case, judge_map[case])

        return judge_map

    def count_citations(self, map, case):
        """
        Counts the judges fully agreed with by the citing judge.
        Returns counter object.
        """
        cited = []
        judges = map[case].keys()

        for judge in judges:
            names = map[case][judge]
            if None in names: names.remove(None)
            names = sorted(names) # Keeps ordering of names consistent
            if names:
                names = ", ".join(names)
            else: names = "NAN"

            cited.append(names)

        # print(cited)
        citations = Counter(cited)

        return citations

    def rule_one(self, map, citations, case):
        """
        Attributes MJ to a most cited judge, unless there are more judges equally cited.
        """

        min_agreement = int(len(map[case].keys())/2)+1 # min agreement is one below the majority of judges ie. for 5 it's 2 for 7 it's 3 for 6 it's 3
        if min_agreement == 1:
            min_agreement = 2
        max = 0
        mj = "NAN"

        for k,v in citations.items():
            if k != "NAN":
                if v == max and mj != "NAN": # Two judges equally cited means NAN is mj
                    mj = "NAN"
                elif v == max and mj == "NAN": # Judge cited equally to NAN, judge is mj
                    mj = k
                if v > max and v >= min_agreement: # Basic rule, max cited judge is mj
                    mj = k
                    max = v

        return mj

    def rule_two(self, map, case):
        """
        If mj agrees with another judge. That judge is also part of MJ.
        """
        # NOTE must work for agreement from more than one judge ie. A, B majority agrees with C
        # print("ruletwo")
        try:
            for k, v in map[case].items():
                self.transitive(list(v), map, case)

        except:
            print("not")

    def transitive(self, keys, map, case):
        names = []
        for k in keys:
            cited = list(map[case][k])
            for c in cited:
                if c != None:
                    names.append(c)
        names += keys
        # print(list(set(names)))

    def rule_three(self, map, citations,  majority, case):
        min_agreement = int(len(map[case].keys())/2)
        max = 0
        mj = []
        for k,v in citations.items():
            if k != "NAN" and v >= min_agreement:
                mj.append(v)

        mj = list(set(mj))

        if mj:
            cleaned = []
            mj = mj[0]
            for k,v in citations.items():
                if v == mj:
                    test = k.split(", ")
                    if len(test) > 1:
                        for t in test:
                            names = list(map[case][t])
                            if "self" in names:
                                names.remove("self")
                            cleaned += names
                        if test[1] == cleaned[0] and test[0] == cleaned[1]:
                            print("Applied rule 3", test)
                            majority = ", ".join(test)

        return majority

    def name_changer(self, name):
        if name == "baroness hale":
            return "lady hale"
        else:
            return name

    def check_change(self, map1, map2, case):
        cnt = 0
        for v1, v2 in zip(map1, map2):
            v1 = list(v1)
            if not v2:
                v2 = [None]
            v1.sort()
            v2.sort()
            if v1 != v2:
                cnt += 1

        print(case, cnt)
        return cnt

    def is_cited(self, map, case, judge):
        if judge in str(map[case].values()):
            return True
        else:
            return False

    def resolve_map(self, map):
        """
        Applies the rules to find majority.
        """

        cases = map.keys()
        all = []

        inferred = []
        for case in cases:
            # citations = self.count_citations(map, case)
            # map1 = list(map[case].values())
            # mj1 = self.rule_one(map, citations, case)
            self.rule_trans(map, case)
            # map2 = list(map[case].values())
            # self.check_change(map1, map2, case)
            # map1 = list(map[case].values())
            inferred.append(self.rule_self(map, case))
            # citations = self.count_citations(map, case)
            # mj2 = self.rule_one(map, citations, case)
            # if mj1 != mj2:
            #     print(case, "yes")
            # else:
            #     print(case, "no")
            # map2 = list(map[case].values())
            # inferred.append(self.check_change(map1, map2, case))
            citations = self.count_citations(map, case)
            mj = self.rule_one(map, citations, case)
            all.append([case, mj])

        his = pd.Series(inferred)
        his.plot.hist()
        plt.show()

        corpus = pd.DataFrame.from_records(all, columns=["case", "mj"])

        return corpus

    def rule_trans(self, map, case):
        """
        If mj agrees with another judge. That judge is also part of MJ.
        """
        matrix = self.to_matrix(map, case)
        matrix = self.warshall(matrix)
        self.upate_map(map, case, matrix)

    def upate_map(self, map, case, matrix):
        names = list(map[case].keys())
        for i in range(len(matrix)):
            agreed_names = []
            for j in range(len(matrix)):
                if matrix[i][j] == 1:
                    n = self.name_changer(names[j])
                    agreed_names.append(names[j])
            n = self.name_changer(names[i])
            map[case][n] = agreed_names

    def to_matrix(self, map, case):
        names = list(map[case].keys())
        size = len(names)
        matrix = [[0 for x in range(size)] for y in range(size)]
        for key, value in map[case].items():
            key = self.name_changer(key)
            j_from = names.index(key)
            for v in value:
                if v is not None:
                    v = self.name_changer(v)
                    j_to = names.index(v)
                    matrix[j_from][j_to] = 1

        return matrix

    def warshall(self, a):
        assert (len(row) == len(a) for row in a)
        n = len(a)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    a[i][j] = a[i][j] or (a[i][k] and a[k][j])

        return a

    def rule_self(self, map, case):
        cnt_self = 0
        cnt_loneself = 0
        for k, v in map[case].items():
            if not v:
                if self.is_cited(map, case, k):
                    cnt_loneself += 1
                cnt_self += 1
                map[case][k] = [k]

        print(case, cnt_loneself)
        return cnt_loneself
