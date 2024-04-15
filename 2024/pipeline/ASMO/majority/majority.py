from collections import Counter
import pandas as pd
import numpy as np

class Majority:

    def __init__(self, corpus, predicted):
        self.corpus = corpus
        self.predicted = predicted

    def predict(self):
        """
        Creates a map (dictionary) of predicted agreements between judges for each case.
        Resolves this map to see who does the majority agree with.
        Compares the predicted MJ with the actual MJ.
        """

        map = self.map_agreement()
        pred_mj = self.resolve_map(map)
        true_mj = self.corpus[["case", "mj"]].drop_duplicates("case").reset_index(drop=True)
        self.evaluate(pred_mj, true_mj)

    def new_predict(self):
        """
        Creates a map (dictionary) of predicted agreements between judges for each case.
        Resolves this map to see who does the majority agree with.
        """

        map = self.map_agreement()
        pred_mj = self.resolve_map(map)
        return map, pred_mj

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
        print(cases)

        for case in cases:
            judges, locations = self.get_judges(self.corpus[self.corpus['case'] == case], case)
            print(judges,locations)
            pred_lines = self.get_lines(case)
            # pred_lines = self.predicted[self.predicted["case"] == case]["line"].tolist()
            for judge, loc in zip(judges, locations):
                agreed_judge = []
                lines = [i for i in range(loc[0], loc[1])]
                sentences = list(set(lines).intersection(pred_lines))
                agreed_judge += self.parse_sent(sentences, case, judges, judge)
                judge_map = self.add_map(judge_map, case, judge, set(agreed_judge)) #set(agreed_judge) removes duplicates
            print('judge_map:', judge_map)

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
                if sorted(ground_truth) != sorted(names):
                    print(case, "truth", ground_truth, "pred", names, sent, "\n")
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
        if "For these reasons" in sent or "in this opinion" in sent or "For these short reasons" in sent or "reasons I have given" in sent:
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
        #print(break_pos)
        if len(break_pos) < 5:
            print("Warning: Case with less than 5 judges, check %s, number of judges %d" % (number, len(break_pos)))

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
        judge = " ".join(judge.lower().split(" ")[1:2]) # keeps two lower-case names i.e. lord x
        return judge

    def add_map(self, judge_map, case, judge, agreed_judge):
        """
        Adds the judge relation to the dictionary.
        {'case': {'from_judge': 'to_judge_x, to_judge_y, to_judge_z'}}
        """
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
            names = sorted(names) # Keeps ordering of names consistent
            if list(names)[0] is not None:
                names = ", ".join(names)
                if judge == names:
                    # print("________TEST________", judge)
                    names = "NAN"
            else: names = "NAN"
            # print(judge, names)
            cited.append(names)

        # print(cited)
        citations = Counter(cited)

        return citations

    def rule_one(self, map, citations, case):
        """
        Attributes MJ to a most cited judge, unless there are more judges equally cited.
        """

        min_agreement = int(len(map[case].keys())/2) # min agreement is one below the majority of judges ie. for 5 it's 2 for 7 it's 3 for 6 it's 3
        print('min_agreement:', min_agreement)
        if min_agreement == 1:
            min_agreement = 2
        max = 0
        mj = "NAN"

        for k,v in citations.items():
            print('k:',k, 'v:', v)
            if k != "NAN":
                if v == max and mj != "NAN": # Two judges equally cited means NAN is mj
                    mj = "NAN"
                elif v == max and mj == "NAN": # Judge cited equally to NAN, judge is mj
                    mj = k
                if v > max and v >= min_agreement: # Basic rule, max cited judge is mj
                    mj = k
                    max = v
            print('mj:', mj, max)

        return mj

    def rule_two(self, map, mj, case):
        """
        If mj agrees with another judge. That judge is also part of MJ.
        """
        # NOTE must work for agreement from more than one judge ie. A, B majority agrees with C
        try:
            names = map[case][mj]
            names = sorted([list(names)[0], mj])
            names = list(set(names))
            print(names)
            if list(names)[0] is not mj:
                names = ", ".join(names)
            elif list(names)[1] is not mj:
                names = ", ".join(names)
            return names
        except:
            return mj

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

    def resolve_map(self, map):
        """
        Applies the rules to find majority.
        """

        cases = map.keys()
        all = []
        for case in cases:
            print("-------NEW-----:", case)
            citations = self.count_citations(map, case)
            print(citations)
            mj = self.rule_one(map, citations, case)
            print("RULE1:", mj)
            mj = self.rule_two(map, mj, case)
            print("RULE2:", mj)
            try:
                mj = self.rule_three(map, citations, mj, case)
                print("RULE3:", mj)
            except:
                pass
            all.append([case, mj])

        corpus = pd.DataFrame.from_records(all, columns=["case", "mj"])
        print('mj return is:', corpus)

        return corpus
