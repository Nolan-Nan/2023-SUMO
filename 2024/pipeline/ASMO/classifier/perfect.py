class Perfect:

    def __init__(self, corpus):
        self.corpus = corpus

    def get_pred(self):
        fullagr = self.corpus[(self.corpus["relation"] == "fullagr") | (self.corpus["relation"] == "ackn")][["case", "line", "relation"]]
        return fullagr
