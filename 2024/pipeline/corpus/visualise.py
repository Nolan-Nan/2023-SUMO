"""
Visualise the annotated sentences on a corpus.
"""

from corpus.corpus import Corpus 
import re
import os

class Visualise(Corpus):

    def __init__(self, user, corpus):
        self.user = user
        self.annotated = corpus
        self.out_path = "corpus/visualise"

    def html_corpus(self):
        """
        returns all sentences and labels as colored
        HTML
        """

        files = os.listdir(self.user.get_corPath())

        #Get main annotators annotations
        # anno_full = self.annotated.loc[(self.annotated['annotator'] == self.user.get_main()) & (self.annotated['relation'] == "fullagr")]
        # anno_ackn = self.annotated.loc[(self.annotated['annotator'] == self.user.get_main()) & (self.annotated['relation'] == "ackn")]
        # anno = anno_full[["line"]]
        # anno2 = anno_ackn[["line"]]

        for file in files:
            if file.endswith(".txt"):
                with open(self.user.get_corPath() + "/" + file, "r") as f, open(self.out_path + "/" + file.strip(".txt") + ".html", "w") as out:
                    lines = f.readlines()
                    lines = [i.strip("\n") for i in lines]
                    file = int(file.strip(".txt"))

                    # if file in anno.index:
                    if file:
                        try:
                            annotations = self.annotated[(self.annotated["case"] == file) & (self.annotated["relation"] == "fullagr")]["line"].values.tolist()
                            # annotations = anno.loc[file,"line"].values.tolist()
                        except:
                            annotations = []
                        try:
                            acknowledgements = self.annotated[(self.annotated["case"] == file) & (self.annotated["relation"] == "ackn")]["line"].values.tolist()
                            # acknowledgements = anno2.loc[file,"line"].values.tolist()
                        except:
                            acknowledgements = []

                        i = 0
                        for line in lines:
                            if line.strip():
                                if i in annotations:
                                    out.write("<p style='background-color:powderblue;'>" + line + "<b>" + "Full Agreement" +"</b></p>\n")
                                elif i in acknowledgements:
                                    out.write("<p style='background-color:powderblue;'>" + line + "<b>" + "Acknowledgement" +"</b></p>\n")
                                else:
                                    out.write("<p>" + line + "</p>\n")
                                i += 1

                        mj = self.annotated[self.annotated["case"] == file]["mj"].values.tolist()
                        mj = " ".join(list(set(mj)))
                        out.write("<h1>MO: " + mj + "</h1>\n")
