import xml.etree.ElementTree as ET
import subprocess
import os
from pycorenlp import StanfordCoreNLP

class Deparser:
    def parse(self, text):
        nlp = StanfordCoreNLP('http://localhost:9000')
        output = nlp.annotate(text, properties={
        'ssplit' :' eolonly',
        'annotators': 'tokenize,ssplit,pos,depparse',
        'outputFormat': 'xml'
        })

        # print("TESTSSSSSSSSSSS")
        #
        # owd = os.getcwd()
        # os.chdir("classifier/NLPStanford/")
        # self.save_text(text)
        # subprocess.call(['java','-cp','*','edu.stanford.nlp.pipeline.StanfordCoreNLP','-ssplit.eolonly','-annotators','tokenize,ssplit,pos,depparse','-file','foo.txt'])

        root = ET.fromstring(output)
        # root = tree.getroot()

        gov = []
        dep = []
        for child in root.findall(".//dependencies"):
            # basic-dependencies / collapsed-dependencies / collapsed-ccprocessed-dependencies
            # / enhanced-dependencies /enhanced-plus-plus-dependencies
            if(child.attrib["type"] == "enhanced-plus-plus-dependencies"):
                for g in child.findall(".//governor"):
                    gov.append(g.text)
                for d in child.findall(".//dependent"):
                    dep.append(d.text)

        output = []
        for g,d in zip(gov, dep):
            output.append(g+d)

        s = " ".join(output)
        out = s.strip("ROOT")
        print("testing")

        # os.chdir(owd)
        # save as foo.txt
        # parse with stanfordnlp
        # parse the tree to get dependency sentence
        # return the sentence
        # print(out)
        return out
    def save_text(self, text):
        with open("foo.txt", "w") as file:
            file.write(text)

    # def parse_nlp(tree):
