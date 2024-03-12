from ASMO.user.user import User
from ASMO.corpus.corpus import Corpus
from ASMO.corpus.visualise import Visualise
from ASMO.classifier.classifier import Classifier
from ASMO.classifier.perfect import Perfect
from ASMO.majority.majority2 import Majority
from ASMO.majority.baselines import Baseline
from ASMO.majority.OptBaseline import Optimal
from ASMO.corpus.stats import Stats
import pandas as pd

class Pipeline:
    # ---- Corpus Settings ----
    key = "/Users/josefvalvoda/Dropbox/key/node.pem" #pem key for SSH
    ip = "ec2-18-182-64-196.ap-northeast-1.compute.amazonaws.com" #Amazon EC2 IP
    user = "ubuntu"
    annotators = ["gr", "alice", "jasleen"]
    corPath = "corpus/corpus/"
    annPath = "./annotator/anno/"
    mainAnno = "gr" # Select the annotator to train with
    download = False # Downloads latest data from www.holj.ml
    MJ_size = 0.33 # Size of the test corpus, the rest is used for training ML

    # ---- Classifier Settings ----
    train = False # Retrains the classifier
    test_size = 0.33 # Selects best ML algorithm/hyper-parameters by evaluating on this size of MJ corpus.
    downsample = True # Train on the same amount of positive and negative samples
    info = True # Prints the results of the algorithm/parameters performance


if __name__ == '__main__':
    pip = Pipeline()

    # Get corpus
    amazon = User(pip.key, pip.ip, pip.user, pip.annotators, pip.corPath, pip.annPath, pip.mainAnno)
    holj_corpus = Corpus(amazon, pip.MJ_size, pip.download)
    ML_corpus = holj_corpus.get_corpus(type = "ml")
    MJ_corpus = holj_corpus.get_corpus(type = "mj")
    ALL_corpus = holj_corpus.get_corpus(type = "all")
    out = ALL_corpus[["case", "line", "body", "from", "to", "relation", "pos", "mj"]]
    out.to_csv(r'AI.csv')
    # # cnt = holj_corpus.get_corpus(type = "count")
    # # st = Stats(cnt)
    # # st.count_AS()
    # # st.count_MO()

    # # st.predict()
    # # st.count_GEN()
    # # st.count_JUD()
    # #
    print("\n\nTraining Classifier")
    #Train ML classifier
    #ALL_corpus[(ALL_corpus.case == 4) & (ALL_corpus.line == 88)].relation.item()
    classifier = Classifier(ML_corpus, pip.test_size, pip.train)
    predicted = classifier.get_prediction(ALL_corpus)
    print(predicted)


    print("\n\nHuman Classifier")
    # Human classifier for pipeline evaluation
    perf = Perfect(MJ_corpus)
    hum_predicted = perf.get_pred()

    # Apply rules
    majority = Majority(MJ_corpus, hum_predicted)
    majority.predict()

    print("\n\nMachine Classifier")
    # Apply rules
    majority = Majority(ALL_corpus, predicted)
    majority.predict()

    print("\n\nBaselines:")
    # Print baselines
    optimal = Optimal(pip.corPath, ML_corpus)
    print("\n\n1")
    num = optimal.find_optimal()
    print("\n\n2")
    baselines = Baseline(num, pip.corPath, ALL_corpus)
    print("\n\n3")
    baselines.find_majority()
    print("\n\n4")
    baselines.find_AS()

    # # Visualise corpus
    # # vis = Visualise(amazon, cnt)
    # # vis.html_corpus()
