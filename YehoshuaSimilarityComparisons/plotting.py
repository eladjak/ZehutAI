from matplotlib import pyplot as plt
from sim import Similarity
from transformers import BertTokenizer, RobertaTokenizer, BertModel

def runModels():
    similarity = Similarity()
    similarity.addModel(BertTokenizer, BertModel, 'bert-base-uncased')
    similarity.runNNModels()