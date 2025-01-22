"""
Load the model, run predictions
"""
import torch, pickle
from models import LSTMClassifier


def load_cp(cp_path):
    word2index = pickle.load(open("data/word2index.pkl", "rb"))
    model = LSTMClassifier(len(word2index), 300, 512, 2, 2)
    model.load_state_dict(torch.load(cp_path))
    return model
