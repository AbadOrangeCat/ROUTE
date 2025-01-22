"""
Raw data -->  prepare/preprocess --> text2indices --> M --> predictions --> loss

I --> M (e --> rnn --> out) --> logits --> loss

"""

import pickle, os
from utils import prepare_dataset
from dataloader import make_dataloader
from models import LSTMClassifier
from sklearn import metrics
from tqdm import tqdm
import torch, numpy as np
import torch.nn.functional as F


def train(dloader, model, criterion, optimizer):
    model.train()
    losses, acc = [], []
    for batch in tqdm(dloader):
        y = batch["label"]
        logits,probabilities  = model(batch)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds = torch.argmax(logits, -1)
        acc.append((preds == y).float().mean().item())

    print(
        f"Train Loss: {np.array(losses).mean():.4f} | Train Accuracy: {np.array(acc).mean():.4f}"
    )


@torch.no_grad()
def test(dloader, model, criterion):
    model.eval()
    losses, acc = [], []
    for batch in dloader:
        y = batch["label"]
        logits,probabilities = model(batch)
        loss = criterion(logits, y)
        losses.append(loss.item())
        preds = torch.argmax(logits, -1)
        redpro = []
        for i in probabilities:
            redpro+= [i[0].item() >i[1].item()]
        acc.append((preds == y).float().mean().item())

    print(f"Loss: {np.array(losses).mean():.4f} | Accuracy: {np.array(acc).mean():.4f}")
    return np.array(losses).mean(),np.array(acc).mean()

def useagagin(dloader, model):
    model.eval()
    losses, acc = [], []
    count = 0
    yesyes = 0
    yesno = 0
    noyes = 0
    nono = 0
    for batch in dloader:
        y = batch["label"]
        logits,probabilities = model(batch)
        preds = torch.argmax(logits, -1)


        for i in range(len(batch["label"])):
            sure = batch["label"][i]
            guess = preds[i]
            if sure == 1 and guess ==1:
                yesyes +=1
            elif sure == 0 and guess ==1:
                yesno += 1
            elif sure == 1 and guess == 0:
                noyes += 1
            elif sure == 0 and guess == 0:
                nono += 1
            if sure == 0 and guess == 1:
                print(batch["text"][i], sure,guess)
                count += 1
        print(yesyes,yesno, noyes,nono )
        acc.append((preds == y).float().mean().item())

    print("xxxx",count)
    precision = yesyes / (yesyes + yesno)
    recall = yesyes / (yesyes + noyes)
    ff = (2 * precision * recall) / (precision + recall)
    print("Precision:", yesyes / (yesyes + yesno), "recall: ", recall, "F1: ", ff)
    print(f"Accuracy: {np.array(acc).mean():.4f}")

def save_cp(model):
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    torch.save(model.state_dict(), "checkpoints/lstm_model.pt")

def save_best(model,name):
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    torch.save(model.state_dict(), "checkpoints/" + name + ".pt")

def main():
    device = "cuda"
    dataset, word2index = prepare_dataset("./url")
    # with open("./data/word2index.pkl", "wb") as f:
    #     pickle.dump(word2index, f)
    train_dloader = make_dataloader(dataset["train"], word2index, 300, 128, device)
    val_dloader = make_dataloader(dataset["val"], word2index, 300, 128, device)
    test_dloader = make_dataloader(dataset["test"], word2index, 300, 128, device)
    final = make_dataloader(dataset["finalcheck"], word2index, 300, 128, device)
    covidcheck = make_dataloader(dataset["covidcheck"], word2index, 300, 128, device)


    model = LSTMClassifier(len(word2index), 300, 512, 2, 2)

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    thebest = 0
    bestmodel = model

    if True:
        for epoch in range(50):
            print(f"===Epoch {epoch}===")
            train(train_dloader, model, criterion, optimizer)
            print("Validating...")
            test(val_dloader, model, criterion)
            print("Testing...")
            loss,acc = test(test_dloader, model, criterion)
            save_best(bestmodel,str(epoch) + "-" + str(loss) + "+" + str(acc))
            print("bestmodel",thebest)


    # print("========================")
    # print("========================")
    # model.load_state_dict(torch.load("checkpoints/49-0.01995301469999049+0.996875.pt"))
    # model.eval()
    useagagin(final, model)
    useagagin(covidcheck, model)

if __name__ == "__main__":
    main()
