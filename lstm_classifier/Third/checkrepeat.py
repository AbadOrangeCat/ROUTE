import pandas as pd

train = pd.read_csv("targetdata_clean/train.csv")
val   = pd.read_csv("targetdata_clean/val.csv")
test  = pd.read_csv("targetdata_clean/test.csv")

def norm(s):
    return str(s).strip()

tr = set(train["text"].map(norm))
va = set(val["text"].map(norm))
te = set(test["text"].map(norm))

print("overlap train-test:", len(tr & te), "/", len(te))
print("overlap train-val :", len(tr & va), "/", len(va))
print("overlap val-test  :", len(va & te), "/", len(te))
