import os, re, unicodedata, numpy as np
import pandas as pd

def unicode_to_ascii(x):
    return "".join(
        c for c in unicodedata.normalize("NFD", x) if unicodedata.category(c) != "Mn"
    )


def normalize_text(x):
    x = unicode_to_ascii(x.lower().strip())
    x = re.sub(r'\.+', r" . ", x)
    x = re.sub(r'\-+', r" - ", x)
    x = re.sub(r'\/+', r" / ", x)
    x = re.sub(r'\++', r" + ", x)
    x = re.sub(r'\\+', r" \ ", x)
    x = re.sub(r'\;+', r" ; ", x)
    x = re.sub(r'\=+', r" = ", x)
    return x


def read_txt_file(file_path):
    with open(file_path, "r") as f:
        return f.readlines()


def create_vocab(lines, min_freq):
    word2index = {"<SOS>": 0, "<EOS>": 1, "<PAD>": 2, "<UNK>": 3}
    word2count = {}
    for sample in lines:
        text = sample.split("\t")[0]

        for w in text.split():
            if w not in word2count:
                word2count[w] = 1
            else:
                word2count[w] += 1

    for w, c in word2count.items():
        if c >= min_freq:
            word2index[w] = len(word2index)
    return word2index

def prepare_dataset(data_dir):
    dataset = []

    fakedf = pd.read_csv('./news/Fake.csv')
    filtered_df = fakedf.iloc[1:][fakedf.iloc[1:, 2] == 'politics']
    b_column_filtered = filtered_df.iloc[:, 1]
    b = b_column_filtered.tolist()
    fakeb_cleaned = [sentence for sentence in b if sentence.strip() and len(sentence.strip()) >= 200]

    print(len(fakeb_cleaned))
    for line in fakeb_cleaned:

        line =  ' '.join(line[i:i + 1] for i in range(0, len(line),1))
        sample = f"{line}\\t0"
        dataset.append(sample)


    truedf = pd.read_csv('./news/True.csv')
    filtered_df = truedf.iloc[1:][truedf.iloc[1:, 2] == 'politicsNews']
    b_column_filtered = filtered_df.iloc[:, 1]
    b = b_column_filtered.tolist()
    b_cleaned = [sentence for sentence in b if sentence.strip() and len(sentence.strip()) >= 200]

    print(len(b_cleaned))
    truecount = 0
    for line in b_cleaned:

        line =  ' '.join(line[i:i + 1] for i in range(0, len(line),1))
        sample = f"{line}\\t1"
        dataset.append(sample)
        truecount +=1
        if(truecount == len(fakeb_cleaned)):
            break

    print(len(dataset))


    covidlist = []
    fakedf = pd.read_csv('./covid/fakeNews.csv')
    c_column_filtered = fakedf.iloc[1:, 2]
    c = c_column_filtered.tolist()
    fakeb_cleaned = [sentence for sentence in c if sentence.strip() and len(sentence.strip()) >= 100]

    print(len(fakeb_cleaned))
    for line in fakeb_cleaned:

        line =  ' '.join(line[i:i + 1] for i in range(0, len(line),1))
        sample = f"{line}\\t0"
        covidlist.append(sample)

    truedf = pd.read_csv('./covid/trueNews.csv')
    c_column_filtered = truedf.iloc[1:, 2]
    c = c_column_filtered.tolist()
    b_cleaned = [sentence for sentence in c if sentence.strip() and len(sentence.strip()) >= 200]

    print(len(b_cleaned))
    truecount = 0
    for line in b_cleaned:

        line =  ' '.join(line[i:i + 1] for i in range(0, len(line),1))
        sample = f"{line}\\t1"
        covidlist.append(sample)
        truecount +=1
        if(truecount == len(fakeb_cleaned)):
            break


    print(len(covidlist))
    np.random.shuffle(covidlist)
    np.random.shuffle(dataset)
    #
    word2index = create_vocab(dataset, 1)
    print(f"Vocab size:", len(word2index))

    # create train/val/test
    n_train = int(len(dataset) * 0.7)
    n_val = int(len(dataset) * 0.1)
    n_final = int(len(dataset) * 0.1)
    dataset = {
        "train": dataset[:n_train],
        "val": dataset[n_train : n_train + n_val],
        "test": dataset[ n_train + n_val: n_train + n_final + n_final],
        "finalcheck" :  dataset[-n_final: ],
        "covidcheck" : covidlist
    }
    print(len(dataset["train"]))
    return dataset, word2index
