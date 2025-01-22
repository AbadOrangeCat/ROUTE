from random import shuffle
import torch


class AmazonReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, data, word2index, max_sent_length, device):
        self.data = data
        self.word2index = word2index
        self.max_sent_length = max_sent_length
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        text, label = sample.split("\\t")
        label = int(label)
        input_ids, _text, text_length = self.text2indices(text)
        return {
            "text": _text,
            "input_ids": torch.tensor(input_ids).to(self.device),
            "label": torch.tensor(label).to(self.device),
            "length": text_length,
        }

    def text2indices(self, x):
        sos_id = self.word2index["<SOS>"]
        eos_id = self.word2index["<EOS>"]
        pad_id = self.word2index["<PAD>"]
        unk_id = self.word2index["<UNK>"]
        input_ids, text = [sos_id], ["<SOS>"]
        for w in x.split():
            token_id = self.word2index.get(w, unk_id)
            input_ids.append(token_id)
            if token_id == unk_id:
                text.append("<UNK>")
            else:
                text.append(w)
        if len(input_ids) >= self.max_sent_length:  # truncate
            text = text[: self.max_sent_length]
            text.append("<EOS>")
            input_ids = input_ids[: self.max_sent_length]
            input_ids.append(eos_id)
            text_len = len(input_ids)
        else:  # pad
            to_add = self.max_sent_length - len(input_ids)
            text.append("<EOS>")
            input_ids.append(eos_id)
            text_len = len(input_ids)
            text.extend(["<PAD>"] * to_add)
            input_ids.extend([pad_id] * to_add)
        return input_ids, " ".join(text), text_len


def make_dataloader(dataset, word2index, max_sent_length, batch_size, device):
    ds = AmazonReviewsDataset(dataset, word2index, max_sent_length, device)
    return torch.utils.data.DataLoader(ds, batch_size, shuffle=True)
