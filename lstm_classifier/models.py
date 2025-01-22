import torch


class LSTMClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, n_classes):
        super(LSTMClassifier, self).__init__()
        self.embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
        self.rnn = torch.nn.LSTM(embedding_dim, hidden_dim, n_layers)
        self.out = torch.nn.Linear(hidden_dim, n_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding_layer(x["input_ids"])
        pcked = torch.nn.utils.rnn.pack_padded_sequence(
            embedded,
            x["length"],
            enforce_sorted=False,
            batch_first=True,
        )
        output, _ = self.rnn(pcked)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            output,
            padding_value=2,
            batch_first=True,
        )
        output = output[range(len(output)), x["length"] - 1]
        logits = self.out(output)
        return self.out(output),self.softmax(logits)

    def forward_nopacked(self, x):
        embedded = self.embedding_layer(x["input_ids"])
        output, _ = self.rnn(embedded)
        logits = self.out(output[:, -1])
        return self.out(output[:, -1]),self.softmax(logits)
