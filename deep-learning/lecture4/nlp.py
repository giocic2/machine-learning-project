import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

def to_idxs(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ("The cat ate the orange".split(), ["Det", "Noun", "Verb", "Det", "Noun"]),
    ("John ate that sandwich".split(), ["Noun", "Verb", "Det", "Noun"]),
    ("Tom drive the car".split(), ["Noun", "Verb", "Det", "Noun"]),
    ("The mouse ate the cheese".split(), ["Det", "Noun", "Verb", "Det", "Noun"])

]
word2ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word2ix:
            word2ix[word] = len(word2ix)
print(word2ix)
tag2ix = {"Det": 0, "Noun": 1, "Verb": 2}

EMB_DIM = 10
H_DIM = 5

class LSTMExample(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, t_size):
        super(LSTMExample, self).__init__()
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, t_size)

    def forward(self, sentence):
        embeds = self.embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(1, len(sentence), -1))
        fc_out = self.fc(lstm_out.view(len(sentence), -1))
        scores = F.log_softmax(fc_out, dim=1)
        return scores


model = LSTMExample(EMB_DIM, H_DIM, len(word2ix), len(tag2ix))
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

with torch.no_grad():
    inputs = to_idxs(training_data[0][0], word2ix)
    scores = model(inputs)
    print(scores)

for epoch in range(200):
    for sentence, tags in training_data:

        input = to_idxs(sentence, word2ix)
        targets = to_idxs(tags, tag2ix)

        scores = model(input)

        loss = loss_function(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

with torch.no_grad():
    inputs = to_idxs(training_data[1][0], word2ix)
    tag_scores = model(inputs)

    print(torch.argmax(tag_scores, dim=1))
    print(tag_scores)

