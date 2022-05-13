
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

seq_length = 20

time_steps = np.linspace(0, np.pi, seq_length + 1)
data = np.sin(time_steps)
data.resize((seq_length + 1, 1))

x = data[:-1]
y = data[1:]

plt.plot(time_steps[1:], x, 'r.', label='input, x') # x
plt.plot(time_steps[1:], y, 'b.', label='target, y') # y

plt.legend(loc='best')
plt.show()


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        print("int. shape:, ", x.shape)
        r_out, hidden = self.rnn(x, hidden)
        r_out = r_out.view(-1, self.hidden_dim)

        output = self.fc(r_out)

        return output, hidden


test_rnn = RNN(input_size=1, output_size=1, hidden_dim=10, n_layers=2)

time_steps = np.linspace(0, np.pi, seq_length)
data = np.sin(time_steps)
data.resize((seq_length, 1))

test_input = torch.Tensor(data).unsqueeze(0) 
print('Input size: ', test_input.size())

test_out, test_h = test_rnn(test_input, None)
print('Output size: ', test_out.size())
print('Hidden state size: ', test_h.size())



input_size=1
output_size=1
hidden_dim=32
n_layers=1

rnn = RNN(input_size, output_size, hidden_dim, n_layers)
print(rnn)



criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)


def train(rnn, n_steps, print_every):
    hidden = None

    for batch_i, step in enumerate(range(n_steps)):
        time_steps = np.linspace(step * np.pi, (step + 1) * np.pi, seq_length + 1)
        data = np.sin(time_steps)
        data.resize((seq_length + 1, 1))

        x = data[:-1]
        y = data[1:]

        x_tensor = torch.Tensor(x).unsqueeze(0)
        y_tensor = torch.Tensor(y)

        prediction, hidden = rnn(x_tensor, hidden)

        hidden = hidden.data

        loss = criterion(prediction, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_i % print_every == 0:
            print('Loss: ', loss.item())
            plt.plot(time_steps[1:], x, 'r.', label="input seq")  # input
            plt.plot(time_steps[1:], prediction.data.numpy().flatten(), 'b.', label="predicted seq")  # predictions
            plt.plot(time_steps[1:], y_tensor.data.numpy().flatten(), 'g.', label="GT")  # ground truth
            plt.legend()
            plt.show()

    return rnn



n_steps = 75
print_every = 1

trained_rnn = train(rnn, n_steps, print_every)

