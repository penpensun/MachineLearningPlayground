import torch.nn as nn;
import torch;
import torch.autograd as autograd;

class TestLstm(nn.Module):
    def __init__(self, seq_num, hidden_num):
        super(TestLstm, self).__init__(); #call the __init__ from super
        self.seq_num = seq_num
        self.hidden_num = hidden_num
        self.lstm_layer = nn.LSTM(seq_num, hidden_num)
        self.linear_layer1 = nn.Linear(hidden_num, 10);
        self.linear_layer2 = nn.Linear(10, 2);
        #self.linear_layer = nn.Linear(hidden_num, 6);
        self.softmax_layer = nn.Softmax(dim = 0);
        #print('class init finished.');

    def forward(self, inputs):
        #print('start forward.');
        out, (hidden, cell_state) = self.lstm_layer(inputs);
        out = self.linear_layer1(out[self.seq_num -1, :, :]).squeeze();
        out = self.linear_layer2(out).squeeze();
        #print('inside network, out shape: ', out.size());
        out = self.softmax_layer(out);
        #print('inside network, out: ',out);
        return out;
