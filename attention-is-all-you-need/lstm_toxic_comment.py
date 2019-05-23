import pandas as pd;
import numpy as np;
import torch.nn as nn;
import torch;
import torch.autograd as autograd;

# The path of glove embeddings
toxic_comment_input_path = '/home/peng/Workspace/data/kaggle/jigsaw/train.csv';
short_toxic_comment_input_path = '/home/peng/Workspace/data/kaggle/jigsaw/short_train.csv';
input_num = 300;
hidden_num = 100;

class TestLstm(nn.Module):
    def __init__(self):
        super(TestLstm, self).__init__(self); #call the __init__ from super
        self.lstm_layer = nn.LSTM(input_num, hidden_num)
        self.linear_layer = nn.Linear(hidden_num, 6);
        self.softmax_layer = nn.Softmax();

    def forward(self, inputs):
        out, (_, _) = self.lstm_layer(inputs);
        out = self.linear_layer(out);
        out = self.softmax_layer(out);
        return out;


def create_short_train_file ():
    with open (toxic_comment_input_path, 'r') as f_train:
        with open(short_toxic_comment_input_path, 'w') as f_short_train:
            idx = 0;
            max_idx = 500;
            for line in f_train:
                f_short_train.write(line);
                idx += 1;
                if idx == 100:
                    break;




if __name__ == '__main__':
    create_short_train_file();
