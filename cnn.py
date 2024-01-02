import torch.nn.functional as F
import torch
from torch import nn







#new class: https://www.kaggle.com/code/mlwhiz/multiclass-text-classification-pytorch

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, dropout, num_filters, filter_sizes):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (size, embedding_dim)) for size in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        x = self.dropout(x)
        x = self.fc(x)
        return x
