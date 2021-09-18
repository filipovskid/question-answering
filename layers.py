import torch
import torch.nn as nn
import torch.nn.functional as F


class InitializedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, relu=False, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.out = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups,
                             bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu == True:
            return F.relu(self.out(x))
        else:
            return self.out(x)


class Embedding(nn.Module):
    """Embedding layer.

    Params:
        word_embeddings: Pre-trained word vectors.
        char_vectors: Character embedding initialization weights of size ().
        char_embed_size: ..
    """

    def __init__(self, word_embeddings, char_embeddings, word_embed_size, char_embed_size):
        super(Embedding, self).__init__()

        self.word_embeddings = nn.Embedding.from_pretrained(word_embeddings)
        self.char_embedding = CharEmbedding(char_embeddings, char_embed_size)
        self.highway = HighwayNetwork(2, 96)

        self.word_dropout = nn.Dropout(0.1)
        self.char_dropout = nn.Dropout(0.05)
        self.conv1d = InitializedConv1d(word_embed_size + 96, 96)

    def forward(self, word_idxs, char_idxs):
        """Parameters:
            word_idxs: torch.Tensor of size (batch_size, num_words). Indices of words in context/question.
            char_idxs: torch.Tensor of size (batch_size, num_words, max_word_len). Indices of characters
                       in context/question.

            :returns torch.Tensor of size ().
        """
        char_embeddings = self.char_embedding(char_idxs)
        char_embeddings = self.char_dropout(char_embeddings)

        word_embeddings = self.word_embeddings(word_idxs)
        word_embeddings = self.word_dropout(word_embeddings)
        word_embeddings = word_embeddings.transpose(1, 2)

        embeddings = torch.cat([char_embeddings, word_embeddings], dim=1)
        embeddings = self.conv1d(embeddings)
        embeddings = self.highway(embeddings)

        return embeddings


class CharEmbedding(nn.Module):
    """Character embedding used as part of the Embedding layer."""

    def __init__(self, char_embeddings, char_embed_size):
        super(CharEmbedding, self).__init__()

        self.char_embedding = nn.Embedding.from_pretrained(char_embeddings, freeze=False)

        # TODO: Parameterize 96
        self.conv2d = nn.Conv2d(char_embed_size, 96, kernel_size=(1, 5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')

    def forward(self, char_idxs):
        """Parameters:
            char_idxs: torch.Tensor of size (batch_size, num_words, max_word_len).

            :returns: torch.Tensor of size (batch_size, 96, num_words).
        """
        batch_size, num_words, max_word_len = char_idxs.size()

        # Size: (batch_size, num_words, max_word_len, char_embed_size)
        x = self.char_embedding(char_idxs)
        x = x.permute(0, 3, 1, 2)

        # Size: (batch_size, 96, max_word_len, 12)
        x = self.conv2d(x)
        x = F.relu(x)
        x, _ = torch.max(x, dim=3)
        x = x.squeeze()

        return x


class HighwayNetwork(nn.Module):

    def __init__(self, num_layers, hidden_size):
        super(HighwayNetwork, self).__init__()

        self.transforms = nn.ModuleList([InitializedConv1d(hidden_size, hidden_size, bias=True)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([InitializedConv1d(hidden_size, hidden_size, bias=True)
                                   for _ in range(num_layers)])

    def forward(self, x):

        for gate, transform in zip(self.gates, self.transforms):
            g = torch.sigmoid(gate(x))
            # t = F.relu(transform(x))
            t = transform(x)
            t = F.dropout(x, p=0.1)
            x = g * t + (1 - g) * x

        return x
