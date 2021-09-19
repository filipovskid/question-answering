import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import masked_softmax


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
        if self.relu:
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
            word_idxs: torch.Tensor of size (batch_size, num_words). Indices of words in context/query.
            char_idxs: torch.Tensor of size (batch_size, num_words, max_word_len). Indices of characters
                       in context/query.

            :returns torch.Tensor of size (batch_size, 96, num_words).
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


class EncoderBlock(nn.Module):
    """QANet encoder block."""

    # TODO: Parameterize 96

    def __init__(self, num_convs, input_dim, kernel_size, num_heads, num_blocks, block_index):
        super(EncoderBlock, self).__init__()

        self.num_convs = num_convs
        self.layers_per_block = num_convs + 1
        self.L = self.layers_per_block * num_blocks  # Total # of layers
        self.l = float(self.layers_per_block * block_index + 1)

        # Layers within residual blocks containing depthwise separable convolution
        self.conv_layernorms = nn.ModuleList([nn.LayerNorm(96) for _ in range(num_convs)])
        self.depthwise_convs = nn.ModuleList(
            [DepthwiseSeparableConvolution(input_dim, kernel_size) for _ in range(num_convs)])

        # Layers within residual block containing self-attention
        self.attention_layernorm = nn.LayerNorm(96)
        self.attention = SelfAttention(input_dim, num_heads)

        # Layers within residual block containing Feedforward layer
        self.feedforward_layernorm = nn.LayerNorm(96)
        self.feedforward = FeedForward(input_dim)

    def forward(self, x, padding_mask):
        """

        Parameters:
            x:
            padding_mask: torch.Tensor representing a binary mask where a value True denotes a padding element.

            :returns: torch.Tensor of size (batch_size, 96, num_words)
        """
        # print('input:', x.size())
        x = PositionEncoder(x)

        # print('PositionEncoder:', x.size())
        # print('---- Residual block ----')
        # First num_convolution residual blocks
        for i, (depthwise_conv, layernorm) in enumerate(zip(self.depthwise_convs, self.conv_layernorms)):
            residual = x
            x = layernorm(x.transpose(1, 2)).transpose(1, 2)
            # print('depthwise_layernorm:', x.size())

            if i % 2 == 0:
                x = F.dropout(x, p=0.1, training=self.training)

            x = depthwise_conv(x)
            # print('depthwise:', x.size())
            x = self.layer_dropout(x, residual, dropout=0.1 * self.l / self.L)
            # print('layer_dropout:', x.size())

            self.l += 1

        # print('---- Residual block ----')
        # Self-attention residual block
        residual = x
        x = self.attention_layernorm(x.transpose(1, 2)).transpose(1, 2)
        # print('attention_layernorm:', x.size())
        x = F.dropout(x, p=0.1, training=self.training)
        # print('att_dropout:', x.size())
        x = self.attention(x, padding_mask)
        # print('attention:', x.size())
        x = self.layer_dropout(x, residual, 0.1 * self.l / self.L)
        # print('attention layer_dropout:', x.size())

        self.l += 1

        # print('---- Residual block ----')
        residual = x
        x = self.feedforward_layernorm(x.transpose(1, 2)).transpose(1, 2)
        # print('feedforward layernorm:', x.size())
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.feedforward(x)
        # print('feedforward:', x.size())
        x = self.layer_dropout(x, residual, 0.1 * self.l / self.L)
        # print('feedforward layer_dropout:', x.size())

        return x

    def layer_dropout(self, inputs, residual, dropout):
        if self.training:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


def PositionEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    x = x.transpose(1, 2)
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + signal).transpose(1, 2)


def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


class DepthwiseSeparableConvolution(nn.Module):
    """Depthwise Separable Convolution used in QANet encoder block."""

    def __init__(self, input_dim, kernel_size, bias=True):
        super(DepthwiseSeparableConvolution, self).__init__()

        self.depthwise = nn.Conv1d(input_dim, input_dim, kernel_size, padding=kernel_size // 2, groups=input_dim,
                                   bias=False)
        self.pointwise = nn.Conv1d(input_dim, input_dim, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        """
        Parameters:
            x:
            :returns: torch.Tensor of size (batch_size, 96, num_words)
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = F.relu(x)

        return x


class SelfAttention(nn.Module):

    def __init__(self, input_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)

    def forward(self, x, key_padding_mask=None):
        """
        Parameters
            key_padding_mask:  If provided, specified padding elements in the key will be ignored by the attention.
                               When given a binary mask and a value is True, the corresponding value on the attention
                               layer will be ignored. When given a byte mask and a value is non-zero, the corresponding
                               value on the attention layer will be ignored.

            :returns: torch.Tensor of size (batch_size, 96, num_words)
        """
        x = x.permute(2, 0, 1)
        x, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = x.permute(1, 2, 0)

        return x


class FeedForward(nn.Module):

    def __init__(self, input_dim):
        super(FeedForward, self).__init__()

        self.non_linear_conv = InitializedConv1d(input_dim, input_dim, relu=True, bias=True)
        self.linear_conv = InitializedConv1d(input_dim, input_dim, relu=True, bias=True)

    def forward(self, x):
        """
        Parameters:
            x:
            :returns: torch.Tensor of size (batch_size, 96, num_words)
        """
        x = self.non_linear_conv(x)
        x = self.linear_conv(x)

        return x


class ContextQueryAttention(nn.Module):

    def __init__(self, hidden_size=96):
        super(ContextQueryAttention, self).__init__()

        self.context_weights = nn.Parameter(torch.empty(hidden_size, 1))
        self.query_weights = nn.Parameter(torch.empty(hidden_size, 1))
        self.cq_weights = nn.Parameter(torch.empty(1, 1, hidden_size))
        for weight in (self.context_weights, self.query_weights, self.cq_weights):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        """
        Parameters
            c: torch.Tensor of size (batch_size, hidden_size, c_len[num_words])
            q: torch.Tensor of size (batch_size, hidden_size, q_len[num_words])
            c_mask: torch.Tensor of size (batch_size, c_len[num_words])
            q_mask: torch.Tensor of size (batch_size, q_len[num_words])

            :returns: torch.Tensor of size (batch_size, c_len[num_words], 4 * hidden_size)
        """

        c, q = c.transpose(1, 2), q.transpose(1, 2)
        batch_size, c_len, q_len = c.size()[0], c.size()[1], q.size()[1]
        S = self.compute_similarity_matrix(c, q)

        c_mask = c_mask.view(batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)
        s1 = masked_softmax(S, q_mask, dim=2)
        s2 = masked_softmax(S, c_mask, dim=1)

        a = torch.bmm(s1, q)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)
        x = torch.cat([c, a, c * a, c * b], dim=2)

        return x

    def compute_similarity_matrix(self, c, q):
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, p=0.1, training=self.training)
        q = F.dropout(q, p=0.1, training=self.training)

        s0 = torch.matmul(c, self.context_weights).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.query_weights).transpose(1, 2).expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weights, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class EncoderLayer(nn.Module):
    """Encoder layer wrapping multiple encoder blocks used. It is used to construct
    the embedding encoder layer and model encoder layer in QANet.
    """

    def __init__(self, num_convs, input_dim, kernel_size, num_heads, num_blocks):
        super(EncoderLayer, self).__init__()

        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(num_convs, input_dim, kernel_size, num_heads, num_blocks, block_index=i)
             for i in range(num_blocks)])

    def forward(self, x, key_padding_mask=None):

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, key_padding_mask)

        return x


