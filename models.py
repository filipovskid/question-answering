import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Embedding, EncoderLayer, ContextQueryAttention, InitializedConv1d, QuestionAnsweringOutput, \
    ConditionalQuestionAnsweringOutput


class QANet(nn.Module):

    def __init__(self, word_embeddings, char_embeddings, word_embed_size, char_embed_size, hidden_size,
                 embed_encoder_num_convs, embed_encoder_kernel_size, embed_encoder_num_heads,
                 embed_encoder_num_blocks, model_encoder_num_convs, model_encoder_kernel_size,
                 model_encoder_num_heads, model_encoder_num_blocks):
        super(QANet, self).__init__()

        self.embedding = Embedding(word_embeddings=word_embeddings,
                                   char_embeddings=char_embeddings,
                                   word_embed_size=word_embed_size,
                                   char_embed_size=char_embed_size,
                                   hidden_size=hidden_size)
        self.embedding_encoder = EncoderLayer(num_convs=embed_encoder_num_convs,
                                              hidden_size=hidden_size,
                                              kernel_size=embed_encoder_kernel_size,
                                              num_heads=embed_encoder_num_heads,
                                              num_blocks=embed_encoder_num_blocks)
        self.cq_attention = ContextQueryAttention(hidden_size=hidden_size)
        self.cq_resizer = InitializedConv1d(hidden_size * 4, hidden_size)
        self.model_encoder = EncoderLayer(num_convs=model_encoder_num_convs,
                                          hidden_size=hidden_size,
                                          kernel_size=model_encoder_kernel_size,
                                          num_heads=model_encoder_num_heads,
                                          num_blocks=model_encoder_num_blocks)
        self.output = QuestionAnsweringOutput(hidden_size)

    def forward(self, context_idxs, context_char_idxs, query_idxs, query_char_idxs):
        context_padding_mask = torch.zeros_like(context_idxs) == context_idxs
        query_padding_mask = torch.zeros_like(query_idxs) == query_idxs

        c = self.embedding(context_idxs, context_char_idxs)
        c = self.embedding_encoder(c, context_padding_mask)

        q = self.embedding(query_idxs, query_char_idxs)
        q = self.embedding_encoder(q, query_padding_mask)

        x = self.cq_attention(c, q, context_padding_mask, query_padding_mask)
        x = self.cq_resizer(x)
        x = F.dropout(x, p=0.1, training=self.training)

        M0 = self.model_encoder(x, context_padding_mask)
        M1 = self.model_encoder(F.dropout(M0, p=0.1, training=self.training), context_padding_mask)
        M2 = self.model_encoder(F.dropout(M1, p=0.1, training=self.training), context_padding_mask)

        y1, y2 = self.output(M0, M1, M2, context_padding_mask)

        return y1, y2


class ConditionalQANet(nn.Module):

    def __init__(self, word_embeddings, char_embeddings, word_embed_size, char_embed_size, hidden_size,
                 embed_encoder_num_convs, embed_encoder_kernel_size, embed_encoder_num_heads,
                 embed_encoder_num_blocks, model_encoder_num_convs, model_encoder_kernel_size,
                 model_encoder_num_heads, model_encoder_num_blocks):
        super(ConditionalQANet, self).__init__()

        self.embedding = Embedding(word_embeddings=word_embeddings,
                                   char_embeddings=char_embeddings,
                                   word_embed_size=word_embed_size,
                                   char_embed_size=char_embed_size,
                                   hidden_size=hidden_size)
        self.embedding_encoder = EncoderLayer(num_convs=embed_encoder_num_convs,
                                              hidden_size=hidden_size,
                                              kernel_size=embed_encoder_kernel_size,
                                              num_heads=embed_encoder_num_heads,
                                              num_blocks=embed_encoder_num_blocks)
        self.cq_attention = ContextQueryAttention(hidden_size=hidden_size)
        self.cq_resizer = InitializedConv1d(hidden_size * 4, hidden_size)
        self.model_encoder = EncoderLayer(num_convs=model_encoder_num_convs,
                                          hidden_size=hidden_size,
                                          kernel_size=model_encoder_kernel_size,
                                          num_heads=model_encoder_num_heads,
                                          num_blocks=model_encoder_num_blocks)
        self.output = ConditionalQuestionAnsweringOutput(hidden_size)

    def forward(self, context_idxs, context_char_idxs, query_idxs, query_char_idxs):
        context_padding_mask = torch.zeros_like(context_idxs) == context_idxs
        query_padding_mask = torch.zeros_like(query_idxs) == query_idxs

        c = self.embedding(context_idxs, context_char_idxs)
        c = self.embedding_encoder(c, context_padding_mask)

        q = self.embedding(query_idxs, query_char_idxs)
        q = self.embedding_encoder(q, query_padding_mask)

        x = self.cq_attention(c, q, context_padding_mask, query_padding_mask)
        x = self.cq_resizer(x)
        x = F.dropout(x, p=0.1, training=self.training)

        M0 = self.model_encoder(x, context_padding_mask)
        M1 = self.model_encoder(F.dropout(M0, p=0.1, training=self.training), context_padding_mask)
        M2 = self.model_encoder(F.dropout(M1, p=0.1, training=self.training), context_padding_mask)

        y1, y2 = self.output(M0, M1, M2, context_padding_mask)

        return y1, y2
