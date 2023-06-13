import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
from .interaction_attention import InteractionSelfAttention
from .gcn import GCNModel


class FusionLayer(nn.Module):

    def __init__(self, config, gen_emb, domain_emb):
        super().__init__()

        self.config = config
        # Double embedding
        self.general_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.general_embedding.weight.data.copy_(gen_emb)
        self.general_embedding.weight.requires_grad = False
        self.domain_embedding = torch.nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight.data.copy_(domain_emb)
        self.domain_embedding.weight.requires_grad = False

        # Pos embedding
        self.pos_embedding = torch.nn.Embedding(5, config.pos_dim)

        # Dropout
        self.dropout1 = torch.nn.Dropout(0.3)
        self.dropout2 = torch.nn.Dropout(0.25)

        # Bilstm encoder
        self.bilstm_encoder = torch.nn.LSTM(300 + 500 + config.pos_dim, config.lstm_dim, num_layers=1, batch_first=True,
                                            bidirectional=True)
        self.bilstm = torch.nn.LSTM(config.lstm_dim * 2, config.lstm_dim, num_layers=1, batch_first=True,
                                    bidirectional=True)

        # GCN Module
        self.gcn_layer = GCNModel(config)
        # self.attention_layer = SelfAttention(config)
        self.interaction_attention = InteractionSelfAttention(config)

        # self.feature_linear = torch.nn.Linear(config.lstm_dim * 4 + config.class_num * 3, config.lstm_dim * 4)
        # self.cls_linear = torch.nn.Linear(config.lstm_dim * 4, config.class_num)

    def _get_double_embedding(self, sentence_tokens, mask):
        general_embedding = self.general_embedding(sentence_tokens)
        domain_embedding = self.domain_embedding(sentence_tokens)
        embedding = torch.cat([general_embedding, domain_embedding], dim=2)
        embedding = self.dropout1(embedding)
        embedding = embedding * mask.unsqueeze(2).float().expand_as(embedding)
        return embedding

    def _get_pos_embedding(self, sentence_poses, mask):
        pos_embedding = self.pos_embedding(sentence_poses)
        # pos_embedding = self.dropout1(pos_embedding)
        embedding = pos_embedding * mask.unsqueeze(2).float().expand_as(pos_embedding)
        return embedding

    def _lstm_feature(self, embedding, lengths):
        embedding = pack_padded_sequence(embedding, lengths.cpu(), batch_first=True, enforce_sorted=False)
        context, _ = self.bilstm_encoder(embedding)
        context, _ = pad_packed_sequence(context, batch_first=True)
        return context

    def _lstm_feature1(self, embedding, lengths):
        embedding = pack_padded_sequence(embedding, lengths.cpu(), batch_first=True, enforce_sorted=False)
        context, _ = self.bilstm(embedding)
        context, _ = pad_packed_sequence(context, batch_first=True)
        return context

    def forward(self, bert_feature, sentence_tokens, pos_pack, adj_pack, lengths, masks):
        # Padding mask
        mask = torch.nn.functional.pad(masks, [1, 1])
        # mask = masks

        # Obtain Embedding
        double_embedding = self._get_double_embedding(sentence_tokens, masks)
        pos_embedding = self._get_pos_embedding(pos_pack, masks)
        embedding = torch.cat([double_embedding, pos_embedding], dim=2)

        # Bilstm encoder
        lstm_feature = self._lstm_feature(embedding, lengths)
        # GCN encoder
        lstm_feature = torch.nn.functional.pad(lstm_feature, [0, 0, 1, 1])
        adj = torch.nn.functional.pad(adj_pack, [1, 1, 1, 1])
        # adj = adj_pack
        gcn_feature = self.gcn_layer(lstm_feature, adj, mask)

        a = bert_feature
        b = gcn_feature

        # Interaction layer
        for i in range(2):
            # interaction attention
            gcn_interaction_attention = self.interaction_attention(gcn_feature, gcn_feature, mask)
            bert_interaction_attention = self.interaction_attention(bert_feature, bert_feature, mask)

            bert_interaction_feature = torch.bmm(gcn_interaction_attention, bert_feature)
            bert_interaction_feature = bert_interaction_feature * mask.unsqueeze(2).float().expand_as(bert_interaction_feature)

            gcn_interaction_feature = torch.bmm(bert_interaction_attention, gcn_feature)
            gcn_interaction_feature = gcn_interaction_feature * mask.unsqueeze(2).float().expand_as(gcn_interaction_feature)

            bert_interaction_feature_drop = self.dropout2(bert_interaction_feature) + a
            gcn_interaction_feature_drop = self.dropout2(gcn_interaction_feature) + b
            # bert_interaction_feature_drop = self.dropout2(bert_interaction_feature)
            # gcn_interaction_feature_drop = self.dropout2(gcn_interaction_feature)

            bert_feature = bert_interaction_feature_drop

            gcn_feature = self._lstm_feature1(gcn_interaction_feature_drop, lengths+2)
            gcn_feature = self.gcn_layer(gcn_feature, adj, mask)

        # feature = bert_feature + a
        feature = bert_feature
        # feature = torch.nn.functional.pad(bert_feature, [0, 0, 1, 1])
        return feature
