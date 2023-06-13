import torch
import torch.nn.functional as F
import numpy


class InteractionSelfAttention(torch.nn.Module):
    def __init__(self, args):
        super(InteractionSelfAttention, self).__init__()
        self.args = args

    def forward(self, query, value, mask):
        # attention_states = self.linear_q(query)
        # attention_states_T = self.linear_k(values)
        attention_states = query
        attention_states_T = value
        attention_states_T = attention_states_T.permute([0, 2, 1])

        # padding mask
        # mask = torch.nn.functional.pad(mask, [1, 1])

        weights = torch.bmm(attention_states, attention_states_T)
        weights = weights.masked_fill(mask.unsqueeze(1).expand_as(weights)==0, float("-inf"))
        attention = F.softmax(weights, dim=2)

        # value=self.linear_v(states)
        # merged = torch.bmm(attention, value)
        # merged = merged * mask.unsqueeze(2).float().expand_as(merged)

        return attention

