import torch
import torch.nn as nn
import torch.nn.functional as F



class AttentionLayer(nn.Module):
    def __init__(self, k, num_heads=8):
        super(AttentionLayer, self).__init__()
        self.k = k
        self.num_heads = num_heads

        # determine queries, keys, values
        self.key_layer = nn.Linear(self.k, self.k * self.num_heads, bias=False)
        self.query_layer = nn.Linear(self.k, self.k * self.num_heads, bias=False)
        self.value_layer = nn.Linear(self.k, self.k * self.num_heads, bias=False)

        # project down all cat-ed heads
        self.unify_layer = nn.Linear(self.num_heads * k, k)

    def forward(self, x):
        # get batch size, t sentences of k items
        b_sz, t_sz, k_sz = x.size()
        h_sz = self.num_heads

        keys = self.key_layer(x).view(b_sz, t_sz, h_sz, self.k)
        queries = self.query_layer(x).view(b_sz, t_sz, h_sz, self.k)
        values = self.value_layer(x).view(b_sz, t_sz, h_sz, self.k)

        # compute dot products (k x k). Same op for every head, so fold in to the
        # batch dim
        # q, k, v, (b, t, h, k) -> (b, h, t, k) -> (bh, t, k)
        # and for the key (bh, t, k) -> (bh, k, t) to be able to use bmm
        #
        keys = keys.transpose(1, 2).contiguous().view(b_sz * h_sz, t_sz, k_sz)
        queries = queries.transpose(1, 2).contiguous().view(b_sz * h_sz, t_sz, k_sz)
        values = values.transpose(1, 2).contiguous().view(b_sz * h_sz, t_sz, k_sz)

        # intermediate scaling
        queries = queries / (self.k ** (1. / 4.))
        keys = keys / (self.k ** (1. / 4.))

        # final transpose for the bmm, out -> (b*h, t, t)
        raw_weights = torch.bmm(queries, keys.transpose(1, 2))

        # row wise softmax normalize
        weights = F.softmax(raw_weights, dim=2)

        # apply self attention to the values
        out = torch.bmm(weights, values).view(b_sz, h_sz, t_sz, k_sz)

        # Unify attention heads
        # reshuffle (b, h, t, k) -> (b, t, h, k) -> (b, t, h*k) with all the heads catted
        # ontop of each other to be able to down project
        out = out.transpose(1, 2).contiguous().view(b_sz, t_sz, h_sz * k_sz)

        # project down
        out = self.unify_layer(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, k, num_heads):
        super(TransformerBlock, self).__init__()

        self.attention = AttentionLayer(k, num_heads)

        self.layer_norm1 = nn.LayerNorm(k)
        self.layer_norm2 = nn.LayerNorm(k)

        self.mlp = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )

    def forward(self, x):
        # Attention block
        x_att = self.attention(x)
        # Residual + norm
        x = self.layer_norm1(x + x_att)
        # MLP
        x_mlp = self.mlp(x)
        out = self.layer_norm2(x + x_mlp)
        return out


class Transformer(nn.Module):
    def __init__(self, k, num_heads, walklength, depth, num_classes):
        super(Transformer, self).__init__()
        # Transformer blocks
        self.tf_network = []
        for _ in range(depth):
            self.tf_network.append(TransformerBlock(k, num_heads))

        self.tf_network = nn.Sequential(*self.tf_network)

        # Sequence to class output
        self.fc = nn.Linear(k*walklength, walklength)
        self.dropout1 = nn.Dropout(p=0.35, inplace=False)
        self.output_layer = nn.Linear(walklength, num_classes)

    def forward(self, x):
        # in (b, t) tensor with int values representing words
        # out (b, c) tensor logprobs over c classes

        b_sz, t_sz, k_sz = x.size()

        # Transformer forward
        x = self.tf_network(x)
        x = x.view(b_sz, -1) #reshape
        # Average pool over t dimension and project to class probabilities
        x = self.fc(x)
        x = self.dropout1(x)
        x = self.output_layer(x) #TODO check if I need this one

        # Optional (auto-regressive) transformer
        # no looking ahead, enforce via mask, prior to softmax
        #         indices = torch.triu_indices(t, t, offset=1)
        #         x[:, indices[0], indices[1]] = float('-inf')

        out = F.logsigmoid(x)

        return out