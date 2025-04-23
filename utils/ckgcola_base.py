import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ckgconv_layer import CKGraphConvLayer


class CKGCoLABase(nn.Module):
    """
    CoLA base model using CKGConv as encoder, adapted for contrastive self-supervised learning.
    """

    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.0,
                 act=F.relu,
                 backbone=CKGraphConvLayer,
                 cfg=None,
                 out_dim=None,
                 num_heads=4,
                 **kwargs):
        super(CKGCoLABase, self).__init__()

        assert backbone is not None, "CKGCoLABase requires a backbone (e.g., CKGraphConvLayer)."
        assert cfg is not None, "CKGCoLABase requires a config node (cfg)."

        self.act = act
        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.ModuleList()
        for i in range(num_layers):
            input_dim = in_dim if i == 0 else hid_dim
            self.encoder.append(
                backbone(
                    in_dim=input_dim,
                    out_dim=hid_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    act=cfg.attn.act,
                    cfg=cfg
                )
            )

        # Only use the encoded features in discriminator
        self.discriminator = nn.Bilinear(hid_dim, hid_dim, 1)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.emb = None

    def forward(self, x, edge_index, edge_attr=None):
        for conv in self.encoder:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = self.act(x)
            x = self.dropout(x)

        self.emb = x  # [num_nodes, hid_dim]

        # Positive pair: emb vs emb
        pos_logits = self.discriminator(self.emb, self.emb)

        # Negative pair: permuted emb vs emb
        perm = torch.randperm(x.size(0), device=x.device)
        neg_logits = self.discriminator(self.emb[perm], self.emb)

        return pos_logits.squeeze(), neg_logits.squeeze()

