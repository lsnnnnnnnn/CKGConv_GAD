import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ckgconv_layer import CKGraphConvLayer
from utils.ckgdominant_base import CKGDOMINANTBase, get_cfg_from_trial
from pygod.detector.base import DeepDetector


class CKGDOMINANT(DeepDetector):
    """
    DOMINANT variant using CKGConv as the backbone
    """

    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 sigmoid_s=False,
                 backbone=CKGraphConvLayer,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 weight=0.5,
                 num_heads=4,
                 save_emb=False,
                 compile_model=False,
                 cfg=None,
                 **kwargs):

        verbose = kwargs.pop("verbose", 0)

        super(CKGDOMINANT, self).__init__(hid_dim=hid_dim,
                                          num_layers=num_layers,
                                          dropout=dropout,
                                          weight_decay=weight_decay,
                                          act=act,
                                          backbone=backbone,
                                          contamination=contamination,
                                          lr=lr,
                                          epoch=epoch,
                                          gpu=gpu,
                                          batch_size=batch_size,
                                          num_neigh=num_neigh,
                                          verbose=verbose,
                                          save_emb=save_emb,
                                          compile_model=compile_model,
                                          **kwargs)

        self.weight = weight
        self.sigmoid_s = sigmoid_s
        self.num_heads = num_heads
        self.cfg = cfg

    def process_graph(self, data):
        CKGDOMINANTBase.process_graph(data)

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes, self.hid_dim)

        return CKGDOMINANTBase(in_dim=self.in_dim,
                               hid_dim=self.hid_dim,
                               num_layers=self.num_layers,
                               dropout=self.dropout,
                               act=self.act,
                               sigmoid_s=self.sigmoid_s,
                               backbone=self.backbone,
                               cfg=self.cfg,
                               out_dim=self.hid_dim,
                               num_heads=self.num_heads,
                               **kwargs).to(self.device)

    def forward_model(self, data):
        batch_size = data.batch_size
        node_idx = data.n_id

        x = data.x.to(self.device)
        s = data.s.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_attr = getattr(data, 'rrwp_val', None)
        edge_index_rrwp = getattr(data, 'rrwp_index', None)

        if edge_attr is not None and edge_index_rrwp is not None:
            edge_index = edge_index_rrwp.to(self.device)
            edge_attr = edge_attr.to(self.device)

        x_, s_ = self.model(x, edge_index, edge_attr=edge_attr)

        score = self.model.loss_func(x[:batch_size],
                                     x_[:batch_size],
                                     s[:batch_size, node_idx],
                                     s_[:batch_size],
                                     self.weight)

        loss = torch.mean(score)

        return loss, score.detach().cpu()


if __name__ == "__main__":
    print("CKGraphConvLayer loaded:", CKGraphConvLayer)
    print("CKGDOMINANTBase loaded:", CKGDOMINANTBase)
    print("DeepDetector loaded:", DeepDetector)





