import torch
from .base import DeepDetector
from ..nn import DOMINANTBase
from ..nn.ckgconv_layer import CKGraphConvLayer  


class DOMINANT(DeepDetector):
    """
    Modified DOMINANT using CKGConv as the backbone
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
                 verbose=0,
                 save_emb=False,
                 compile_model=False,
                 **kwargs):

        super(DOMINANT, self).__init__(hid_dim=hid_dim,
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

    def process_graph(self, data):
        DOMINANTBase.process_graph(data)

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes, self.hid_dim)

        return DOMINANTBase(in_dim=self.in_dim,
                            hid_dim=self.hid_dim,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            act=self.act,
                            sigmoid_s=self.sigmoid_s,
                            backbone=self.backbone,
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

