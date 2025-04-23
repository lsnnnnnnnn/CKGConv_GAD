import torch
from torch_geometric.nn import GCN
from utils.ckgconv_layer import CKGraphConvLayer
from utils.ckgcola_base import CKGCoLABase  
from pygod.detector.base import DeepDetector


class CKGCoLA(DeepDetector):
    """
    CoLA variant using CKGConv as the backbone.
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
                 save_emb=False,
                 compile_model=False,
                 cfg=None,
                 **kwargs):
        verbose = kwargs.pop("verbose", 0)

        super(CKGCoLA, self).__init__(hid_dim=hid_dim,
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

        self.cfg = cfg
        self.sigmoid_s = sigmoid_s

    def process_graph(self, data):
        pass  

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes, self.hid_dim)

        return CKGCoLABase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            act=self.act,
            backbone=self.backbone,
            out_dim=self.hid_dim,
            sigmoid_s=self.sigmoid_s,
            cfg=self.cfg,
            **kwargs
        ).to(self.device)

    def forward_model(self, data):
        batch_size = data.batch_size
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)

        edge_attr = getattr(data, 'rrwp_val', None)
        edge_index_rrwp = getattr(data, 'rrwp_index', None)

        if edge_attr is not None and edge_index_rrwp is not None:
            edge_index = edge_index_rrwp.to(self.device)
            edge_attr = edge_attr.to(self.device)

        pos_logits, neg_logits = self.model(x, edge_index, edge_attr=edge_attr)

        logits = torch.cat([pos_logits[:batch_size], neg_logits[:batch_size]])
        con_label = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).to(self.device)

        loss = self.model.loss_func(logits, con_label)
        score = neg_logits[:batch_size] - pos_logits[:batch_size]

        return loss, score.detach().cpu()
