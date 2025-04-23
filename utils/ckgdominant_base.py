import torch
import torch.nn as nn
import torch.nn.functional as F
from pygod.nn.decoder import DotProductDecoder
from pygod.nn.functional import double_recon_loss
from yacs.config import CfgNode as CN


from yacs.config import CfgNode as CN

def get_cfg_from_trial(trial_or_dict):
    cfg = CN()
    cfg.attn = CN()

    
    if isinstance(trial_or_dict, dict):
        def get(name, default):
            return trial_or_dict.get(name, default)
    else:
        def get(name, default):
            
            suggest_fn = {
                bool: trial_or_dict.suggest_categorical,
                int: trial_or_dict.suggest_int,
                float: trial_or_dict.suggest_float,
                str: trial_or_dict.suggest_categorical
            }
            
            param_space = {
                "batch_norm": bool,
                "ffn": bool,
                "ffn_ratio": float,
                "n_mlp_blocks": int,
                "mlp_dropout": float,
                "blur_kernel": bool,
                "deg_scaler": bool,
            }
            if name in param_space:
                typ = param_space[name]
                if typ == bool:
                    return suggest_fn[bool](name, [True, False])
                elif typ == float:
                    return suggest_fn[float](name, 1.0, 4.0)
                elif typ == int:
                    return suggest_fn[int](name, 1, 3)
                elif typ == str:
                    return suggest_fn[str](name, ["gelu"])
            return default

    
    cfg.attn.batch_norm = get("batch_norm", True)
    cfg.attn.ffn = get("ffn", True)
    cfg.attn.ffn_ratio = get("ffn_ratio", 2.0)
    cfg.attn.n_mlp_blocks = get("n_mlp_blocks", 2)
    cfg.attn.mlp_dropout = get("mlp_dropout", 0.1)
    cfg.attn.blur_kernel = get("blur_kernel", False)
    cfg.attn.deg_scaler = get("deg_scaler", False)
    cfg.attn.use_bias = True
    cfg.attn.weight_norm = False
    cfg.attn.graph_norm = False
    cfg.attn.kernel_norm = False
    cfg.attn.group_norm = False
    cfg.attn.average = True
    cfg.attn.dynamic_avg = False
    cfg.attn.clamp = 5.0
    cfg.attn.act = "gelu"
    cfg.attn.value_proj = trial_or_dict.get("value_proj", True)
    cfg.attn.out_proj = True

    return cfg



class CKGDOMINANTBase(nn.Module):
    """
    DOMINANT base model using CKGConv as encoder.
    """

    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.0,
                 act=F.relu,
                 sigmoid_s=False,
                 backbone=None,
                 cfg=None,
                 out_dim=None,
                 num_heads=4,
                 **kwargs):
        super().__init__()

        assert backbone is not None, "You must provide a backbone (CKGraphConvLayer)."
        assert cfg is not None, "You must provide a configuration for CKGConv (cfg)."

        encoder_layers = num_layers // 2
        decoder_layers = num_layers - encoder_layers

        
        act_name = cfg.attn.get("act", "relu")
        act_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU()
        }
        self.act = act_map.get(act_name, nn.ReLU())

        self.encoder = nn.ModuleList()
        for i in range(encoder_layers):
            input_dim = in_dim if i == 0 else hid_dim
            self.encoder.append(
                backbone(
                    in_dim=input_dim,
                    out_dim=hid_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    act=act_name,
                    cfg=cfg
                )
            )

        self.attr_decoder = nn.Linear(hid_dim, in_dim)
        self.struct_decoder = DotProductDecoder(
            in_dim=hid_dim,
            hid_dim=hid_dim,
            num_layers=max(1, decoder_layers - 1),
            dropout=dropout,
            act=self.act,
            sigmoid_s=sigmoid_s
        )

        self.loss_func = double_recon_loss
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        for conv in self.encoder:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = self.act(x)
            x = self.dropout(x)

        self.emb = x  

        x_ = self.attr_decoder(x)
        s_ = self.struct_decoder(x, edge_index)
        return x_, s_

    @staticmethod
    def process_graph(data):
        from torch_geometric.utils import to_dense_adj
        data.s = to_dense_adj(data.edge_index)[0]
