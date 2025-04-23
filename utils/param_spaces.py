class ParamSpace:
    """
    A factory to return parameter space functions for different models.
    """

    @staticmethod
    def get(model_name):
        if model_name.lower() == "dominant":
            return ParamSpace._dominant
        elif model_name.lower() == "cola":
             return ParamSpace._cola
        elif model_name.lower() == "ckgconv_dominant":
             return ParamSpace.ckgconv_dominant
        elif model_name.lower() == "ckgconv_cola":
            return ParamSpace.ckgconv_cola
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    @staticmethod
    def _dominant(trial):
        return {
            "hid_dim": trial.suggest_categorical("hid_dim", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 2, 6),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight": trial.suggest_float("weight", 0.1, 0.9),
        }
    
    @staticmethod
    def _cola(trial):
        return {
            "hid_dim": trial.suggest_categorical("hid_dim", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.01),
            "activation": trial.suggest_categorical("activation", ["relu", "elu"]),
        }
    
    @staticmethod
    def ckgconv_dominant(trial):
        return {
            "hid_dim": trial.suggest_categorical("hid_dim", [64, 128]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-3),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "activation": trial.suggest_categorical("activation", ["relu", "elu", "leaky_relu"]),
            "num_layers": trial.suggest_int("num_layers", 4, 6, step=2),
            "num_heads": trial.suggest_categorical("num_heads", [2, 4, 8]),
            "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
            "ffn": trial.suggest_categorical("ffn", [True, False]),
            "ffn_ratio": trial.suggest_float("ffn_ratio", 1.0, 4.0),
            "n_mlp_blocks": trial.suggest_int("n_mlp_blocks", 1, 3),
            "mlp_dropout": trial.suggest_float("mlp_dropout", 0.0, 0.5),
            "blur_kernel": trial.suggest_categorical("blur_kernel", [True, False]),
            "deg_scaler": trial.suggest_categorical("deg_scaler", [False, True]),
            "value_proj": trial.suggest_categorical("value_proj", [True]),  
        }
    
    @staticmethod
    def ckgconv_cola(trial):
        return {
            "hid_dim": trial.suggest_categorical("hid_dim", [64, 128]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-3),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "activation": trial.suggest_categorical("activation", ["relu", "elu", "leaky_relu"]),
            "num_layers": trial.suggest_int("num_layers", 2, 4),  # 通常CoLA层数较少
            "num_heads": trial.suggest_categorical("num_heads", [2, 4]),
            "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
            "ffn": trial.suggest_categorical("ffn", [True, False]),
            "ffn_ratio": trial.suggest_float("ffn_ratio", 1.0, 4.0),
            "n_mlp_blocks": trial.suggest_int("n_mlp_blocks", 1, 3),
            "mlp_dropout": trial.suggest_float("mlp_dropout", 0.0, 0.5),
            "blur_kernel": trial.suggest_categorical("blur_kernel", [True, False]),
            "deg_scaler": trial.suggest_categorical("deg_scaler", [False, True]),
            "value_proj": trial.suggest_categorical("value_proj", [True]),  # 通常启用
        }
    
    
        




