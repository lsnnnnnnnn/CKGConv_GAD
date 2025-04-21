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
             return ParamSpace._ckgconv_dominant
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
        }


