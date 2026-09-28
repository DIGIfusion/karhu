"""
EarlyStopping class for PyTorch
"""
import logging
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    EarlyStopping is a class that implements early stopping for training machine learning models.
    It monitors the validation loss and stops training if it does not improve for a specified
    number of epochs (patience).
    Parameters:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    Attributes:
        best_score (float): The best score observed so far.
        early_stop (bool): Flag indicating whether to stop training.
        counter (int): Counter for the number of epochs without improvement.
        best_model_state (dict): The state of the best model observed so far.
    Methods:
        __call__(val_loss, model): Call this method to check if training should be stopped.
        load_best_model(model): Load the best model state into the provided model.
    Usage:
        early_stopping = EarlyStopping(patience=5, delta=0.0)
        for epoch in range(num_epochs):
            train(...)
            val_loss = validate(...)
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        early_stopping.load_best_model(model)
    """
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        self.best_model_epoch = None

    def __call__(self, val_loss, model, epoch=None):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score > self.best_score + self.delta:
            logger.debug(
                f"{score:.3e} {self.best_score:.3e} {self.best_score + self.delta:.3e}, "
                + f"best score: {self.best_score:.3e}")
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.best_model_epoch = epoch
            self.counter = 0

    def load_best_model(self, model):
        logger.info(f"Loading best model from epoch: {self.best_model_epoch}")
        logger.info(f"Best score: {self.best_score}")
        model.load_state_dict(self.best_model_state)
