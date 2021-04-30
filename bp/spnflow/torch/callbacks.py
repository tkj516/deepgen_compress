import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given number of consecutive epochs"""
    def __init__(self, patience=1, delta=1e-3):
        """
        Instantiate an EarlyStopping object.

        :param patience: The number of consecutive epochs to wait.
        :param delta: The minimum change of the monitored quantity.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.should_stop = False
        self.counter = 0

    def __call__(self, loss):
        """
        Call the object.

        :param loss: The validation loss measured.
        """
        # Check if an improved of the loss happened
        if loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        # Check if the training should stop
        if self.counter >= self.patience:
            self.should_stop = True
