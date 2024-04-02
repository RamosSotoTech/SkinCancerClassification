# src/training/callbacks.py
from tensorflow.keras.callbacks import Callback

class BestValueTracker(Callback):
    def __init__(self, monitor='val_accuracy'):
        super(BestValueTracker, self).__init__()
        self.best_value = 0
        self.best_epoch = 0
        self._monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        if self._monitor not in logs:
            raise ValueError(f"Monitoring metric '{self._monitor}' not found in logs.")

        current_best = logs.get(self._monitor)
        if current_best > self.best_value:
            self.best_value = current_best
            self.best_epoch = epoch