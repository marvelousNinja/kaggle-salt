from salt.callbacks.callback import Callback

noop = lambda logs, outputs, gt: None

class LambdaCallback(Callback):
    def __init__(self, on_train_batch_end=noop, on_validation_end=noop):
        self.on_train_batch_end_handler = on_train_batch_end
        self.on_validation_end_handler = on_validation_end

    def on_train_batch_end(self, logs, outputs, gt):
        self.on_train_batch_end_handler(logs, outputs, gt)

    def on_validation_end(self, logs, outputs, gt):
        self.on_validation_end_handler(logs, outputs, gt)
