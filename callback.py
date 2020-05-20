from keras.callbacks import Callback
import os
import numpy as np



class LogProgress(Callback):
    """
    logs val loss and psnr
    """

    def __init__(self, experiment_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path = "logs/" + experiment_name + ".md"
        self.experiment_name = experiment_name
        os.makedirs("logs", exist_ok=True)
        with open(self.log_path, "w") as f:
            pass

    def _do_log(self, message):
        with open(self.log_path, "a") as f:
            f.write(message)

    def on_train_begin(self, logs=None):
        msg = "# " + self.experiment_name + "\n\n### Validation Loss by Epoch\n"
        self._do_log(msg)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            self._do_log(
                "E{0}: {1:>10.6f}\n".format(epoch, logs["val_loss"])
            )

    def log_psnr(self, results):
        msg = "\n### PSNR\nNoisy: {0}\nDenoised{1}\n".format(results[0], results[1])
        self._do_log(msg)