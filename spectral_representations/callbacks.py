import time
from pathlib import Path
import pandas as pd
import tensorflow as tf


class SaveHistory(tf.keras.callbacks.Callback):
    def __init__(self, output, additional_logs_callback=None):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.filename_logs = output / "data.csv"
        self.filename_model = output / "model_save"
        self.filename_model_best = output / "model_save_best"
        self.filename_model_best_data = output / "model_save_best_data.csv"

        self.logs_history = []
        if not isinstance(additional_logs_callback, (list, tuple)):
            additional_logs_callback = [additional_logs_callback]
        self.additional_logs_callback = additional_logs_callback

        self.best_val_acc = 0

    def on_train_begin(self, logs={}):
        self.on_epoch_end(-1, logs)

    def on_epoch_end(self, epoch, logs={}):
        # add epoch and time to the logs
        logs["epoch"] = epoch
        logs["time"] = time.time()
        # maybe add additional metrics
        if self.additional_logs_callback is not None:
            for func in self.additional_logs_callback:
                if func is not None:
                    logs.update(func(self.model))
        # store the logs
        self.logs_history.append(logs)

        # save the logs to a csv file
        self.filename_logs.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.logs_history).to_csv(self.filename_logs, index=False)

        # save the current model
        Path(self.filename_model).mkdir(parents=True, exist_ok=True)
        self.model.save(self.filename_model)

        # if we found something better than the previous "best"
        if "val_accuracy" in logs and logs["val_accuracy"] > self.best_val_acc:
            self.best_val_acc = logs["val_accuracy"]
            Path(self.filename_model_best).mkdir(parents=True, exist_ok=True)
            self.model.save(self.filename_model_best)
            Path(self.filename_model_best_data.parent).mkdir(parents=True, exist_ok=True)
            pd.DataFrame([logs]).to_csv(self.filename_model_best_data, index=False)


class SaveSpectrum(tf.keras.callbacks.Callback):
    def __init__(self, output, dataset):
        self.dataset = dataset
        self.output = output

    def on_epoch_end(self, epoch, logs={}):
        print("on_epoch_end, save spectrum", Path(self.output) / f"_spectrum-{epoch}")
        x_test, y_test = self.dataset

        for layer in self.model.layers:
            layer.save_spectrum = Path(self.output) / f"_spectrum-{epoch}"

        self.model(x_test)

        for layer in self.model.layers:
            layer.save_spectrum = None
