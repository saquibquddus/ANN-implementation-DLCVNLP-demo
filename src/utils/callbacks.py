import os
import tensorflow as tf
import numpy as np
import logging
from src.utils.model import get_timestamp


def get_callbacks(config, X_train):
    logging.info("\n tensorboard implementation started \n")
    logs= config["logs"]
    unique_tb_name= get_timestamp(logs["tensorboard_name"])
    tensorboard_root_log_dir= os.path.join(logs["logs_dir"], logs["tensorboard_root_log_dir"])
    os.makedirs(tensorboard_root_log_dir, exist_ok=True)
    tensorboard_path=os.path.join(tensorboard_root_log_dir, unique_tb_name)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path)
    file_writer = tf.summary.create_file_writer(logdir=tensorboard_path)
    with file_writer.as_default():
        images = np.reshape(X_train[10:30], (-1, 28, 28, 1))
        tf.summary.image("20 handwritten digit samples", images, max_outputs=25, step=0)
    logging.info("\n tensorboard implementation ended \n")

    logging.info("\n early_stopping_cb implementation started \n")
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=config["params"]["patience"], restore_best_weights=config["params"]["restore_best_weights"])
    logging.info("\n early_stopping_cb implementation ended \n")

    logging.info("\n check_point_cb implementation started \n")
    ckpt_dir= os.path.join(config["artifacts"]["artifacts_dir"], config["artifacts"]["ckpt_model_dir"])
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path= os.path.join(ckpt_dir, config["artifacts"]["ckpt_model_name"])
    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True)
    logging.info("\n check_point_cb implementation ended \n")

    return [tensorboard_cb, early_stopping_cb, checkpointing_cb]




