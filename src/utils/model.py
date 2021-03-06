import tensorflow as tf
import logging
import io
import time
import os
import matplotlib.pyplot as plt

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):
    logging.info(f"\n Model creation started \n")
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
          tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
          tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="outputLayer")]
    model_clf = tf.keras.models.Sequential(LAYERS)

    with io.StringIO() as stream:
          model_clf.summary(print_fn= lambda x: stream.write(f"{x}\n"))
          logging.info(stream.getvalue())
          
    model_clf.compile(loss=LOSS_FUNCTION,
                optimizer=OPTIMIZER,
                metrics=METRICS)
    logging.info(f"\n Model creation ended \n")
    return model_clf ## this will returned untrained model

def get_unique_filename(filename):
    unique_file_name= time.strftime(f"%Y-%m-%d_%H-%M-%S_{filename}")
    return unique_file_name

def save_model(model, model_name, model_dir):
    unique_file_name= get_unique_filename(model_name)
    path_to_model= os.path.join(model_dir, unique_file_name)
    model.save(path_to_model)

def save_plot(df, plots_name, plots_dir):
    df.plot(figsize=(10, 7))
    plt.grid(True)
    unique_file_name = get_unique_filename(plots_name)
    path_to_model = os.path.join(plots_dir, unique_file_name)
    plt.savefig(path_to_model)

def get_tensorboard_log_path(tensorboard_name,tensorboard_dir):
  unique_name=get_unique_filename(tensorboard_name)
  tensorboard_log_path=os.path.join(tensorboard_dir,unique_name)
  print(f"saving logs at : {tensorboard_log_path}")
  return tensorboard_log_path

def get_general_log_path(general_logs_name, general_log_path_dir):
    unique_name = get_unique_filename(general_logs_name)
    general_log_path = os.path.join(general_log_path_dir, unique_name)
    print(f"saving logs at : {general_log_path}")
    return general_log_path

def get_timestamp(name):
    timestamp = time.asctime().replace(" ","_").replace(":","_") 
    unique_name = f"{name}_at_{timestamp}"
    return unique_name

