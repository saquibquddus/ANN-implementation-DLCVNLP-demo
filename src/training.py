from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, save_model, save_plot, get_tensorboard_log_path, get_general_log_path, get_unique_filename
from src.utils.callbacks import get_callbacks
import argparse
import logging
import os
import pandas as pd
import tensorflow as tf
import  numpy as np

def training(config_path):
    #reading config file
    config = read_config(config_path)

    #creating general log files for logging info
    logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
    log_dir=config["logs"]["logs_dir"]
    general_logs = config["logs"]["general_logs"]
    general_log_path_dir=os.path.join(log_dir,general_logs)
    os.makedirs(general_log_path_dir, exist_ok=True)
    general_logs_name = config["logs"]["general_logs_name"]
    general_log_path = get_general_log_path(general_logs_name, general_log_path_dir)
    logging.basicConfig(filename=general_log_path, level=logging.INFO, format=logging_str,
                        filemode="a")
    logging.info("\nlogging directory created\n")

    #Splitting Data and model creation
    validation_datasize=config["params"]["validation_datasize"]
    (X_train, y_train),(X_valid, y_valid), (X_test, y_test)=get_data(validation_datasize)

    LOSS_FUNCTION=config["params"]["loss_function"]
    OPTIMIZER=config["params"]["optimizer"]
    METRICS=config["params"]["metrics"] 
    NUM_CLASSES=config["params"]["num_classes"]
    
    model=create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    logging.info(f"\n Model fitting started \n")
    EPOCHS = config["params"]["epochs"]
    VALIDATION_SET = (X_valid, y_valid)

    #comments added to main
    CALLBACKS_LIST_NEW= get_callbacks(config, X_train)

    history = model.fit(X_train, y_train, epochs=EPOCHS,
                   validation_data=VALIDATION_SET,
                        callbacks=CALLBACKS_LIST_NEW)
    logging.info(f"\n {pd.DataFrame(history.history)} \n")

    artifacts_dir=config["artifacts"]["artifacts_dir"]
    model_name=config["artifacts"]["model_name"]
    model_dir=config["artifacts"]["model_dir"]
    model_dir_path=os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    save_model(model, model_name, model_dir_path)
    logging.info(f"\n model fitting ended \n")

    #creating and saving the plots
    df=pd.DataFrame(history.history)
    plots_name= config["artifacts"]["plots_name"]
    plots_dir=config["artifacts"]["plots_dir"]
    plot_dir_path = os.path.join(artifacts_dir, plots_dir)
    os.makedirs(plot_dir_path, exist_ok=True)
    save_plot(df, plots_name, plot_dir_path)

    #test_model = tf.keras.models.load_model(ckpt_path)
    #history = test_model.fit(X_train, y_train, epochs=EPOCHS,
    #                         validation_data=VALIDATION_SET, callbacks=CALLBACKS_LIST)


if __name__ == '__main__':
    args=argparse.ArgumentParser()
    #logging.info(f"\n {args} \n")
    args.add_argument("--config","-c",default="config.yaml")
    parsed_args=args.parse_args()
    #logging.info(f"\n {parsed_args} \n")
    training(config_path=parsed_args.config)

