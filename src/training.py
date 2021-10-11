from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, save_model, save_plot
import argparse
import logging
import os
import pandas as pd


def training(config_path):
    config = read_config(config_path)

    logging_str= "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
    log_dir=config["logs"]["logs_dir"]
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename= os.path.join(log_dir, "general_logs.log"), level=logging.INFO, format=logging_str, filemode="w")

    validation_datasize=config["params"]["validation_datasize"]
    (X_train, y_train),(X_valid, y_valid), (X_test, y_test)=get_data(validation_datasize)

    LOSS_FUNCTION=config["params"]["loss_function"]
    OPTIMIZER=config["params"]["optimizer"]
    METRICS=config["params"]["metrics"]
    NUM_CLASSES=config["params"]["num_classes"]
    
    model=create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)
    

    EPOCHS = config["params"]["epochs"]
    VALIDATION_SET = (X_valid, y_valid)
    logging.info(f"\n Model fitting started \n")
    history = model.fit(X_train, y_train, epochs=EPOCHS,
                   validation_data=VALIDATION_SET)
    logging.info(f"\n {pd.DataFrame(history.history)} \n")

    artifacts_dir=config["artifacts"]["artifacts_dir"]
    model_name=config["artifacts"]["model_name"]
    model_dir=config["artifacts"]["model_dir"]
    model_dir_path=os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    save_model(model, model_name, model_dir_path)

    logging.info(f"\n model fitting ended \n")

    df=pd.DataFrame(history.history)
    plots_name= config["artifacts"]["plots_name"]
    plots_dir=config["artifacts"]["plots_dir"]
    plot_dir_path = os.path.join(artifacts_dir, plots_dir)
    os.makedirs(plot_dir_path, exist_ok=True)
    save_plot(df, plots_name, plot_dir_path)



if __name__ == '__main__':
    args=argparse.ArgumentParser()
    logging.info(f"\n {args} \n")
    args.add_argument("--config","-c",default="config.yaml")
    parsed_args=args.parse_args()
    logging.info(f"\n {parsed_args} \n")
    training(config_path=parsed_args.config)

