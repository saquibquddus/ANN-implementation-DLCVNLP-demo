import tensorflow as tf
import logging
import io
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