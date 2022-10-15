import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold

import utils


# Clears the '/model' directory.
def clear_model_directory():
    print("Clearing '/model' directory")

    path = os.path.join(os.getcwd() + "/model")
    for d in os.listdir(path):
        d_path = os.path.join(path, d)
        utils.clear_directory(d_path)


# Reads in the input and target data.
def read_data(num_simulations=100):
    print("Reading data")

    input_data = np.empty((1, 12))
    target_data = np.empty((1, 12))

    for i in range(0, num_simulations):
        raw_data = pd.read_csv(os.path.join(os.getcwd() + "/data/processed/simulation_" + str(i) + ".csv"), header=None)
        simulation_input = np.delete(raw_data.to_numpy(), -1, axis=0)
        input_data = np.vstack((input_data, simulation_input))
        simulation_target = np.delete(raw_data.to_numpy(), 0, axis=0)
        target_data = np.vstack((target_data, simulation_target))

    input_data = np.delete(input_data, 0, axis=0)
    target_data = np.delete(target_data, 0, axis=0)

    print("Dimensions of data (timesteps, state vectors):",
          "\n\tInput ", input_data.shape,
          "\n\tTarget ", target_data.shape)

    return input_data, target_data


# Builds the model.
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(12,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(12, activation='linear'),
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss='mse',
                  metrics=[tf.keras.metrics.MeanSquaredError()]
                  )

    return model


# Trains the model with optional callbacks.
def train_model(model, train_input, train_target, test_input, test_target, use_tensorboard, save_weights):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=4)
    callbacks_list = [early_stopping_callback]
    if use_tensorboard:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(os.getcwd() + "/model/tensorboard"))
        callbacks_list.append(tensorboard_callback)
    if save_weights:
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(os.getcwd() + "/model/weights/"),
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        callbacks_list.append(model_checkpoint_callback)

    model.fit(train_input, train_target,
              epochs=12,
              validation_data=(test_input, test_target),
              callbacks=callbacks_list,
              )

    return model


# Evaluates the model based on k-fold cross validation.
def k_fold_evaluate_model(input_data, target_data, k=5):
    print("Running k-fold cross evaluation")
    fold_idx = 1

    for train_index, test_index in KFold(k).split(input_data, target_data):
        print("\tIteration ", fold_idx, " of ", k)
        fold_idx = fold_idx + 1

        train_input = input_data[train_index]
        train_target = target_data[train_index]
        test_input = input_data[test_index]
        test_target = target_data[test_index]

        model = build_model()
        train_model(model, train_input, train_target, test_input, test_target, use_tensorboard=True, save_weights=False)


# Trains the model on the full dataset and predicts on the lockboxed start states batch.
def predict_on_model(input_data, target_data, timesteps=5000, num_simulations=20):
    print("Building and training model on the full dataset to predict")
    model = build_model()
    model = train_model(model, input_data, target_data, input_data, target_data, use_tensorboard=False, save_weights=True)

    print("Predicting on model")
    start_states = pd.read_csv(os.path.join(os.getcwd() + "/data/lockbox/start_states.csv"), header=None).to_numpy()
    for i in range(timesteps):
        for j in range(num_simulations):
            with open(os.path.join(os.getcwd() + "/model/predict/simulation_" + str(j) + ".csv"), "a") as f:
                np.savetxt(f, start_states[j:j+1, :], delimiter=",", newline='\n')
        start_states = model.predict_on_batch(start_states)


def main():
    clear_model_directory()
    input_data, target_data = read_data()
    k_fold_evaluate_model(input_data, target_data)
    predict_on_model(input_data, target_data)


if __name__ == "__main__":
    main()
