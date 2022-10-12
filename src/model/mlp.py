import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
import os


def read_data(num_simulations=100):
    print("Reading data")

    input_data = np.empty((1, 12))
    target_data = np.empty((1, 12))

    for i in range(0, num_simulations):
        raw_data = pd.read_csv(os.getcwd() + "/data/processed/simulation_" + str(i) + ".csv")
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


def train_model(model, train_input, train_target, test_input, test_target, use_tensorboard, save_weights):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=4)
    callbacks_list = [early_stopping_callback]
    if use_tensorboard:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.getcwd() + "/model/tensorboard")
        callbacks_list.append(tensorboard_callback)
    if save_weights:
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.getcwd() + "/model/weights/",
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        callbacks_list.append(model_checkpoint_callback)

    model.fit(train_input, train_target,
              epochs=2,
              validation_data=(test_input, test_target),
              callbacks=callbacks_list,
              )

    return model


def k_fold_evaluate_model(input_data, target_data, k=2):
    print("Starting k-fold evaluation")
    fold_idx = 1

    for train_index, test_index in KFold(k).split(input_data, target_data):
        print("\tRunning iteration ", fold_idx, " of ", k)
        fold_idx = fold_idx + 1

        train_input = input_data[train_index]
        train_target = target_data[train_index]
        test_input = input_data[test_index]
        test_target = target_data[test_index]

        model = build_model()
        train_model(model, train_input, train_target, test_input, test_target, use_tensorboard=True, save_weights=False)

    print("Finished k-fold evaluation")


def predict_on_model(input_data, target_data, timesteps=10, num_simulations=20):
    print("Rebuilding and training model on the full dataset for prediction")

    model = build_model()
    model = train_model(model, input_data, target_data, input_data, target_data, use_tensorboard=False, save_weights=True)

    print("Starting prediction on model")

    start_states = pd.read_csv(os.getcwd() + "/data/lockbox/start_states.csv").to_numpy()
    for i in range(timesteps):
        for j in range(num_simulations):
            with open(os.getcwd() + "/model/predict/simulation_" + str(j) + ".csv", "a") as f:
                np.savetxt(f, start_states[j:j + 1, :], delimiter=",", newline='\n')
        start_states = model.predict_on_batch(start_states)

    print("Finished prediction on model")


def main():
    input_data, target_data = read_data()

    k_fold_evaluate_model(input_data, target_data)

    predict_on_model(input_data, target_data)


if __name__ == "__main__":
    main()
