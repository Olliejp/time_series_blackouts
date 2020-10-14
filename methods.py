import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def normalise_zero_mean(train_df, val_df, test_df):

  """
  Function to return datasets normalised by subtracting feature mean and divide by
  standard deviation
  """

  train_mean = train_df.mean(axis=1, keepdims=True).mean(axis=0, keepdims=True)
  train_std = train_df.std(axis=1, keepdims=True).mean(axis=0, keepdims=True)
  train_df = (train_df - train_mean) / train_std
  val_df = (val_df - train_mean) / train_std
  test_df = (test_df - train_mean) / train_std

  return train_df, val_df, test_df

def normalise_min_max(train_df, val_df, test_df):

  """
  Function to scale dataset features within a range of [-1,1]
  """

  train_min = train_df.min(axis=1, keepdims=True).min(axis=0, keepdims=True)
  train_max = train_df.max(axis=1, keepdims=True).max(axis=0, keepdims=True)
  train_df = (2*(train_df-train_min)/(train_max-train_min))-1
  val_df = (2*(val_df-train_min)/(train_max-train_min))-1
  test_df = (2*(test_df-train_min)/(train_max-train_min))-1

  return train_df, val_df, test_df

def get_feature_or_label_columns_and_indices(columns, start_index, end_index=None):

  """
  Function to return features columns as list and column indexes
  """

  return columns[start_index:end_index], {name: i for i, name in enumerate(columns[start_index:end_index])}


def get_window(dataset, start_index, look_back, look_forward,
               step, columns, end_index=None, feature_columns=None,
               label_columns=None, single_step=False):

  """
  Function to create windowed input output pairs from timeseries dataset

  :param dataset: Dataset to be windowed
  :param start_index: Index to start windows
  :param look_back: Input sequence length
  :param look_forward: Output sequence length to be forecast
  :param step: How many time steps to shift the sequence
  :param columns: all column names in dataset as a list
  :param end_index: Index to stop windows, by default last time step in series
  :param feature_columns: features to include in input window sequence
  :param label_columns: features to include in output window sequence
  :param single_step: If true, only forecast next step
  :return: Input and output windows as numpy arrays
  """

  data = []
  labels = []

  column_indices = {name: i for i, name in enumerate(columns)}

  start_index = start_index + look_back
  if end_index is None:
    end_index = len(dataset) - look_forward

  for i in range(start_index, end_index):
    indices = range(i - look_back, i, step)
    if feature_columns is None:
      data.append(dataset[indices])
    else:
      data.append(np.array([dataset[:, column_indices[name]]
                            for name in feature_columns]).T[indices])

    if label_columns is not None:
      targets = np.array([dataset[:, column_indices[name]]
                          for name in label_columns]).T

      if single_step:
        labels.append(targets[i + look_forward])
      else:
        labels.append(targets[i:i + look_forward])

    else:
      if single_step:
        labels.append(dataset[i + look_forward])
      else:
        labels.append(dataset[i:i + look_forward])

  return np.array(data), np.array(labels)


def create_time_steps(length):

  """
  Function to create time steps for plotting input window

  :param length: length of input window - look_back
  :return: list from look_back length to 0
  """
  return list(range(-length, 0))


def multi_step_plot(history, true_future, prediction,
                    feature_columns_indices, plot_col = "rotor_angle Gen 10",
                    step=1):

  """
  Function to plot input/output windows and respective predictions

  :param history: input window
  :param true_future: output window or labels
  :param prediction: output window predictions
  :param plot_col: feature to be plotted
  """

  fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, feature_columns_indices[plot_col]]),
           marker='.', label='Inputs', zorder=-20)
  plt.scatter(np.arange(num_out)/step,
              np.array(true_future[:, feature_columns_indices[plot_col]]),
              edgecolors='k', label='Labels', c='#2ca02c', s=64)
  if prediction.any():
    plt.scatter(np.arange(num_out)/step,
                np.array(prediction[:, feature_columns_indices[plot_col]]),
                marker='X', edgecolors='k', label='Predictions', c='#ff7f0e',
                s=64)
  plt.grid(color='k', ls = '-.', lw = 0.25)
  plt.legend()
  plt.ylabel(f'{plot_col} [normalised]')
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  plt.xlabel('Time Steps')
  plt.show()

def plot_train_history(history, title='training loss'):

  """
  Function to plot model training and validation loss

  :param history: Model training history
  :param title: Plot title
  """

  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(loss))
  plt.figure()
  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.grid(color='k', ls = '-.', lw = 0.25)
  plt.title(title)
  plt.legend()
  plt.show()


def compile_and_fit(model, train, val, steps_per_epoch, validation_steps, max_epochs, 
                    patience=2, learning_rate=0.001, checkpoint_filepath=None):

  """
  Function to compile and fit models

  :param model: model built in keras/tensorflow
  :param patience: number of epochs to wait before stopping training
  :param learning_rate: learning rate
  :param checkpoint_filepath: file path to save model weights between epochs
  :return: model history
  """

  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min',
                                                    restore_best_weights=True)

  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                              filepath=checkpoint_filepath,
                              save_weights_only=False,
                              monitor='val_loss',
                              mode='min',
                              save_best_only=True)

  model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
                optimizer=tf.optimizers.Adam(lr=learning_rate, clipnorm=1.0),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(train, epochs=max_epochs,
                      validation_data=val,
                      steps_per_epoch=steps_per_epoch,
                      validation_steps=validation_steps,
                      callbacks=[early_stopping] if checkpoint_filepath 
                        is None else [early_stopping, model_checkpoint_callback])
  return history


def learning_rate_scheduler(model, dataset, steps_per_epoch, epochs=100):

  """
  Function to run learning rate scheduler to find optimum learning rate over specified number of epochs

  :param model: model to train
  :param dataset: dataset to train on
  :param epochs: number of epochs for learning rate scheduler
  :return:
  """

  lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10 ** (epoch / 20))
  optimizer = tf.keras.optimizers.Adam(lr=1e-8)
  model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
                optimizer=optimizer,
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(dataset,
                      epochs=epochs,
                      steps_per_epoch=steps_per_epoch // 5,
                      callbacks=[lr_schedule])

  return history