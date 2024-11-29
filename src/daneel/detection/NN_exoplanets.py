# daneel/detection/NN_exoplanets.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from daneel.parameters import Parameters
from daneel.detection import *  

class NNExoplanetDetector:
    def __init__(self, params):
        # Extract parameters from params
        self.params = params
        self.path_to_train_dataset = str(params.get('path_to_train_dataset', '!!train path not given!!'))
        self.path_to_dev_dataset = str(params.get('path_to_dev_dataset', '!!dev path not given!!'))
        self.complex_model = params.get('complex_model', False)
        self.load_model = params.get('load_model', False)
        self.render_plot = params.get('render_plot', True)
        self.weights_path = params.get('weights_path', 'model_weights.h5')
        self.save_weights = params.get('save_weights', False)

        # Initialize placeholders 
        self.X_train = None
        self.Y_train = None
        self.X_dev = None
        self.Y_dev = None
        self.model = None
        self.history = None
        self.metrics = None

    def np_X_Y_from_df(self, df):
        df = shuffle(df)
        df_X = df.drop(['LABEL'], axis=1)
        X = np.array(df_X)
        Y_raw = np.array(df['LABEL']).reshape((len(df['LABEL']), 1))
        Y = (Y_raw == 2).astype(int)
        return X, Y.ravel()

    def build_network(self, input_shape):
        if self.complex_model:
            model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Input(input_shape),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(5, activation="relu"),
                    tf.keras.layers.Dense(2, activation="relu"),
                    #tf.keras.layers.Dense(2, activation="relu"),
                    #tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                ]
            )
            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
        else:
            model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Input(input_shape),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(1, activation="relu"),
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                ]
            )
            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])

        return model

    def load_data(self):
        # Check if the datasets exist
        if not os.path.isfile(self.path_to_train_dataset):
            raise FileNotFoundError(f"Training dataset not found at '{self.path_to_train_dataset}'")
        if not os.path.isfile(self.path_to_dev_dataset):
            raise FileNotFoundError(f"Development dataset not found at '{self.path_to_dev_dataset}'")

        # Loading datasets
        print("Loading datasets...")
        df_train = pd.read_csv(self.path_to_train_dataset, encoding="ISO-8859-1")
        df_dev = pd.read_csv(self.path_to_dev_dataset, encoding="ISO-8859-1")

        # Generate X and Y dataframe sets
        df_train_x = df_train.drop('LABEL', axis=1)
        df_dev_x = df_dev.drop('LABEL', axis=1)
        df_train_y = df_train['LABEL']
        df_dev_y = df_dev['LABEL']

        print("Loaded datasets!")

        # Preprocessing
        LFP = LightFluxProcessor(fourier=True, normalize=True, gaussian=True, standardize=True)
        df_train_x, df_dev_x = LFP.process(df_train_x, df_dev_x)

        df_train_processed = pd.DataFrame(df_train_x).join(pd.DataFrame(df_train_y))
        df_dev_processed = pd.DataFrame(df_dev_x).join(pd.DataFrame(df_dev_y))

        # Reshaping data
        self.X_train, self.Y_train = self.np_X_Y_from_df(df_train_processed)
        self.X_dev, self.Y_dev = self.np_X_Y_from_df(df_dev_processed)

        print("X_train.shape: ", self.X_train.shape)
        print("Y_train.shape: ", self.Y_train.shape)
        print("X_dev.shape: ", self.X_dev.shape)
        print("Y_dev.shape: ", self.Y_dev.shape)
        print("Number of features: ", self.X_train.shape[1])
        print("Number of exoplanets in training set: ", np.sum(self.Y_train))
        print("Number of exoplanets in development set: ", np.sum(self.Y_dev))

    def prepare_model(self):
        # Build model
        self.model = self.build_network(self.X_train.shape[1:])

        # Load weights if required
        if self.load_model:
            if os.path.isfile(self.weights_path):
                self.model.load_weights(self.weights_path)
                print("------------")
                print(f"Loaded saved weights from '{self.weights_path}'")
                print("------------")
            else:
                print(f"Weights file '{self.weights_path}' not found. Proceeding without loading weights.")

    def train(self):
        # Oversampling using SMOTE
        sm = SMOTE()
        X_train_sm, Y_train_sm = sm.fit_resample(self.X_train, self.Y_train.ravel())

        # Train the model
        print("Training...")
        self.history = self.model.fit(X_train_sm, Y_train_sm, epochs=50, batch_size=32)
        print("Training completed.")

        # Save the model weights to the same path
        if self.save_weights:
            # Ensure directory exists
            weights_dir = os.path.dirname(self.weights_path)
            if weights_dir and not os.path.exists(weights_dir):
                os.makedirs(weights_dir)
            self.model.save_weights(self.weights_path)
            print(f"Model weights saved to '{self.weights_path}'")

    def evaluate(self):
        # Evaluate the model
        train_outputs = self.model.predict(self.X_train, batch_size=32)
        dev_outputs = self.model.predict(self.X_dev, batch_size=32)
        train_outputs = np.rint(train_outputs)
        dev_outputs = np.rint(dev_outputs)
        accuracy_train = accuracy_score(self.Y_train, train_outputs)
        accuracy_dev = accuracy_score(self.Y_dev, dev_outputs)
        precision_train = precision_score(self.Y_train, train_outputs, zero_division=0)
        precision_dev = precision_score(self.Y_dev, dev_outputs, zero_division=0)
        recall_train = recall_score(self.Y_train, train_outputs, zero_division=0)
        recall_dev = recall_score(self.Y_dev, dev_outputs, zero_division=0)
        confusion_matrix_train = confusion_matrix(self.Y_train, train_outputs)
        confusion_matrix_dev = confusion_matrix(self.Y_dev, dev_outputs)

        # Print results
        print("Train Set Error", 1.0 - accuracy_train)
        print("Dev Set Error", 1.0 - accuracy_dev)
        print("------------")
        print("Precision - Train Set", precision_train)
        print("Precision - Dev Set", precision_dev)
        print("------------")
        print("Recall - Train Set", recall_train)
        print("Recall - Dev Set", recall_dev)
        print("------------")
        print("Confusion Matrix - Train Set")
        print(confusion_matrix_train)
        print("Confusion Matrix - Dev Set")
        print(confusion_matrix_dev)

        self.metrics = {
            'accuracy_train': accuracy_train,
            'accuracy_dev': accuracy_dev,
            'precision_train': precision_train,
            'precision_dev': precision_dev,
            'recall_train': recall_train,
            'recall_dev': recall_dev,
            'confusion_matrix_train': confusion_matrix_train,
            'confusion_matrix_dev': confusion_matrix_dev
        }

    def plot_results(self):
        # Plotting if required
        if self.render_plot and self.history is not None:
            # Plot accuracy and loss
            plt.figure(figsize=(12, 5))

            # Plot accuracy
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history["accuracy"], label="Training Accuracy")
            plt.title("Model Accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.legend(loc="upper left")

            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history["loss"], label="Training Loss")
            plt.title("Model Loss")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.legend(loc="upper right")

            plt.tight_layout()
            plt.show()

    def run(self):
        # Execute the steps
        self.load_data()
        self.prepare_model()
        self.train()
        self.evaluate()
        self.plot_results()


if __name__ == "__main__":
    path = '../../../Assignment2/taskI.yaml'
    param= Parameters(path).params
    nn_detector = NNExoplanetDetector(param)
    nn_detector.run()