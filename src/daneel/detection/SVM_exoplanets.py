# daneel/detection/SVM_exoplanets.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.svm import SVC, LinearSVC
from daneel.parameters import Parameters
from daneel.detection import *  

class SVMExoplanetDetector:
    def __init__(self, params):
        # Extract parameters from params
        self.params = params
        self.path_to_train_dataset = str(params.get('path_to_train_dataset', '!!train path not given!!'))
        self.path_to_dev_dataset = str(params.get('path_to_dev_dataset', '!!dev path not given!!'))
        self.kernel = str(params.get('kernel', 'linear_svc'))
        self.degree = params.get('degree', 3)
        
        # Initialize placeholders 
        self.X_train = None
        self.Y_train = None
        self.X_dev = None
        self.Y_dev = None
        self.model = None
        self.metrics = None

    def np_X_Y_from_df(self, df):
        df = shuffle(df)
        df_X = df.drop(['LABEL'], axis=1)
        X = np.array(df_X)
        Y_raw = np.array(df['LABEL']).reshape((len(df['LABEL']), 1))
        Y = Y_raw == 2
        return X, Y.ravel()

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

    def build_model(self):
        # Create the model based on the specified kernel
        if self.kernel == 'linear_svc':
            self.model = LinearSVC(class_weight='balanced')
        elif self.kernel in ['linear', 'rbf', 'poly']:
            if self.kernel == 'poly':
                self.model = SVC(kernel=self.kernel, degree=self.degree)
            else:
                self.model = SVC(kernel=self.kernel)
        else:
            raise ValueError("Unsupported kernel. Use 'linear', 'rbf', 'poly', or 'linear_svc'.")

        print(f"Model initialized with kernel='{self.kernel}'")
        if self.kernel == 'poly':
            print(f"Polynomial degree: {self.degree}")

    def train(self):
        print("Training...")
        # Train the model on the training data
        self.model.fit(self.X_train, self.Y_train.ravel())
        print("Finished Training!")

    def evaluate(self):
        # Evaluate the model
        train_outputs = self.model.predict(self.X_train)
        dev_outputs = self.model.predict(self.X_dev)
        accuracy_train = accuracy_score(self.Y_train, train_outputs)
        accuracy_dev = accuracy_score(self.Y_dev, dev_outputs)
        precision_train = precision_score(self.Y_train, train_outputs)
        precision_dev = precision_score(self.Y_dev, dev_outputs)
        recall_train = recall_score(self.Y_train, train_outputs)
        recall_dev = recall_score(self.Y_dev, dev_outputs)
        confusion_matrix_train = confusion_matrix(self.Y_train, train_outputs)
        confusion_matrix_dev = confusion_matrix(self.Y_dev, dev_outputs)
        classification_report_train = classification_report(self.Y_train, train_outputs)
        classification_report_dev = classification_report(self.Y_dev, dev_outputs)

        # Print results
        print(" ")
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
        print("------------")
        print("Classification Report - Train Set")
        print(classification_report_train)
        print("Classification Report - Dev Set")
        print(classification_report_dev)

        self.metrics = {
            'accuracy_train': accuracy_train,
            'accuracy_dev': accuracy_dev,
            'precision_train': precision_train,
            'precision_dev': precision_dev,
            'recall_train': recall_train,
            'recall_dev': recall_dev,
            'confusion_matrix_train': confusion_matrix_train,
            'confusion_matrix_dev': confusion_matrix_dev,
            'classification_report_train': classification_report_train,
            'classification_report_dev': classification_report_dev
        }



    def run(self):
        # Execute the steps
        self.load_data()
        self.build_model()
        self.train()
        self.evaluate()



if __name__ == "__main__":
    param = {'path_to_train_dataset': "../../../../data/kepler/data_no_injection/exoTrain.csv",
            'path_to_dev_dataset': "../../../../data/kepler/data_no_injection/exoTest.csv",
            'kernel': 'linear_svc',
            'degree': 4
            }
    svm_detector = SVMExoplanetDetector(param)
    svm_detector.run()
    
    
    