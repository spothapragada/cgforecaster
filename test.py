# A class to detect anomalies in a dataset using isolation forest, Mahalanobis distance and standard deviation
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import covariance

import pandas as pd
import numpy as np

from pyspark.ml.feature import StandardScaler as spark_StandardScaler


import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filemode='w'
                    )


class AnomalyDetector:
    def __init__(self, dataset, features, method='mahalanobis', contamination=0.1, random_state=42):
        self.dataset = dataset
        self.features = features
        self.method = method
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.scaler.fit(self.dataset[self.features])
        self.dataset[self.features] = self.scaler.transform(self.dataset[self.features])
        self.outlier_detector = None
        self.outlier_detector_name = None
        self.outlier_detector_score = None
        self.outlier_detector_predictions = None
        self.outlier_detector_predictions_proba = None
        self.outlier_detector_predictions_proba_threshold = None
        self.outlier_detector_predictions_threshold = None

    def fit(self):
        if self.method == 'isolation_forest':
            self.outlier_detector = IsolationForest(contamination=self.contamination, random_state=42)
            self.outlier_detector_name = 'Isolation Forest'
        elif self.method == 'local_outlier_factor':
            self.outlier_detector = LocalOutlierFactor(n_neighbors=20, contamination=self.contamination)
            self.outlier_detector_name = 'Local Outlier Factor'
        elif self.method == 'mahalo':
            self.outlier_detector = EllipticEnvelope(contamination=self.contamination, store_precision=True,
                                                     assume_centered=False, support_fraction=None,
                                                     random_state=self.random_state)
            self.outlier_detector_name = 'Mahalonobis Elliptic Envelope'

        else:
            raise ValueError('Method not supported')
        self.outlier_detector.fit(self.dataset[self.features])
        self.outlier_detector_score = self.outlier_detector.score_samples(self.dataset[self.features])
        self.outlier_detector_predictions = self.outlier_detector.predict(self.dataset[self.features])
        self.outlier_detector_predictions_proba = self.outlier_detector.predict_proba(self.dataset[self.features])
        self.outlier_detector_predictions_proba_threshold = self.outlier_detector.predict_proba(
            self.dataset[self.features])[:, 1]
        self.outlier_detector_predictions_threshold = self.outlier_detector.predict(self.dataset[self.features])
