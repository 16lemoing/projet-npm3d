#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

import numpy as np
from sklearn import preprocessing

class NeighbourhoodClassifier:
    
    def __init__(self, classifier_type, classifier_kwargs, scale_data = False, c_D = 0.25, segment_out_ground = True, threshold_in = 1, threshold_normals = 0.8):
        """
            Build a classifier to identify true neighbouring voxels
    
            Parameters
            ----------
            classifier_type : string
                Type of classifier from ('random_forest', 'SGD' and 'SVM')
            classifier_kwargs : kwargs
                Arguments to initialize classifier
            scale_data : bool (optional)
                To tell whether or not to preprocess features
        """
        
        self.classifier_type = classifier_type
        self.scale_data = scale_data
        self.c_D = c_D
        self.segment_out_ground = segment_out_ground
        self.threshold_in = threshold_in
        self.threshold_normals = threshold_normals
        self.scaler = preprocessing.StandardScaler()
        
        if self.classifier_type == 'random_forest':
            self.clf = RandomForestClassifier(**classifier_kwargs)
        elif self.classifier_type == 'SGD':
            self.clf = SGDClassifier(**classifier_kwargs)
        elif self.classifier_type == 'SVM':
            self.clf = SVC(**classifier_kwargs)
        else:
            raise NotImplementedError()
        
    def fit(self, train_vc):
        """
            Train a classifier based from one or multiple VoxelCloud objects
    
            Parameters
            ----------
            train_vc : VoxelCloud object or list of VoxelCloud objects
        """
        
        if not isinstance(train_vc, list):
            train_cc = [train_vc]
        
        # Retrieve training features and labels
        features = []
        labels = []
        for vc in train_vc:
            feat, lbl = vc.get_train_neighbourhood_feature_and_label(self.c_D, self.segment_out_ground, self.threshold_in, self.threshold_normals)
            features.append(feat)
            labels.append(lbl)
        features = np.vstack(features)
        labels = np.hstack(labels)
        
        # Scale features if needed
        if self.scale_data:
            features = self.scaler.fit_transform(features)
        
        # Train classifier
        self.clf.fit(features, labels)
        
        # Get train error
        predicted_labels = self.predict(features)
        correct = np.sum(labels == predicted_labels) / len(predicted_labels)
        print(f"{correct * 100:.2f}% correctly classified pairs [{len(predicted_labels)} samples]")
    
    def predict(self, x):
        if len(x) > 0:
            if self.scale_data:
                x = self.scaler.transform(x)
            return self.clf.predict(x)
        return np.array([])

class ComponentClassifier:
    
    def __init__(self, classifier_type, classifier_kwargs, scale_data = False):
        """
            Build a classifier for component classification
    
            Parameters
            ----------
            classifier_type : string
                Type of classifier from ('random_forest', 'SGD' and 'SVM')
            classifier_kwargs : kwargs
                Arguments to initialize classifier
            scale_data : bool (optional)
                To tell whether or not to preprocess features
        """

        self.classifier_type = classifier_type
        self.scale_data = scale_data
        self.scaler = preprocessing.StandardScaler()
        
        if self.classifier_type == 'random_forest':
            self.clf = RandomForestClassifier(**classifier_kwargs)
        elif self.classifier_type == 'SGD':
            self.clf = SGDClassifier(**classifier_kwargs)
        elif self.classifier_type == 'SVM':
            self.clf = SVC(**classifier_kwargs)
        else:
            raise NotImplementedError()
        
    def fit(self, train_cc):
        """
            Train a classifier based from one or multiple ComponentCloud objects
    
            Parameters
            ----------
            train_cc : ComponentCloud object or list of ComponentCloud objects
        """
        
        if not isinstance(train_cc, list):
            train_cc = [train_cc]
        
        # Retrieve training features and labels
        features = []
        labels = []
        for cc in train_cc:
            features.append(cc.get_features())
            labels.append(cc.get_labels())
        features = np.vstack(features)
        labels = np.hstack(labels)
        
        # Scale features if needed
        if self.scale_data:
            features = self.scaler.fit_transform(features)
        
        # Train classifier
        self.clf.fit(features, labels)
    
    def predict(self, cc):
        """
            Predict component classes for one ComponentCloud object
    
            Parameters
            ----------
            cc : ComponentCloud object
        """
        
        features = cc.get_features()
        if self.scale_data:
            features = self.scaler.transform(features)
        
        return self.clf.predict(features)