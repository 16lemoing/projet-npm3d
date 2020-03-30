#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier

class Classifier:
    
    def __init__(self, classifier_type, classifier_kwargs):
        self.classifier_type = classifier_type
        
        if self.classifier_type == 'random_forest':
            self.clf = RandomForestClassifier(**classifier_kwargs)
        
    def fit(self, component_cloud):
        self.clf.fit(component_cloud.get_features(), component_cloud.get_labels())
    
    def predict(self, component_cloud):
        return self.clf.predict(component_cloud.get_features())
    


## Test random forest

n_estimators = 20 # number of decision trees
classifier = Classifier('random_forest', {'n_estimators': n_estimators})