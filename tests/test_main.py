import unittest
import pandas as pd
import numpy as np
from src.main import add_magic_feature
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import logging

class TestMainFunctions(unittest.TestCase):
    
    def setUp(self):
        # Creating test data for use in tests
        self.train_data = pd.DataFrame({
            'ID_code': ['id_1', 'id_2', 'id_3'],
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        
        self.test_data = pd.DataFrame({
            'ID_code': ['id_4', 'id_5'],
            'feature1': [2, 3],
            'feature2': [5, 6]
        })
    
    def test_add_magic_feature(self):
        # Checking the correctness of the add_magic_feature function
        result_train, result_test = add_magic_feature(
            self.train_data.drop(['ID_code', 'target'], axis=1),
            self.test_data.drop(['ID_code'], axis=1)
        )
        
        # Checking that the data dimensions match
        self.assertEqual(result_train.shape[0], self.train_data.shape[0])
        self.assertEqual(result_test.shape[0], self.test_data.shape[0])
        
        # Checking for the presence of new magic features
        self.assertTrue('feature1_magic' in result_train.columns)
        self.assertTrue('feature2_magic' in result_test.columns)
    
    def test_model_training(self):
        # Checking that the model trains without errors
        X = self.train_data.drop(['ID_code', 'target'], axis=1)
        y = self.train_data['target']
        
        model = LGBMClassifier()
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        
        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_val)[:, 1]

            # Checking that both classes are present in the validation set
            if len(np.unique(y_val)) == 2:
                auc = roc_auc_score(y_val, preds)
                # Checking that AUC is in the range from 0 to 1
                self.assertGreaterEqual(auc, 0)
                self.assertLessEqual(auc, 1)
            else:
                # If only one class is present, skip AUC calculation
                logging.info(f"Skipping AUC calculation for this fold due to only one class in y_val: {np.unique(y_val)}")
    
    def tearDown(self):
        # Cleanup after tests, if needed
        pass

if __name__ == '__main__':
    unittest.main()