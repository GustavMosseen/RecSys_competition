import pandas as pd
import numpy as np
import scipy.sparse as sps

class DataLoader:

    USER = "UserID"
    ITEM = "ItemID"
    DATA = "data"

    def __init__(self, data_train_file_path, data_test_file_path):

        self.data_train_file_path = data_train_file_path
        self.data_test_file_path = data_test_file_path
        self.data_train = None
        self.data_test = None
        self.URM = None
        self.URM_train = None
        self.URM_validation = None

        self.load_the_datasets()

    def load_the_datasets(self):

        # Load training data
        self.data_train = pd.read_csv(self.data_train_file_path)

        # Load test data
        self.data_test = pd.read_csv(self.data_test_file_path)

        (self.data_train).columns = [self.USER, self.ITEM, self.DATA]
        (self.data_test).columns = [self.USER]

        self.URM = sps.coo_matrix((self.data_train[self.DATA].values, 
                                (self.data_train[self.USER].values, self.data_train[self.ITEM].values)))

    def split_the_dataset(self, train_test_split):

        n_interactions = (self.URM_all).nnz

        train_mask = np.random.choice([True,False], n_interactions, p=[train_test_split, 1-train_test_split])

        self.URM_train = sps.csr_matrix((self.URM.data[train_mask],
                                    (self.URM.row[train_mask], self.URM.col[train_mask])))

        validation_mask = np.logical_not(train_mask)

        self.URM_validation = sps.csr_matrix((self.URM.data[validation_mask],
                                    (self.URM.row[validation_mask], self.URM.col[validation_mask])))

    def get_target_users(self):
        
        return self.data_test[self.USER].tolist()
