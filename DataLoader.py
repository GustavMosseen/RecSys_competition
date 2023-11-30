import pandas as pd
import numpy as np
import scipy.sparse as sps
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

class DataLoader:

    USER = "UserID"
    ITEM = "ItemID"
    DATA = "data"

    def __init__(self, path):

        self.data_train_file_path = path + "/data_train.csv"
        self.data_test_file_path = path + "/data_target_users_test.csv"
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

    @staticmethod
    def split_the_dataset(URM, train_test_split, sets):
        if sets not in [2, 3]:
            raise ValueError('Input needs to be 2 or 3')
        URMs = split_train_in_two_percentage_global_sample(URM, train_percentage=train_test_split)
        if sets == 3:
            URMs3 = split_train_in_two_percentage_global_sample(URMs[0], train_percentage=train_test_split)
            return URMs3[0], URMs3[1], URMs[1]
        return URMs

    def get_target_users(self):
        
        return self.data_test[self.USER].tolist()
