import pandas as pd
import numpy as np
import scipy.sparse as sps


class PreProcess(object):
    def __init__(self, path: str):
        self.target_users_path_name = path + "/data_target_users_test.csv"
        self.train_path_name = path + "/data_train.csv"
        self.URM_all_dataframe = pd.read_csv(filepath_or_buffer=self.train_path_name)
        self.target_users = pd.read_csv(self.target_users_path_name)
        self.URM_all_dataframe.columns = ["UserID", "ItemID", "data"]
        self.URM_all = sps.coo_matrix((self.URM_all_dataframe["data"].values,
                                       (self.URM_all_dataframe["UserID"].values,
                                        self.URM_all_dataframe["ItemID"].values)))
        self.URM_all = self.URM_all.tocsr()

    def getURM_all(self):
        return self.URM_all.copy()

    def get_split_data(self, URM_all, train_test_split):
        train_mask = np.random.choice([True, False], len(URM_all.data),
                                      p=[train_test_split, 1 - train_test_split])
        test_mask = np.logical_not(train_mask)

        URM_train = sps.csr_matrix((URM_all.data[train_mask],
                                    (URM_all.indices[train_mask],
                                    URM_all.indptr)))

        URM_test = sps.csr_matrix((URM_all.data[test_mask],
                                   (URM_all.indices[test_mask],
                                   URM_all.indptr)))

        return URM_train, URM_test

    def __len__(self):
        userID_unique = self.URM_all_dataframe["UserID"].unique()
        itemID_unique = self.URM_all_dataframe["ItemID"].unique()
        # Number of users, items and interactions
        n_users = len(userID_unique)
        n_items = len(itemID_unique)
        n_interactions = len(self.URM_all_dataframe)
        return n_users, n_items, n_interactions

    def split_train_val(self, URM_all, r):

        if not 0 <= r <= 1:
            raise ValueError("Invalid value for the ratio 'r'. It should be between 0 and 1.")

        # Calculate the number of interactions to be used for validation
        total_interactions = URM_all.nnz
        val_interactions = int(r * total_interactions)

        # Get the indices of non-zero elements in URM_all
        non_zero_indices = list(zip(URM_all.row, URM_all.col))

        # Randomly choose indices for the validation set
        val_indices = np.random.choice(total_interactions, val_interactions, replace=False)

        # Create a boolean mask for the validation set
        val_mask = np.zeros(total_interactions, dtype=bool)
        val_mask[val_indices] = True

        # Separate non-zero indices into training and validation sets
        train_indices = np.array(non_zero_indices)[~val_mask]
        val_indices = np.array(non_zero_indices)[val_mask]

        # Create sparse matrices for training and validation sets
        URM_train = sps.csr_matrix((URM_all.data[train_indices[:, 0], train_indices[:, 1]],
                                    (train_indices[:, 0], train_indices[:, 1])),
                                   shape=URM_all.shape)

        URM_val = sps.csr_matrix((URM_all.data[val_indices[:, 0], val_indices[:, 1]],
                                  (val_indices[:, 0], val_indices[:, 1])),
                                 shape=URM_all.shape)

        return URM_train, URM_val


