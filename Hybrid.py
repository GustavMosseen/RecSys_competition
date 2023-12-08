from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from tqdm import tqdm
import numpy as np


class Hybrid(BaseRecommender):

    def __init__(self, URM_train): 
        super(BaseRecommender, self).__init__()
        self.URM_train = URM_train
        self.lambda1 = 1
        self.lambda2 = 1
        self.lambda3 = 1
        
    def fit(self,
            lambda1=6,
            lambda2=7,
            lambda3=1,
            ):      
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.pure_svd = PureSVDRecommender(self.URM_train)
        self.user_knn = UserKNNCFRecommender(self.URM_train)
        self.item_knn = P3alphaRecommender(self.URM_train)

        print('PureSVD running')
        self.pure_svd.fit(num_factors=54)
        print('UserKNN running')
        self.user_knn.fit(shrink=1,
                          topK=384,
                          similarity="cosine"
                          )
        print('P3alpha running')
        self.item_knn.fit(topK=48,
                          alpha=0.29
                          )
      
    def _compute_item_score(self, user_id_array, items_to_compute=None):

        w1_all = np.vstack(
            [self.pure_svd._compute_item_score(user_id, items_to_compute) for user_id in user_id_array])
        w1_all /= np.linalg.norm(w1_all, 1, axis=1)[:, np.newaxis]

        w2_all = np.vstack(
            [self.user_knn._compute_item_score(user_id, items_to_compute) for user_id in user_id_array])
        w2_all /= np.linalg.norm(w2_all, 1, axis=1)[:, np.newaxis]

        w3_all = np.vstack(
            [self.item_knn._compute_item_score(user_id, items_to_compute) for user_id in user_id_array])
        w3_all /= np.linalg.norm(w3_all, 1, axis=1)[:, np.newaxis]

        item_weights = self.lambda1 * w1_all + self.lambda2 * w2_all + self.lambda3 * w3_all

        return item_weights

    def get_weights(self, user_id_array, items_to_compute = None):
        w1_all = np.vstack(
            [self.pure_svd._compute_item_score(user_id, items_to_compute) for user_id in user_id_array])
        w1_all /= np.linalg.norm(w1_all, 1, axis=1)[:, np.newaxis]

        w2_all = np.vstack(
            [self.user_knn._compute_item_score(user_id, items_to_compute) for user_id in user_id_array])
        w2_all /= np.linalg.norm(w2_all, 1, axis=1)[:, np.newaxis]

        w3_all = np.vstack(
            [self.item_knn._compute_item_score(user_id, items_to_compute) for user_id in user_id_array])
        w3_all /= np.linalg.norm(w3_all, 1, axis=1)[:, np.newaxis]

        return [w1_all, w2_all, w3_all]

    def update_lambda(self,
            lambda1=6,
            lambda2=7,
            lambda3=1,
            ):

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
