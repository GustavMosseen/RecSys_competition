from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from numpy import linalg as LA
from tqdm import tqdm
import numpy as np

class Hybrid(BaseRecommender):

    def __init__(self, URM_train): 
        super(BaseRecommender, self).__init__()
        self.URM_train = URM_train
        
    def fit(self,  
            lambda1 = 5, 
            lambda2 = 1, 
            lambda3 = 2, 
            ):      
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.pure_svd = PureSVDRecommender(self.URM_train)
        self.user_knn = UserKNNCFRecommender(self.URM_train)
        self.item_knn = P3alphaRecommender(self.URM_train)
                                                                                                
        self.pure_svd.fit()
        self.user_knn.fit()
        self.item_knn.fit()        
      
    def _compute_item_score(self, user_id_array, items_to_compute = None):

        item_weights = np.empty([10882, self.URM_train.shape[1]])
       
        for i in tqdm(range(len(user_id_array))):
            
            w1 = self.pure_svd._compute_item_score(user_id_array[i], items_to_compute) 
            w1 /= LA.norm(w1,1)

            w2 = self.user_knn._compute_item_score(user_id_array[i], items_to_compute)  
            w2 /= LA.norm(w2,1)
              
            w3 = self.item_knn._compute_item_score(user_id_array[i], items_to_compute) 
            w3 /= LA.norm(w3,1)
                             
            w = self.lambda1 * w1 + self.lambda2 * w2 + self.lambda3 * w3
                        
            item_weights[i,:] = w 
                        
        return item_weights