from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from DataLoader import DataLoader
import numpy as np
from EvaluateRecommender import evaluate
from PrintTargetUsers import write_target_users


# Paths
# data_path = "/Users/anadrmic/Desktop/POLIMI/New Folder With Items/2/RS/competition/recommender-system-2023-challenge-polimi/"
data_path = "/Users/gustavmosseen/PycharmProjects/pythonProject/RecommenderSystems"
output_file = "svd_user_alpha_.csv"

# Load the URM
dl = DataLoader(data_path)
targets = dl.get_target_users()
dl_URM = dl.URM.tocsr()
URM_train, URM_validation = dl.split_the_dataset(dl_URM, 0.8, 2)

# Recommender
recommender = PureSVDRecommender(URM_train)
recommender.fit(num_factors=54)

# Evaluation
evaluate(recommender, URM_validation)

# Print the target users
yes_no = input('\nDo you want to save result ("yes", "no"):\t ')
if yes_no == 'yes':
    print("\nSaving results to: " + output_file)
    write_target_users(recommender, output_file, targets)
