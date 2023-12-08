from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from DataLoader import DataLoader
import numpy as np
from EvaluateRecommender import evaluate
from PrintTargetUsers import write_target_users


def kaggle_run(data_path, output_file):
    # Load the URM
    dl = DataLoader(data_path)
    targets = dl.get_target_users()
    dl_URM = dl.URM.tocsr()
    URM_train, URM_validation = dl.split_the_dataset(dl_URM, 0.8, 2)

    # Recommender
    print("Running Recommender")
    recommender = PureSVDRecommender(URM_train)
    recommender.fit(num_factors=54)

    # Evaluation
    print("Running Evaluation")
    evaluate(recommender, URM_validation)

    print("Running write to csv")
    write_target_users(recommender, output_file, targets)
