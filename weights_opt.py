from Hybrid import Hybrid
from tqdm import tqdm
from DataLoader import DataLoader
from PrintTargetUsers import write_target_users
from Evaluation.Evaluator import EvaluatorHoldout
# Optuna
import optuna
import pandas as pd
import os
import numpy as np

# Recommenders
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender

# data_path = "/Users/anadrmic/Desktop/POLIMI/New Folder With Items/2/RS/competition/recommender-system-2023-challenge-polimi"
data_path = "/Users/gustavmosseen/PycharmProjects/pythonProject/RecommenderSystems"
output_file = "hybrid_test_after_hp2.csv"

dl = DataLoader(data_path)
targets = dl.get_target_users()
dl_URM = dl.URM.tocsr()

# Split data
URM_train, URM_validation = dl.split_the_dataset(dl_URM, 0.7, 2)
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
user_id_array = dl.get_user_id_array(URM_train)


class SaveResults(object):
    def __init__(self, results_folder="optuna_results"):
        self.results_folder = results_folder
        self.results_df = pd.DataFrame()

        # Create the results folder if it doesn't exist
        os.makedirs(self.results_folder, exist_ok=True)

    def __call__(self, optuna_study, optuna_trial):
        hyperparam_dict = optuna_trial.params.copy()
        # Retrieve the optimal number of epochs from the "user attributes" of the trial
        if "epochs" in optuna_trial.user_attrs:
            hyperparam_dict["epochs"] = optuna_trial.user_attrs["epochs"]
        hyperparam_dict["result"] = optuna_trial.values[0]

        # Concatenate hyperparameters and result into a DataFrame
        result_df = pd.DataFrame([hyperparam_dict])

        # Save the result to a text file based on the recommender name
        recommender_name = optuna_trial.user_attrs["recommender_name"]

        result_file_path = f"{self.results_folder}/evaluation_results_{recommender_name}.txt"

        # Append the result to the main results DataFrame
        self.results_df = pd.concat([self.results_df, result_df], ignore_index=True)

        # Save the main results DataFrame to a CSV file
        self.results_df.to_csv(result_file_path, sep='\t', index=True)


def objective_function_Hybrid(optuna_trial):
    hybrid_instance = hybrid_ins

    recommender_name = "Hybrid_lamda_opt"
    optuna_trial.set_user_attr("recommender_name", recommender_name)

    # hybrid_instance.update_lambda(
    #     lambda1=optuna_trial.suggest_float('lambda1', 0.0, 10.0),
    #     lambda2=optuna_trial.suggest_float('lambda2', 0.0, 10.0),
    #     lambda3=optuna_trial.suggest_float('lambda3', 0.0, 10.0)
    # )

    hybrid_instance.update_lambda(
        lambda1=optuna_trial.suggest_int('lambda1', 0, 10),
        lambda2=optuna_trial.suggest_int('lambda2', 0, 10),
        lambda3=optuna_trial.suggest_int('lambda3', 0, 10)
    )

    print('Evaluating val')
    result_df, _ = evaluator_validation.evaluateRecommender(hybrid_instance)  # Implement your evaluation method
    print('Finish val')
    return result_df.loc[10]["MAP"]

hybrid_ins = Hybrid(URM_train)
hybrid_ins.fit()

optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_Hybrid,
                      callbacks=[save_results],
                      n_trials=30)

# optuna_study.best_trial.params
#
# optuna_study.best_trial
#
# save_results.results_df
