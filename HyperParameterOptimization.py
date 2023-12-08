# Standard
from tqdm import tqdm
import numpy as np

# Recommenders
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender

# RecSys functions
from Evaluation.Evaluator import EvaluatorHoldout
from DataLoader import DataLoader
# Optuna
import optuna
import pandas as pd
import os

# SVD++
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_SVDpp_Cython

# The hyperparameters: batch_size, epochs, item_reg, learning_rate, num_factors, result, sgd_mode, user_reg

# Preprocess
path = "/Users/gustavmosseen/PycharmProjects/pythonProject/RecommenderSystems"
output_file = "svd_user_alpha_.csv"

dl = DataLoader(path)
targets = dl.get_target_users()
URM_all = dl.URM.tocsr()
URM_train, URM_validation = dl.split_the_dataset(URM_all, 0.7, 2)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
# evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])


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


def objective_function_svdpp(optuna_trial):
    # Earlystopping hyperparameters available in the framework
    full_hyperp = {"validation_every_n": 5,
                   "stop_on_validation": True,
                   "evaluator_object": evaluator_validation,
                   "lower_validations_allowed": 5,  # Higher values will result in a more "patient" earlystopping
                   "validation_metric": "MAP",

                   # MAX number of epochs (usually 500)
                   "epochs": 100,
                   }

    recommender_instance = MatrixFactorization_SVDpp_Cython(URM_train)
    recommender_instance.fit(num_factors=optuna_trial.suggest_int("num_factors", 1, 200),
                             sgd_mode=optuna_trial.suggest_categorical("sgd_mode", ["sgd", "adagrad", "adam"]),
                             batch_size=optuna_trial.suggest_categorical("batch_size",
                                                                         [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
                             item_reg=optuna_trial.suggest_float("item_reg", 1e-5, 1e-2, log=True),
                             user_reg=optuna_trial.suggest_float("user_reg", 1e-5, 1e-2, log=True),
                             learning_rate=optuna_trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
                             **full_hyperp)

    recommender_name = "SVDpp_Cython"
    optuna_trial.set_user_attr("recommender_name", recommender_name)

    # Add the number of epochs selected by earlystopping as a "user attribute" of the optuna trial
    epochs = recommender_instance.get_early_stopping_final_epochs_dict()["epochs"]
    optuna_trial.set_user_attr("epochs", epochs)

    result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)

    return result_df.loc[10]["MAP"]


def objective_function_funksvd(optuna_trial):
    # Earlystopping hyperparameters available in the framework
    full_hyperp = {"validation_every_n": 5,
                   "stop_on_validation": True,
                   "evaluator_object": evaluator_validation,
                   "lower_validations_allowed": 5,  # Higher values will result in a more "patient" earlystopping
                   "validation_metric": "MAP",
                   #"sgd_mode": "adagrad", # best one in previous testing
                   # MAX number of epochs (usually 500)
                   "epochs": 100,
                   }

    recommender_name = "FunkSVD"
    optuna_trial.set_user_attr("recommender_name", recommender_name)

    recommender_instance = MatrixFactorization_SVDpp_Cython(URM_train)
    recommender_instance.fit(num_factors=optuna_trial.suggest_int("num_factors", 1, 200),
                             sgd_mode=optuna_trial.suggest_categorical("sgd_mode", ["sgd", "adagrad", "adam"]),
                             batch_size=optuna_trial.suggest_categorical("batch_size",
                                                                         [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
                             item_reg=optuna_trial.suggest_float("item_reg", 1e-5, 1e-2, log=True),
                             user_reg=optuna_trial.suggest_float("user_reg", 1e-5, 1e-2, log=True),
                             learning_rate=optuna_trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
                             **full_hyperp)

    # Add the number of epochs selected by earlystopping as a "user attribute" of the optuna trial
    epochs = recommender_instance.get_early_stopping_final_epochs_dict()["epochs"]
    optuna_trial.set_user_attr("epochs", epochs)

    result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)

    return result_df.loc[10]["MAP"]  # at cutoff 10


def objective_function_puresvd(optuna_trial):
    # Earlystopping hyperparameters available in the framework
    full_hyperp = {"validation_every_n": 5,
                   "stop_on_validation": True,
                   "evaluator_object": evaluator_validation,
                   "lower_validations_allowed": 10,  # Higher values will result in a more "patient" earlystopping
                   "validation_metric": "MAP",
                   }

    recommender_name = "PureSVD"
    optuna_trial.set_user_attr("recommender_name", recommender_name)

    recommender_instance = PureSVDRecommender(URM_train)
    recommender_instance.fit(num_factors=optuna_trial.suggest_int("num_factors", 25, 80))

    result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)

    return result_df.loc[10]["MAP"]


def objective_function_UserKNNCF(optuna_trial):
    recommender_name = "UserKNNCF"
    optuna_trial.set_user_attr("recommender_name", recommender_name)

    recommender_instance = UserKNNCFRecommender(URM_train)
    # {'similarity': 'cosine', 'topK': 384, 'shrink': 0}. Best is trial: 387 with value: 0.04826361646061196.
    similarity = optuna_trial.suggest_categorical("similarity",
                                                  [
                                                'cosine',
                                                   # 'dice',
                                                   'jaccard',
                                                   'asymmetric',
                                                   # 'tversky',
                                                   # 'euclidean'
                                                   ])

    full_hyperp = {"similarity": similarity,
                   "topK": optuna_trial.suggest_int("topK", 100, 700),
                   "shrink": optuna_trial.suggest_int("shrink", 0, 250),
                   }

    if similarity == "asymmetric":
        full_hyperp["asymmetric_alpha"] = optuna_trial.suggest_float("asymmetric_alpha", 0, 2, log=False)
        full_hyperp["normalize"] = True

    elif similarity == "tversky":
        full_hyperp["tversky_alpha"] = optuna_trial.suggest_float("tversky_alpha", 0, 2, log=False)
        full_hyperp["tversky_beta"] = optuna_trial.suggest_float("tversky_beta", 0, 2, log=False)
        full_hyperp["normalize"] = True

    elif similarity == "euclidean":
        full_hyperp["normalize_avg_row"] = optuna_trial.suggest_categorical("normalize_avg_row", [True, False])
        full_hyperp["similarity_from_distance_mode"] = optuna_trial.suggest_categorical("similarity_from_distance_mode",
                                                                                        ["lin", "log", "exp"])
        full_hyperp["normalize"] = optuna_trial.suggest_categorical("normalize", [True, False])

    recommender_instance.fit(**full_hyperp)

    result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)

    return result_df.loc[10]["MAP"]


def objective_function_P3alpha(optuna_trial):
    full_hyperp =   {"topK": optuna_trial.suggest_int("TopK", 5, 400),
                    "alpha": optuna_trial.suggest_float("alpha", 0, 0.7, log=False),
                    "normalize_similarity": True,
                     }

    recommender_name = "P3alpha"
    optuna_trial.set_user_attr("recommender_name", recommender_name)


    recommender_instance = P3alphaRecommender(URM_train)
    recommender_instance.fit(**full_hyperp)

    result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)

    return result_df.loc[10]["MAP"]


# def objective_function_xxx(optuna_trial):
#     full_hyperp = {
#                    }############ Enter the parameters #############
#
#     recommender_name = "xxx"
#     optuna_trial.set_user_attr("recommender_name", recommender_name)
#
#
#     recommender_instance = xxx(URM_train)
#     recommender_instance.fit(**full_hyperp)
#
#     result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)
#
#     return result_df.loc[10]["MAP"]


recommenders = {
                # objective_function_puresvd: 20,
                objective_function_UserKNNCF: 50,
                # objective_function_P3alpha: 100,
                # objective_function_svdpp: 10,
                }

for rec, val in recommenders.items():
    optuna_study = optuna.create_study(direction="maximize")
    save_results = SaveResults()
    optuna_study.optimize(rec,
                        callbacks=[save_results],
                        n_trials=val)

    optuna_study.best_trial.params

    optuna_study.best_trial

    save_results.results_df




