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

# The hyperparameters: batch_size, epochs, item_reg, learning_rate, num_factors, result, sgd_mode, user_reg

# Preprocess
path = "/Users/gustavmosseen/PycharmProjects/pythonProject/RecommenderSystems"
output_file = "svd_user_alpha_.csv"

dl = DataLoader(path)
targets = dl.get_target_users()
URM_all = dl.URM.tocsr()
URM_train, URM_validation, URM_test = dl.split_the_dataset(URM_all, 0.8, 3)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

# Optuna
import optuna
import pandas as pd

# SVD++
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_SVDpp_Cython


class SaveResults(object):

    def __init__(self):
        self.results_df = pd.DataFrame()

    def __call__(self, optuna_study, optuna_trial):
        hyperparam_dict = optuna_trial.params.copy()
        hyperparam_dict["result"] = optuna_trial.values[0]

        # Retrieve the optimal number of epochs from the "user attributes" of the trial
        hyperparam_dict["epochs"] = optuna_trial.user_attrs["epochs"]

        self.results_df = pd.concat([self.results_df, pd.DataFrame([hyperparam_dict])], ignore_index=True)


def objective_function_funksvd(optuna_trial):
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

    # Add the number of epochs selected by earlystopping as a "user attribute" of the optuna trial
    epochs = recommender_instance.get_early_stopping_final_epochs_dict()["epochs"]
    optuna_trial.set_user_attr("epochs", epochs)

    result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)

    return result_df.loc[10]["MAP"]


[17]
optuna_study = optuna.create_study(direction="maximize")

save_results = SaveResults()

optuna_study.optimize(objective_function_funksvd,
                      callbacks=[save_results],
                      n_trials=10)

optuna_study.best_trial.params

optuna_study.best_trial

save_results.results_df




