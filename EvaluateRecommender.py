import pandas as pd
from Evaluation.Evaluator import EvaluatorHoldout
import csv


def evaluate(recommender, URM_validation):
    # Evaluation
    cutoff_list = [5, 10, 15]
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list)
    res, _ = evaluator_validation.evaluateRecommender(recommender)

    # Display Result
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(res[["MAP", "PRECISION", "RECALL"]])
    res.to_csv('MAP_results.txt', sep='\t', index=False)  # Print to a file
