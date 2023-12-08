from Hybrid import Hybrid
from tqdm import tqdm
from DataLoader import DataLoader
from PrintTargetUsers import write_target_users

# data_path = "/Users/anadrmic/Desktop/POLIMI/New Folder With Items/2/RS/competition/recommender-system-2023-challenge-polimi"
data_path = "/Users/gustavmosseen/PycharmProjects/pythonProject/RecommenderSystems"
output_file = "hybrid_test_after_hp3_hplambda1.csv"

dl = DataLoader(data_path)
targets = dl.get_target_users()
dl_URM = dl.URM.tocsr()

# Split data
# URM_train, URM_validation = dl.split_the_dataset(dl_URM, 0.7, 2)
# evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])

recommender = Hybrid(dl_URM)
recommender.fit(lambda1=6,
                lambda2=7,
                lambda3=1)

# Print the target users
write_target_users(recommender, output_file, targets)
