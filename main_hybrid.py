from Hybrid import Hybrid
from tqdm import tqdm
from DataLoader import DataLoader

data_train_file_path = "/Users/anadrmic/Desktop/POLIMI/New Folder With Items/2/RS/competition/recommender-system-2023-challenge-polimi/data_train.csv"
data_test_file_path = "/Users/anadrmic/Desktop/POLIMI/New Folder With Items/2/RS/competition/recommender-system-2023-challenge-polimi/data_target_users_test.csv"
output_file = "svd_user_alpha_.csv"

dl = DataLoader(data_train_file_path, data_test_file_path)
targets = dl.get_target_users()
dl_URM = dl.URM.tocsr()

recommender = Hybrid(dl_URM)
recommender.fit()  

f = open(output_file, "w+")
f.write("user_id,item_list\n")

for t in tqdm(targets):
    recommended_items = recommender.recommend(t, cutoff=10, remove_seen_flag=True)
    well_formatted = " ".join([str(x) for x in recommended_items])
    f.write(f"{t}, {well_formatted}\n")