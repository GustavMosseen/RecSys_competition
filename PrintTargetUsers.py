import csv
from tqdm import tqdm


def write_target_users(recommender, output_file, targets):
    with open(output_file, mode='w', newline='') as file:
        # Create a CSV writer object
        csv_writer = csv.writer(file)
        # Write the header
        csv_writer.writerow(['user_id', 'item_list'])

        for t in tqdm(targets):
            recommended_items = recommender.recommend(t, cutoff=10, remove_seen_flag=True)
            well_formatted = " ".join([str(x) for x in recommended_items])
            csv_writer.writerow([t, well_formatted])
