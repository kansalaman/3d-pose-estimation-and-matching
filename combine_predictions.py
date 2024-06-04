import pandas as pd


prediction_files = ['results/predictions.csv', 'results/orig/predictions.csv', 'results/comb_predictions.csv']
output_file = 'results/combined_predictions.csv'

# use id column to match
# use first file as base, any column in other files will be added to the base

# Load the data file
data = pd.read_csv(prediction_files[0])

for file in prediction_files[1:]:
    # dont repeat same columns and merge on id
    new_data = pd.read_csv(file)
    new_data = new_data.drop(columns=[col for col in new_data.columns if col in data.columns and col != 'id'])
    data = data.merge(new_data, on='id')


data.to_csv(output_file, index=False)
