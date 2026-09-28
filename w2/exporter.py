# exporter.py
import pandas as pd
import os

def export_predictions(dataset_name, algorithm, k, r, predictions):
    # Construct a filename based on parameters
    filename = f"datasets/predictions/predictions_{dataset_name}_{algorithm}_k{k}_r{r}.csv"

    # Convert predictions to DataFrame and save as CSV
    predictions_df = pd.DataFrame(predictions, columns=["Predictions"])
    predictions_df.to_csv(filename, index=False)


def load_predictions(dataset_name, algorithm, k, r):
    # Construct the filename based on the parameters
    filename = f"datasets/predictions/predictions_{dataset_name}_{algorithm}_k{k}_r{r}.csv"

    # Check if the file exists
    if os.path.isfile(filename):
        # Load the predictions from the CSV file
        predictions_df = pd.read_csv(filename)
        return predictions_df['Predictions'].tolist()  # Return predictions as a list

    return []  # Return an empty list if the file does not exist