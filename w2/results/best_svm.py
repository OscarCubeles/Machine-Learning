import pandas as pd

# Load the dataset
df_poly = pd.read_csv('svm_poly_analysis.csv')

# Find the top 3 rows with the highest avg_accuracy
top_3_accuracy = df_poly.nlargest(3, 'avg_accuracy')

# Find the top 3 rows with the highest avg_efficiency
top_3_efficiency = df_poly.nlargest(3, 'avg_efficiency')

# Display the results
print("Top 3 rows with highest avg_accuracy:")
print(top_3_accuracy)

print("\nTop 3 rows with highest avg_efficiency:")
print(top_3_efficiency)
