import pandas as pd
from IPython.display import display

# i) Load the .csv file into a dataframe
try:
  df = pd.read_csv('housing.csv')
  print("File loaded successfully.\n")
except FileNotFoundError:
  print("File 'housing.csv' not found. Please upload it to the runtime files.")

# ii) Display information of all columns
print("--- Column Information ---")
info_df = pd.DataFrame({
    'Column Name': df.columns,
    'Non-Null Count': df.count(),
    'Dtype': df.dtypes
}).reset_index(drop=True)
display(info_df)

# iii) Display statistical information of all numerical columns
print("--- Statistical Information (Numerical) ---")
display(df.describe())

# iv) Display the count of unique labels for ocean_proximity column
print("--- Ocean Proximity Value Counts ---")
ocean_counts = df['ocean_proximity'].value_counts().to_frame(name='Count')
display(ocean_counts)

# v) Display which attribute in a dataset have missing values count greater than 0
print("--- Attributes with Missing Values ---")
missing_values = df.isnull().sum()
missing_table = missing_values[missing_values > 0].to_frame(name='Missing Values')
display(missing_table)
