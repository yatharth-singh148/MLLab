import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# URL of the housing.csv file
file_url = 'https://media.geeksforgeeks.org/wp-content/uploads/20240319120216/housing.csv'
file_name = 'housing.csv'

try:
    # Download the file if it doesn't exist
    with open(file_name, 'wb') as f:
        response = requests.get(file_url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        f.write(response.content)
    print(f"File '{file_name}' downloaded successfully.")

    # Load the .csv file into a dataframe
    df = pd.read_csv(file_name)
    print(f"File '{file_name}' loaded successfully for plotting.")

    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns

    # Plot histograms for all numerical columns
    plt.figure(figsize=(18, 15))
    for i, col in enumerate(numerical_cols):
        plt.subplot(4, 5, i + 1) # Adjust subplot grid as needed
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Analyze median_income and housing_median_age
    print("\n--- Analysis of median_income histogram ---")
    print("The median_income histogram shows a right-skewed distribution, indicating that most households have lower to moderate incomes, with fewer households having very high incomes. There's a clear peak at lower income values, and the distribution tapers off towards higher incomes.")

    print("\n--- Analysis of housing_median_age histogram ---")
    print("The housing_median_age histogram shows a fairly uniform distribution for much of its range, but with significant peaks at the maximum value (52). This suggests that ages are capped at 52, which could be an artifact of data collection rather than the true age distribution of houses. There might also be a slight increase in frequency towards newer houses and older houses.")

except FileNotFoundError:
    print(f"File '{file_name}' not found locally after download attempt.")
    df = pd.DataFrame() # Create an empty DataFrame to avoid further errors if file is missing
except requests.exceptions.RequestException as e:
    print(f"Failed to download file from '{file_url}': {e}")
    df = pd.DataFrame() # Create an empty DataFrame
except pd.errors.EmptyDataError:
    print(f"Error: '{file_name}' is empty.")
    df = pd.DataFrame() # Create an empty DataFrame
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    df = pd.DataFrame() # Create an empty DataFrame

if df.empty:
    print("Cannot plot histograms as 'df' is empty or could not be loaded.")
