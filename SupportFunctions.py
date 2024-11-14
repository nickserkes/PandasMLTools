
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import math

def check_stationarity(series):
    # Copied from https://machinelearningmastery.com/time-series-data-stationary-python/

    result = adfuller(series)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")

def print_metrics(y_true, y_pred):
    print(f"MAE: {mean_absolute_error(y_true, y_pred)}")
    print(f"MSE: {mean_squared_error(y_true, y_pred)}")
    print(f"R2: {r2_score(y_true, y_pred)}")

def plot_correlation_matrix(df, method='pearson'):
    """
    Calculates and plots the correlation matrix for the given DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        method (str): The method to calculate correlation ('pearson', 'kendall', 'spearman').

    Returns:
        pd.DataFrame: The correlation matrix.
    """
    # Calculate the correlation matrix
    correlation_matrix = df.corr(method=method)
    
    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix')
    plt.show()
    
    # Return the correlation matrix for further analysis if needed
    return correlation_matrix
def plot_columns_vs_index_subgrid(df):
    """
    Plots each column in the DataFrame with respect to the index in a subgrid.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
    # Determine the grid size based on the number of columns
    num_columns = df.shape[1]
    num_rows = math.ceil(num_columns / 3)  # Adjust the number of columns in each row if needed

    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 4))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, column in enumerate(df.columns):
        axes[i].plot(df.index, df[column], label=column)
        axes[i].set_title(f'{column} vs Index')
        axes[i].set_xlabel('Index')
        axes[i].set_ylabel(column)
        axes[i].legend()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()