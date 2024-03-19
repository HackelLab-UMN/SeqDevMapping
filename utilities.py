import re
import ast

# from getpass import getpass
# import mysql.connector
# from mysql.connector import connect, Error

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Set a default DPI for all figures
plt.rcParams['figure.dpi'] = 1000
from itertools import combinations

import os

from architectures import*
from raw_data import*

def make_scatterplot_from_model_and_dataset(raw_dataset_df, study_name, db_name):
    best_pipeline, model_name, scaler_name = load_best_model_from_study_name(study_name, db_name)
    X, y, input_names, output_name = load_dataset_from_study_name(raw_dataset_df, study_name) 
    input_names = [name.replace('Average', '') for name in input_names]

    best_pipeline.fit(X,y)
    y_pred = best_pipeline.predict(X)

    # Create a scatterplot of the predicted vs actual y values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y, y=y_pred, color='black')

    # Label the axes and the plot
    plt.ylabel('True ' + output_name)
    plt.xlabel('Predicted ' + output_name)
    plt.title(f'Scatterplot of Predicted vs Actual {output_name} using model: {model_name} \n and feature set {input_names}') #with {scaler_name}')
    
    # Make the y and x axes have the exact same scale
    plt.axis('equal')

    # Set the limits of the y and x axes to the same range
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.xlim(min_val, max_val)
    # plt.ylim(min_val, max_val)

    # Calculate the step size for 7 tick marks and round to the nearest value of 10
    step = round((max_val - min_val) / 8 / 10) * 10

    # Adjust the y-axis limits to include the exact min and max values as ticks
    plt.ylim(min_val - step / 2, max_val + step / 2)

    # Set the tick locations to match the xlim and ylim ranges
    plt.xticks(np.arange(min_val, max_val, step=step)) # Adjust the step as needed
    plt.yticks(np.arange(min_val, max_val, step=step)) # Adjust the step as needed

    # Calculate the Pearson correlation between y_pred and y
    correlation = np.corrcoef(y, y_pred)[0, 1]

    # Add the number of data points as a label within the box
    num_data_points = len(y)
    # plt.text(0.05, 0.95, f'N={num_data_points}, ρ={correlation:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.05, 0.95, f'ρ={correlation:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    plt.tight_layout()
    
    # Create a directory to save the plot
    plot_dir = "scatterplot_pngs"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save the figure
    plot_name = f'{model_name}_{scaler_name}_{output_name}_{str(input_names)}_scatterplot.png'
    plot_name = sanitize_study_filename(plot_name)
    full_output_path = os.path.join(plot_dir, plot_name)
    plt.savefig(full_output_path)
    
    # # Show the plot
    # plt.show()




def load_dataset_from_study_name(raw_dataset_df, study_name):
    """
    Load the dataset corresponding to a given study name.

    Args:
        raw_dataset_df (pd.DataFrame): The raw dataset dataframe.
        study_name (str): The name of the study.

    Returns:
        X_np (np.ndarray): The input features.
        y_np (np.ndarray): The output labels.
    """
    # Extract the output metric and feature set from the study name
    study_parts = study_name.split('__')
    output_metric_name = study_parts[4]
    feature_name_set = study_parts[5].split('_')
    
    # Format the input metric names correctly
    input_metric_names = []
    for feature_name in feature_name_set:
        if feature_name == 'AverageTL':
            input_metric_names.append('Average TL')
        elif feature_name == 'AveragesGFP':
            input_metric_names.append('Average sGFP')
        elif feature_name == 'NSBAverage':
            input_metric_names.append('NSB Average')
        else:
            raise ValueError(f'Unrecognized feature {feature_name}')
    
    # Convert the feature set names into an np array 
    feature_set = []
    for input_name in input_metric_names:
        feature_set.append(raw_dataset_df[input_name])
    
    if isinstance(feature_set, list):
        X_df = pd.concat(feature_set, axis=1)
    elif isinstance(feature_set, pd.DataFrame):
        X_df = feature_set
    elif isinstance(feature_set, pd.Series):
        X_df = feature_set.to_frame()  # Convert the Series to a DataFrame
    else:
        raise ValueError(f'feature set is not a list, a dataframe, or a series: it is type={type(feature_set)}')
    
    # Convert the y metric name to an np array 
    if output_metric_name == 'CDTm_degC':
        formatted_output_metric_name = 'CD Tm (degC)'
    elif output_metric_name == 'SDSPAGE_mg_mL':
        formatted_output_metric_name = 'SDS PAGE (mg/mL)'
    elif output_metric_name == 'DotBlotAvg_ug_mL':
        formatted_output_metric_name = 'Dot Blot Avg (ug/mL)'
    else: 
        raise ValueError(f'Unrecognized output metric {output_metric_name}')

    output_metric = raw_dataset_df[formatted_output_metric_name]
    y_df = pd.DataFrame({'output_metric': output_metric})
    full_df = pd.concat([X_df, y_df], axis=1)
    full_df = full_df.dropna()
    X_new = full_df.iloc[:, :-1]  # All columns except the last column
    y_new = full_df.iloc[:, -1]   # Only the last column
    X_np = X_new.to_numpy()
    y_np = y_new.to_numpy().ravel()  # Flatten the array

    print(f'dim X_np={X_np.shape}')
    print(f'dim of y_np={y_np.shape}')
    print('\n')

    return X_np, y_np, input_metric_names, formatted_output_metric_name



def load_best_model_from_study_name(study_name, db_name):
    # Load the study from the database
    study = optuna.load_study(storage=f'sqlite:///{db_name}.db', study_name=study_name)
    
    # Get the best trial
    best_trial = study.best_trial
    
    # Extract the model class name and the scaler class name from the study name
    study_parts = study_name.split('__')
    model_class_name = study_parts[3]
    scaler_class_name = study_parts[2]
    
    # Instantiate the scaler class
    scaler_class = scaler_classes[scaler_class_name]
    scaler_instance = scaler_class()
    
    # Instantiate the model class with the best trial's parameters
    model_class = model_classes[model_class_name]
    best_model = model_class(**best_trial.params)
    
    # Instantiate the best pipeline (scaler + model)
    best_pipeline = Pipeline([
        ('scaler', scaler_instance),
        ('model', best_model)
    ])
    
    return best_pipeline, model_class_name, scaler_class_name

def list_study_names(db_name):
    # Connect to the database
    storage = optuna.storages.RDBStorage(f'sqlite:///{db_name}.db')
    
    # Get the list of study names
    studies = storage.get_all_studies()
    study_names = [study.study_name for study in studies]
    
    return study_names

def find_best_row(csv_name, output_metric, feature_set, model_list, filter_metric, filter_function):
    """
    Find the best row in a CSV file based on the specified criteria.

    Args:
        csv_name (str): The name of the CSV file.
        output_metric (str): The name of the output metric.
        feature_set (str): The name of the feature set.
        model_list (list): The list of models.

    Returns:
        pandas.DataFrame: The best row.

    """

    # Read the CSV file
    df = pd.read_csv(csv_name)
    
    # Filter rows based on the output metric and feature set
    filtered_df = df[(df['Output Metric'] == output_metric) & (df['Feature Set'] == feature_set) & (df['Model'].isin(model_list))]
    
    # Find the index of the row with the highest full R^2 value
    # max_r2_index = filtered_df['Full Dataset R^2'].idxmax()
    if not filtered_df.empty:
        max_r2_index = filter_function(filtered_df[filter_metric])
    
    else: 
        return None

    
    # Return the first row with the highest full R^2 value
    best_row = filtered_df.loc[max_r2_index]
    
    # Return the best row as a DataFrame
    return best_row

def make_all_name_combinations(var_list):
    '''
    Go over a list of string names, return a list of lists
    of all combinations of names
    '''
    subset_list = []
    for r in range(1,len(var_list) +  1):   # generate all combinations of
        combos = combinations(var_list, r)    # N Choose R over all R from R=N to R=0
        subset_list.extend(combos)
    
    all_combinations=[]
    for sublist in subset_list:     # convert all tuples to lists
        all_combinations.append(list(sublist))
    return all_combinations

def examine_all_combinations(csv_name, protein, output_names, input_names, linear_model_list, nonlinear_model_list, filter_metric, filter_function):
    """
    Evaluates all possible combinations of input and output metrics for a given list of linear and nonlinear models.

    Args:
        csv_name (str): The name of the CSV file containing the model evaluation results.
        protein (str): The name of the protein to filter the data for.
        output_names (List[str]): The list of output metric names.
        input_names (List[str]): The list of input metric names.
        linear_model_list (List[str]): The list of linear model names.
        nonlinear_model_list (List[str]): The list of nonlinear model names.
        filter_metric (str): The name of the metric used to filter the data.
        filter_function: A function that takes a pandas DataFrame and returns the index of the row with the highest value of the specified metric.

    Returns:
        pd.DataFrame: A DataFrame containing the best rows for each combination of output and feature set.
    """
    collapsed_combination_dir = 'collapsed_combination_csvs'
    os.makedirs(collapsed_combination_dir, exist_ok=True)
    
    all_input_combinations = make_all_name_combinations(input_names)

    # Initialize an empty list to store the best rows
    best_rows_list = []

    for output_name in output_names:
        for feature_set in all_input_combinations:
            # Convert the feature set to the format used in the CSV
            formatted_feature_set = str(feature_set) #now this should be a string of a list of strings
            # Find the best row for linear models
            best_row_linear = find_best_row(csv_name, output_name, formatted_feature_set, linear_model_list, filter_metric, filter_function)

            # Find the best row for nonlinear models
            best_row_nonlinear = find_best_row(csv_name, output_name, formatted_feature_set, nonlinear_model_list, filter_metric, filter_function)

            # Convert the Series to a DataFrame with a single row and add the "Model Type" column
            if best_row_linear is not None:
                best_row_linear_df = best_row_linear.to_frame().T
                best_row_linear_df['Model Type'] = 'Linear'
            else: 
                best_row_linear_df = pd.DataFrame()

            if best_row_nonlinear is not None:
                best_row_nonlinear_df = best_row_nonlinear.to_frame().T
                best_row_nonlinear_df['Model Type'] = 'Nonlinear'
            else: 
                best_row_nonlinear_df = pd.DataFrame()
            
            best_rows_list.append(best_row_linear_df)
            best_rows_list.append(best_row_nonlinear_df)

    # Concatenate all the rows into a single DataFrame
    best_rows_df = pd.concat(best_rows_list, ignore_index=True)
    
    # Calculate the ratio of Test MSE to CV MSE
    best_rows_df['Test_CV_MSE_Ratio'] = best_rows_df['Test MSE'] / best_rows_df['CV MSE']

    collapsed_csv_name = f'{protein}_best_combinations_{output_names}_{filter_metric}'
    collapsed_csv_name = sanitize_study_filename(collapsed_csv_name)
    csv_file_path = os.path.join(collapsed_combination_dir, f'{collapsed_csv_name}.csv')
    
    best_rows_df.to_csv(csv_file_path)

    return best_rows_df

def make_fullR2_barchart(df, protein, output_name, new_filename):
    """
    This function reads a CSV file containing model evaluation results, filters the data for a specific output metric, 
    groups the data by feature set and model type, calculates the mean R^2 score for each group, reshapes the data, 
    and saves the result to a new CSV file.

    Parameters:
    input_csv_filename (str): The name of the CSV file to read.
    protein (str): The name of the protein to filter the data for. (Note: This parameter is currently overwritten in the function and not used.)
    output_name (str): The name of the output metric to filter the data for.
    new_filename (str): The base name of the new CSV file to save the results to.

    Returns:
    None
    """
    # Define useful names for saving results
    refined_output_name = sanitize_study_filename(output_name)
    csv_dir = "barchart_data_csvs"
    png_dir = "barchart_pngs"
    sub_dir = "R2_barchart_pngs"
    sub_dir_path = os.path.join(png_dir, sub_dir) # Create the full path for the subdirectory
    os.makedirs(sub_dir_path, exist_ok=True) # Create the subdirectory if it doesn't exist

    # Create the directories if they don't exist
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    # Filter the DataFrame for linear models
    filtered_df = df[df['Output Metric'] == output_name]

    # Group the data by feature set and model type, and calculate the mean R^2 score for each group
    grouped_df = filtered_df.groupby(['Feature Set', 'Model Type']) # gives a GroupBy object that groups the data by matching feature set and model type

    # Calculate the mean 'Full Dataset R^2' score for each group. This results in a Series (or DataFrame if there are multiple columns to aggregate) where the index is a MultiIndex consisting of the unique combinations of 'Feature Set' and 'Model Type', and the values are the mean 'Full Dataset R^2' scores for each group.
    grouped_df = grouped_df['Full Dataset R^2'].mean() 

    # Reshape the DataFrame or Series produced by the .mean() function. This "unstacks" the levels of the index to create a new DataFrame where the first level of the index ('Feature Set') becomes columns, and the second level of the index ('Model Type') becomes the index of the new DataFrame. Each cell in this new DataFrame contains the mean 'Full Dataset R^2' score for the corresponding combination of 'Feature Set' and 'Model Type'.
    grouped_df = grouped_df.unstack()
    csv_name = f'{new_filename}_{refined_output_name}_grouped.csv'
    csv_name = os.path.join(csv_dir, csv_name)
    grouped_df.to_csv(csv_name)

    ### Trying to move rows correctly
    # Create a new index for sorting based on the number of features in each feature set
    grouped_df['SortIndex'] = grouped_df.index.map(lambda x: len(eval(x))) # this is applied to each tuple in the MultiIndex

    # Sort the DataFrame based on the new index
    grouped_df = grouped_df.sort_values(by='SortIndex', ascending=False)

    # Drop the temporary 'SortIndex' column
    grouped_df = grouped_df.drop(columns=['SortIndex'])

    # Save the sorted DataFrame to a CSV file
    csv_name = f'{new_filename}_{refined_output_name}_grouped_sorted.csv'
    csv_name = os.path.join(csv_dir, csv_name)
    grouped_df.to_csv(csv_name)

    # Check if the DataFrame is empty or if it contains non-numeric data
    if grouped_df.empty:
        print(f"No numeric data to plot for {protein} {output_name}.")
        return

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    grouped_df.plot(kind='bar', stacked=False, ax=plt.gca())
    plt.xlabel('Feature Sets')
    plt.ylabel('Full Dataset R^2 Score')
    plt.title(f'Full Dataset R^2 Score for {protein} Feature Sets for {output_name}')
    plt.xticks(rotation=45)
    plt.legend(title='Model Type')
    plt.tight_layout()

    plot_name = f'{protein}_{new_filename}_{refined_output_name}_grouped_bar.png'
    plot_name = os.path.join(sub_dir_path, plot_name)
    plt.savefig(plot_name)
    # plt.show()




def make_mse_barchart(df, protein, output_name, new_filename, sort_by='num_features'):
    """
    This function reads a CSV file containing model evaluation results, filters the data for a specific output metric, 
    groups the data by feature set and model type, calculates the mean MSE score for each group, reshapes the data, 
    and saves the result to a new CSV file.

    Args:
        df (pandas DataFrame): The pandas DataFrame to be used as input.
        protein (str): The name of the protein to filter the data for.
        output_name (str): The name of the output metric to filter the data for.
        new_filename (str): The base name of the new CSV file to save the results to.

    Returns:
        None
    """
    # Define useful names for saving results
    refined_output_name = sanitize_study_filename(output_name)
    csv_dir = "barchart_data_csvs"
    png_dir = "barchart_pngs"
    sub_dir = "barchart_TestvsCV_only_pngs"
    sub_dir_path = os.path.join(png_dir, sub_dir) # Create the full path for the subdirectory
    os.makedirs(sub_dir_path, exist_ok=True) # Create the subdirectory if it doesn't exist

    # Create the directories if they don't exist
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    
    # Filter the DataFrame for the specified output metric
    filtered_df = df[df['Output Metric'] == output_name]

    # Group the data by feature set and model type, and calculate the mean MSE score for each group
    grouped_df = filtered_df.groupby(['Feature Set', 'Model Type'])
    grouped_df = grouped_df[['Test MSE', 'CV MSE']].mean().reset_index()
    
    # Create a new column that combines 'Feature Set' and 'Model Type' for the x-axis labels with a newline character
    grouped_df['FeatureSet_ModelType'] = grouped_df['Feature Set'] + '\n' + grouped_df['Model Type']

    # Create a new index for sorting based on the specified criteria
    if sort_by == 'num_features':
        # Sort by the number of features in each feature set
        grouped_df['SortIndex'] = grouped_df['Feature Set'].apply(lambda x: len(eval(x)))
        
        # Sort the DataFrame based on the new index: want to see most features first
        grouped_df = grouped_df.sort_values(by='SortIndex', ascending=False)
    elif sort_by == 'test_score':
        # Sort by the Test MSE score from lowest to highest
        grouped_df['SortIndex'] = grouped_df['Test MSE']
        
        # Sort the DataFrame based on the new index: want to see lowest test MSEs first
        grouped_df = grouped_df.sort_values(by='SortIndex', ascending=True)
    else:
        raise ValueError("Invalid sort_by parameter. Must be 'num_features' or 'test_score'.")

    

    # Drop the temporary 'SortIndex' column
    grouped_df = grouped_df.drop(columns=['SortIndex'])

    # # Save the reshaped DataFrame dto a CSV file
    # csv_name = f'{new_filename}_{refined_output_name}_grouped.csv'
    # csv_name = os.path.join(csv_dir, csv_name)
    # grouped_df.to_csv(csv_name)

    # Create a bar chart
    fig, ax = plt.subplots(figsize=(16, 6))
    bar_width = 0.35 # width of the bars
    index = np.arange(len(grouped_df['FeatureSet_ModelType']))
    # ax.bar(grouped_df['FeatureSet_ModelType'], grouped_df['Test MSE'], label=grouped_df['Model Type'])

    # Plot the Test MSE bars
    rects1 = ax.bar(index, grouped_df['Test MSE'], bar_width, label='Test MSE')

    # Plot the CV MSE bars
    rects2 = ax.bar(index + bar_width, grouped_df['CV MSE'], bar_width, label='CV MSE')

    # Add labels and title
    ax.set_xlabel('Feature Sets and Model Types')
    ax.set_ylabel('MSE Score')
    ax.set_title(f'{protein} MSE Score for Feature Sets and Model Types for {output_name} \n Sorted by {sort_by}')
    ax.set_xticks(index + bar_width / 2) # Set the x-ticks to the middle of the bars
    ax.set_xticklabels(grouped_df['FeatureSet_ModelType'], rotation=45) # Rotation is set to 0 for multi-line labels
    ax.legend()

    # Add labels to the bars
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    plt.tight_layout()

    plot_name = f'{new_filename}_{refined_output_name}_grouped_MSE_CVtest_bar_sortedBy{sort_by}.png'
    plot_name = os.path.join(sub_dir_path, plot_name) # Use the subdirectory for saving the figure
    plt.savefig(plot_name)
    # plt.show()


def make_mse_ratio_barchart(df, protein, output_name, new_filename):
    # Define useful names for saving results
    refined_output_name = sanitize_study_filename(output_name)
    csv_dir = "barchart_data_csvs"
    
    png_dir = "barchart_pngs"
    sub_dir = "barchart_TestCV_Ratio_pngs"
    sub_dir_path = os.path.join(png_dir, sub_dir) # Create the full path for the subdirectory
    os.makedirs(sub_dir_path, exist_ok=True) # Create the subdirectory if it doesn't exist

    # Create the directories if they don't exist
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    # Filter the DataFrame for the specified output metric
    filtered_df = df[df['Output Metric'] == output_name]

    # Sort the DataFrame based on the Test_CV_MSE_Ratio in ascending order
    filtered_df = filtered_df.sort_values(by='Test_CV_MSE_Ratio', ascending=True)

    # Create a new column that combines 'Feature Set' and 'Model Type' for the x-axis labels with a newline character
    filtered_df['FeatureSet_ModelType'] = filtered_df['Feature Set'] + '\n' + filtered_df['Model Type']

    # Drop the 'Test MSE' and 'CV MSE' columns as they are not needed for this plot
    filtered_df = filtered_df.drop(columns=['Test MSE', 'CV MSE'])

    # Rename the 'Test_CV_MSE_Ratio' column to 'MSE Score' for clarity in the plot
    filtered_df = filtered_df.rename(columns={'Test_CV_MSE_Ratio': 'MSE Score'})

    # Create a bar chart
    fig, ax = plt.subplots(figsize=(20, 6))
    bar_width = 0.35 # width of the bars
    index = np.arange(len(filtered_df['FeatureSet_ModelType']))


    # Plot the MSE Score bars
    rects = ax.bar(index, filtered_df['MSE Score'], bar_width, label='MSE Score')

    # Add labels and title
    ax.set_xlabel('Feature Sets and Model Types')
    ax.set_ylabel('Test/CV MSE Ratio')
    ax.set_title(f'{protein} Test/CV MSE Ratio for Feature Sets and Model Types for {output_name}')
    ax.set_xticks(index) # Set the x-ticks to the middle of the bars
    ax.set_xticklabels(filtered_df['FeatureSet_ModelType'], rotation=45) # Rotation is set to 0 for multi-line labels

    # Add labels to the bars
    ax.bar_label(rects, padding=3)

    plt.tight_layout()

    # Save the figure
    plot_name = f'{new_filename}_{refined_output_name}_grouped_MSE_ratio.png'
    plot_name = os.path.join(sub_dir_path, plot_name) # Use the subdirectory for saving the figure
    fig.savefig(plot_name)



def make_mse_triple_barchart(df, protein, output_name, new_filename):
    # Define useful names for saving results
    refined_output_name = sanitize_study_filename(output_name)
    csv_dir = "barchart_data_csvs"
    
    png_dir = "barchart_pngs"
    sub_dir = "barchart_TestCV_Ratio_Triple_pngs"
    sub_dir_path = os.path.join(png_dir, sub_dir) # Create the full path for the subdirectory
    os.makedirs(sub_dir_path, exist_ok=True) # Create the subdirectory if it doesn't exist

    # Create the directories if they don't exist
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    
    # Filter the DataFrame for the specific protein and output metric
    df_filtered = df[df['Protein'] == protein]
    df_filtered = df_filtered[df_filtered['Output Metric'] == output_name]
    
    # Group the data by feature set and model type, and calculate the mean MSE score for each group
    grouped_df = df_filtered.groupby(['Feature Set', 'Model Type'])
    grouped_df = grouped_df[['Test MSE', 'CV MSE', 'Test_CV_MSE_Ratio']].mean().reset_index()
    
    # Create a new column that combines 'Feature Set' and 'Model Type' for the x-axis labels with a newline character
    grouped_df['FeatureSet_ModelType'] = grouped_df['Feature Set'] + '\n' + grouped_df['Model Type']
    
    # Sort the DataFrame based on the number of features in each feature set
    grouped_df['NumFeatures'] = grouped_df['Feature Set'].apply(lambda x: len(eval(x)))
    grouped_df = grouped_df.sort_values(by='NumFeatures', ascending=False)

    # Create utila bar chart with Test MSE and CV MSE on the left y-axis
    fig, ax1 = plt.subplots(figsize=(24, 6))
    bar_width = 0.15 # width of the bars
    index = np.arange(len(grouped_df['FeatureSet_ModelType']))
    
    # Plot the Test MSE bars
    rects1 = ax1.bar(index, grouped_df['Test MSE'], bar_width, label='Test MSE')
    
    # Plot the CV MSE bars
    rects2 = ax1.bar(index + bar_width, grouped_df['CV MSE'], bar_width, label='CV MSE')
    
    # Create a second y-axis for Test_CV_MSE_Ratio
    ax2 = ax1.twinx()
    ax2.bar(index + 2 * bar_width, grouped_df['Test_CV_MSE_Ratio'], bar_width, label='Test_CV_MSE_Ratio', color='g')
    
    # Add a horizontal green dashed line at Test_CV_MSE_Ratio = 1.5
    ax2.axhline(y=1.5, color='g', linestyle='--', linewidth=1)

    # Add labels and title
    ax1.set_xlabel('Feature Sets and Model Types')
    ax1.set_ylabel('MSE Score')
    ax2.set_ylabel('Test/CV MSE Ratio')
    plt.title(f'{protein} MSE Comparison for Output Metric: {output_name}')
    
    # Set the x-ticks to the middle of the bars
    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(grouped_df['FeatureSet_ModelType'], rotation=45)
    
    # Add labels to the bars
    ax1.bar_label(rects1, padding=3)
    ax1.bar_label(rects2, padding=3)
    ax2.bar_label(ax2.containers[0], padding=3)

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Adjust the layout
    plt.tight_layout()
    
    # Save the figure
    plot_name = f'{new_filename}_{refined_output_name}_grouped_MSE_ratio.png'
    plot_name = os.path.join(sub_dir_path, plot_name) # Use the subdirectory for saving the figure
    fig.savefig(plot_name)

# Example usage
# df = pd.read_csv('Aby_best_combinations__DotBlotAvg_ug_mL__CDTm_degC__TestMSE.csv')
# make_mse_triple_barchart(df, 'Aby', 'Dot Blot Avg (ug/mL)', 'Aby_mse_triple_barchart.png')

def make_mse_double_barchart(df, protein, output_name, new_filename):
    # Define useful names for saving results
    refined_output_name = sanitize_study_filename(output_name)
    csv_dir = "barchart_data_csvs"
    
    png_dir = "barchart_pngs"
    sub_dir = "barchart_TestCV_Ratio_Double_pngs"
    sub_dir_path = os.path.join(png_dir, sub_dir) # Create the full path for the subdirectory
    os.makedirs(sub_dir_path, exist_ok=True) # Create the subdirectory if it doesn't exist
    
    # Create the directories if they don't exist
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    
    # Filter the DataFrame for the specific protein and output metric
    df_filtered = df[df['Protein'] == protein]
    df_filtered = df_filtered[df_filtered['Output Metric'] == output_name]
    
    # Group the data by feature set and model type, and calculate the mean MSE score for each group
    grouped_df = df_filtered.groupby(['Feature Set', 'Model Type'])
    grouped_df = grouped_df[['Test MSE', 'Test_CV_MSE_Ratio']].mean().reset_index()
    
    # Create a new column that combines 'Feature Set' and 'Model Type' for the x-axis labels with a newline character
    grouped_df['FeatureSet_ModelType'] = grouped_df['Feature Set'] + '\n' + grouped_df['Model Type']
    
    # Sort the DataFrame based on the number of features in each feature set
    grouped_df['NumFeatures'] = grouped_df['Feature Set'].apply(lambda x: len(eval(x)))
    grouped_df = grouped_df.sort_values(by='NumFeatures', ascending=False)
    
    # Create a bar chart with Test MSE and Test MSE/CV MSE Ratio on the left y-axis
    fig, ax1 = plt.subplots(figsize=(24, 6))
    bar_width = 0.15 # width of the bars
    index = np.arange(len(grouped_df['FeatureSet_ModelType']))
    
    # Plot the Test MSE bars
    rects1 = ax1.bar(index, grouped_df['Test MSE'], bar_width, label='Test MSE')
    
    # Create a second y-axis for Test_CV_MSE_Ratio
    ax2 = ax1.twinx()
    rects2  = ax2.bar(index + bar_width, grouped_df['Test_CV_MSE_Ratio'], bar_width, label='Test_CV_MSE_Ratio', color='g')
    
    # Add a horizontal green dashed line at Test_CV_MSE_Ratio = 1.5
    ax2.axhline(y=1.0, color='g', linestyle='--', linewidth=1)
    
    feature_sets = grouped_df['FeatureSet_ModelType'].tolist()
    # print(f'feature_sets = {feature_sets}')
    # print(f'output_name = {output_name}')
    # print('\n'*2)

    # Conditionally set the alpha of the bars
    # This hardcode the best 3 models we discuss in the
    # Nonlinear analysis section
    for rect1, rect2, feature_set in zip(rects1, rects2, feature_sets):
        if feature_set == "['Average TL', 'Average sGFP']\nLinear" and protein == "Aby" and output_name == 'CD Tm (degC)': # Replace 'some_threshold' with the value you want to use
            print(f'feature_set={feature_set} "\n" and protein={protein}\n')
            rect1.set_alpha(1.0) # Set the alpha to 1.0 for full opacity if the condition is met
            rect2.set_alpha(1.0)
        elif feature_set == "['Average sGFP', 'NSB Average']\nNonlinear" and protein == "Aby" and output_name == 'Dot Blot Avg (ug/mL)':
            print(f'feature_set={feature_set} "\n" and protein={protein}\n')
            rect1.set_alpha(1.0) # Set the alpha to 1.0 for full opacity if the condition is met
            rect2.set_alpha(1.0)
        elif feature_set == "['Average TL']\nNonlinear" and protein == "Fn" and output_name == 'CD Tm (degC)':
            print(f'feature_set={feature_set} "\n" and protein={protein}\n')
            rect1.set_alpha(1.0) # Set the alpha to 1.0 for full opacity if the condition is met
            rect2.set_alpha(1.0)
        else:
            rect1.set_alpha(0.5) # Set the alpha to 0.5 for transparency if the condition is not met
            rect2.set_alpha(0.5)


    # Add labels and title
    # ax1.set_xlabel('Feature Sets and Model Types')
    ax1.set_ylabel('Test MSE')
    ax2.set_ylabel('Test/CV MSE Ratio')
    plt.title(f'{protein} MSE Comparison for {output_name}')
    
    # Set the x-ticks to the middle of the bars
    ax1.set_xticks(index + bar_width / 2)
    # ax1.set_xticklabels(grouped_df['FeatureSet_ModelType'], rotation=45)
    ax1.set_xticklabels([])

    # Add labels to the bars
    # ax1.bar_label(rects1, padding=3)
    # ax2.bar_label(ax2.containers[0], padding=3)
    
    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Adjust the layout
    plt.tight_layout()
    
    # Save the figure
    plot_name = f'{new_filename}_{refined_output_name}_grouped_MSE_ratio_double.png'
    plot_name = os.path.join(sub_dir_path, plot_name) # Use the subdirectory for saving the figure
    fig.savefig(plot_name) #dpi = 300)
    # fig.savefig(plot_name, dpi = 300)

def make_stacked_barcharts(dfs, proteins, output_names, new_filename):
    # Define useful names for saving results
    # refined_output_name = sanitize_study_filename(output_name)
    csv_dir = "barchart_data_csvs"
    
    png_dir = "barchart_pngs"
    sub_dir = "barchart_TestCV_Ratio_Double_pngs_combo"
    sub_dir_path = os.path.join(png_dir, sub_dir) # Create the full path for the subdirectory
    os.makedirs(sub_dir_path, exist_ok=True) # Create the subdirectory if it doesn't exist
    
    # Create the directories if they don't exist
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    
    # Define the number of subplots
    num_subplots = len(dfs)
    # bar_width = 0.15 # width of the bars
    bar_width = .4 # width of the bars

    # Create a figure with subplots
    fig, axes = plt.subplots(2,2, figsize=(12, 12))
    # bar_width = 0.15 # width of the bars

    # Flatten the axes array to make it easier to iterate over
    axes = axes.flatten()
    
    # Iterate over the lists of parameters
    for i, (df, protein, output_name) in enumerate(zip(dfs, proteins, output_names)):

        # Filter the DataFrame for the specific protein and output metric
        df_filtered = df[df['Protein'] == protein]
        df_filtered = df_filtered[df_filtered['Output Metric'] == output_name]
        
        # Group the data by feature set and model type, and calculate the mean MSE score for each group
        grouped_df = df_filtered.groupby(['Feature Set', 'Model Type'])

        # Create a second y-axis for Test_CV_MSE_Ratio
        ax1 = axes[i]
        print(f'ax1 is {type(ax1)}')
        grouped_df = grouped_df[['Test MSE', 'Test_CV_MSE_Ratio']].mean().reset_index()
        
        # Create a new column that combines 'Feature Set' and 'Model Type' for the x-axis labels with a newline character
        grouped_df['FeatureSet_ModelType'] = grouped_df['Feature Set'] + '\n' + grouped_df['Model Type']
        
        # Sort the DataFrame based on the number of features in each feature set
        grouped_df['NumFeatures'] = grouped_df['Feature Set'].apply(lambda x: len(eval(x)))
        grouped_df = grouped_df.sort_values(by='NumFeatures', ascending=False)
        feature_sets = grouped_df['FeatureSet_ModelType'].tolist()  
        index = np.arange(len(grouped_df['FeatureSet_ModelType']))

        # Plot the Test MSE bars
        rects1 = ax1.bar(index, grouped_df['Test MSE'], bar_width, label='Test MSE')
        
        # Create a second y-axis for Test_CV_MSE_Ratio
        ax2 = ax1.twinx()
        rects2  = ax2.bar(index + bar_width, grouped_df['Test_CV_MSE_Ratio'], bar_width, label='Test MSE / CV MSE Ratio', color='g')
        
        # Add a horizontal green dashed line at Test_CV_MSE_Ratio = 1.5
        ax2.axhline(y=1.0, color='g', linestyle='--', linewidth=1)

        # Conditionally set the alpha of the bars
        # This hardcode the best 3 models we discuss in the
        # Nonlinear analysis section
        for rect1, rect2, feature_set in zip(rects1, rects2, feature_sets):
            if feature_set == "['Average TL', 'Average sGFP']\nLinear" and protein == "Aby" and output_name == 'CD Tm (degC)': # Replace 'some_threshold' with the value you want to use
                print(f'feature_set={feature_set} "\n" and protein={protein}\n')
                rect1.set_alpha(1.0) # Set the alpha to 1.0 for full opacity if the condition is met
                rect2.set_alpha(1.0)
            elif feature_set == "['Average sGFP', 'NSB Average']\nNonlinear" and protein == "Aby" and output_name == 'Dot Blot Avg (ug/mL)':
                print(f'feature_set={feature_set} "\n" and protein={protein}\n')
                rect1.set_alpha(1.0) # Set the alpha to 1.0 for full opacity if the condition is met
                rect2.set_alpha(1.0)
            elif feature_set == "['Average TL']\nNonlinear" and protein == "Fn" and output_name == 'CD Tm (degC)':
                print(f'feature_set={feature_set} "\n" and protein={protein}\n')
                rect1.set_alpha(1.0) # Set the alpha to 1.0 for full opacity if the condition is met
                rect2.set_alpha(1.0)
            else:
                rect1.set_alpha(0.5) # Set the alpha to 0.5 for transparency if the condition is not met
                rect2.set_alpha(0.5)

        if i==0: # Add labels and title
            # ax1.set_xlabel('Feature Sets and Model Types')
            # ax1.set_ylabel('Test MSE')
            # ax2.set_ylabel('Test/CV MSE Ratio')
                
            # ax1.set_xticklabels(grouped_df['FeatureSet_ModelType'], rotation=45)
        
            # Add labels to the bars
            # ax1.bar_label(rects1, padding=3)
            # ax2.bar_label(ax2.containers[0], padding=3)
            
            # Add legends
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        # Set the x-ticks to the middle of the bars
        ax1.set_xticks(index + bar_width / 2)
        ax1.set_xticklabels([])
        plt.title(f'{protein} {output_name}')
        
        # ax2.set_yticks(np.linspace(round(ax2.get_ylim()[0], 3), round(ax2.get_ylim()[1], 3), 5))
        # ax2.set_yticks()

    # Set the x-label and y-labels for the entire figure
    # fig.text(0.5, 0.04, 'Model Types', ha='center', va='center')
    # fig.text(0.06, 0.5, 'Test MSE', ha='center', va='center', rotation='vertical')
    # fig.text(0.94, 0.5, 'Test/CV MSE Ratio', ha='center', va='center', rotation='vertical')

    # Adjust the layout
    plt.tight_layout()

    # Save the combined figure
    plot_name = f'{new_filename}_grouped_MSE_ratio_double_combo.png'
    plot_name = os.path.join(sub_dir_path, plot_name) # Use the subdirectory for saving the figure
    fig.savefig(plot_name) #dpi = 300)
    # fig.savefig(plot_name, dpi = 300)




def calculate_information_array(raw_df, names_combined, information_metric_function):
    """
    Calculate the pairwise information between each pair of features in a dataframe.

    Args:
        raw_df (pd.DataFrame): The input dataframe, containing the features of interest.
        names_combined (List[str]): The names of the combined features, used to calculate the mutual information.
        information_metric_function: A function that calculates the information between two features, given as a NumPy array.

    Returns:
        np.ndarray: A square numpy array containing the pairwise information values.

    Raises:
        TypeError: If the return type of the information_metric_function is unexpected.

    """
    info_array = np.zeros((len(names_combined), len(names_combined)))
    for i, name1 in enumerate(names_combined):
        for j, name2 in enumerate(names_combined):
            if i != j: # don't want to compare the same pair of features
                series1 = raw_df[name1]
                series2 = raw_df[name2]
                
                # Filter out or replace '#REF!' values
                series1 = series1.replace('#REF!', np.nan)
                series2 = series2.replace('#REF!', np.nan)
                
                # Drop rows with NaN values
                df = pd.DataFrame({name1: series1, name2: series2}).dropna()

                # print(f'df shape is {df.shape}')

                variable1 = df[name1].values.flatten().reshape(-1,1)
                variable2 = df[name2].values.flatten().reshape(-1,1)
                
                # print(f'variable1 shape is {variable1.shape}, variable2 shape is {variable2.shape}')
                # print(f'variable1 is {variable1}, variable2 is {variable2}')
                
                # Apply the information metric function
                result = information_metric_function(variable1, variable2)
                # print(f'result={result} of type: {type(result)}')

                # Check if the result is a scalar or a tuple/object
                if isinstance(result, (float, int)):
                    info = result
                elif isinstance(result, np.ndarray):
                    info = result[0]
                elif hasattr(result, 'correlation'): # Check if the result has a 'correlation' attribute (e.g., for spearmanr)
                    info = result.correlation
                else:
                    # Raise an error for unexpected return types
                    raise TypeError(f"Unexpected return type from information metric function: {type(result)}")
                # print(f'info={info} of type: {type(info)}')
                
                info_array[i, j] = info
    return info_array

from scipy.stats import spearmanr

def calculate_full_spearman_information_array(raw_df, names_combined):
    """
    Calculate the pairwise Spearman correlation and significance between each pair of features in a dataframe.

    Args:
        raw_df (pd.DataFrame): The input dataframe, containing the features of interest.
        names_combined (List[str]): The names of the combined features, used to calculate the Spearman correlation.

    Returns:
        np.ndarray: A square numpy array containing the pairwise Spearman correlation and significance values as tuples.
    """
    info_array = np.zeros((len(names_combined), len(names_combined), 2)) # Add an extra dimension for the tuple
    for i, name1 in enumerate(names_combined):
        for j, name2 in enumerate(names_combined):
            if i != j: # don't want to compare the same pair of features
                series1 = raw_df[name1]
                series2 = raw_df[name2]
                
                # Filter out or replace '#REF!' values
                series1 = series1.replace('#REF!', np.nan)
                series2 = series2.replace('#REF!', np.nan)
                
                # Drop rows with NaN values
                df = pd.DataFrame({name1: series1, name2: series2}).dropna()
                
                variable1 = df[name1].values.flatten().reshape(-1,1)
                variable2 = df[name2].values.flatten().reshape(-1,1)
                
                # Calculate Spearman correlation and significance
                result = spearmanr(variable1, variable2)
                
                info_array[i, j] = result
    return info_array


def plot_information_heatmap(protein, metric_string, info_array, names_combined):
    """
    Plots a heatmap of pairwise mutual information between features for a given protein.

    Args:
        protein (str): The name of the protein for which to plot the heatmap.
        metric_string (str): The name of the information metric used to calculate the pairwise mutual information.
        info_array (numpy.ndarray): A square numpy array containing the pairwise mutual information values.
        names_combined (list): A list of the names of the combined features used to calculate the mutual information.

    Returns:
        None

    """
    heatmap_dir = "heatmap_pngs"
    os.makedirs(heatmap_dir,exist_ok=True)

    # sub_dir = """ # Not implemented
    # sub_dir_path = os.path.join(png_dir, sub_dir) # Create the full path for the subdirectory
    # os.makedirs(sub_dir_path, exist_ok=True) # Create the subdirectory if it doesn't exist


    np.fill_diagonal(info_array, np.nan)
    if metric_string == "Mutual Information":
        vmin = 0
        vmax = 0.5
    elif metric_string == "Spearman Correlation":
        vmin = -1
        vmax = 1
    else:
        vmin = np.nanmin(info_array)
        vmax = np.nanmax(info_array)
    mutual_info_df = pd.DataFrame(info_array, index=names_combined, columns=names_combined)
    
    # # Mask the upper triangle of the DataFrame
    mutual_info_df = mutual_info_df.where(np.tril(np.ones(mutual_info_df.shape)).astype(bool))
    
    # Remove the last row and column from the DataFrame
    mutual_info_df = mutual_info_df.iloc[1:, :-1]

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(mutual_info_df, annot=False, fmt=".2f", cmap="coolwarm", cbar=True, xticklabels=True, yticklabels=True, vmin=vmin, vmax=vmax)
    
    # Tilt x-axis labels by 90 degrees
    plt.xticks(rotation=90)
    
    # Set y-axis labels to read left to right
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()
    plt.yticks(rotation=0) # Ensure y-axis labels are horizontal
    
    # Improve the appearance of the color bar
    cbar = ax.collections[0].colorbar
    cbar.set_label(f'{metric_string}', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)

    # # Round the minimum, middle, and maximum values to the nearest .05
    # vmin_rounded = round(vmin / 0.05) * 0.05
    # vmax_rounded = round(vmax / 0.05) * 0.05

    # Calculate the middle value
    middle_value = (vmin + vmax) / 2
    
    # Set tick positions and labels on the color bar
    cbar.ax.set_yticks([vmin, middle_value, vmax])
    cbar.ax.set_yticklabels(['{:.2f}'.format(vmin), '{:.2f}'.format(middle_value), '{:.2f}'.format(vmax)])

    # # Set tick positions and labels on the color bar
    # cbar.ax.set_yticks([vmin_rounded, middle_value, vmax_rounded])
    # cbar.ax.set_yticklabels(['{:.2f}'.format(vmin_rounded), '{:.2f}'.format(middle_value), '{:.2f}'.format(vmax_rounded)])
    
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.title(f'{protein} Pairwise {metric_string} Heatmap')
    plt.tight_layout()

    heatmap_name = f'{protein}_{metric_string}_heatmap.png'
    heatmap_name = os.path.join(heatmap_dir, heatmap_name)
    plt.savefig(heatmap_name)
    # plt.show()


def plot_information_table(protein, metric_string, info_array, names_combined):
    """
    Plots a table of pairwise mutual information between features for a given protein.

    Args:
        protein (str): The name of the protein for which to plot the table.
        metric_string (str): The name of the information metric used to calculate the pairwise mutual information.
        info_array (numpy.ndarray): A square numpy array containing the pairwise mutual information values.
        names_combined (list): A list of the names of the combined features used to calculate the mutual information.

    Returns:
        None
    """
    table_dir = "table_pngs"
    os.makedirs(table_dir, exist_ok=True)

    # Create the DataFrame without replacing -1 with "NA"
    mutual_info_df = pd.DataFrame(info_array, index=names_combined, columns=names_combined)

    # Create a boolean mask for the lower triangle of the DataFrame
    lower_triangle_mask = np.tril(np.ones(mutual_info_df.shape)).astype(bool)

    # Use the mask to keep only the lower triangle of the DataFrame and replace the upper triangle with NaN
    mutual_info_df = mutual_info_df.where(lower_triangle_mask)
    
    # Replace NaN values with "NA" in the DataFrame
    mutual_info_df = mutual_info_df.replace(np.nan, "NA")

    # Remove the last row and column from the DataFrame
    mutual_info_df = mutual_info_df.iloc[1:, :-1]

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.axis('tight')
    ax.axis('off')

    # Format the numbers in the DataFrame to have 2 decimal places
    formatted_info_array = [[f"{num:.3f}" if isinstance(num, (float, int)) else num for num in row] for row in mutual_info_df.values]

    # Create a DataFrame from the formatted_info_array
    formatted_df = pd.DataFrame(formatted_info_array, index=mutual_info_df.index, columns=mutual_info_df.columns)

    # Create a table
    table = plt.table(cellText=formatted_info_array,
                 rowLabels=mutual_info_df.index,
                 colLabels=mutual_info_df.columns,
                 cellLoc = 'center',
                 loc='center')

    # Set the font size of the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Set the cell colors
    for i in range(len(mutual_info_df.columns)):
        for j in range(len(mutual_info_df.index)):
            cell = table.get_celld()[(j,i)]
            cell.set_text_props(ha='center', va='center')
            cell.set_facecolor('white')

    # Set the table title
    plt.title(f'{protein} Pairwise {metric_string} Table')

    plt.tight_layout()

    # Increase the resolution of the PNG by setting a higher DPI
    # dpi = 300 # You can adjust this value as needed
    table_name = f'{protein}_{metric_string}_table.png'
    table_name = os.path.join(table_dir, table_name)
    plt.savefig(table_name)

    # Save the DataFrame to an Excel file
    excel_name = f'{protein}_table_{metric_string}.xlsx'
    excel_name = os.path.join(table_dir, excel_name)
    formatted_df.to_excel(excel_name, index=True)

    # Reset the DPI to the default value (usually 100) for subsequent plots
    # plt.rcParams['figure.dpi'] = 100

    # plt.show()

def plot_spearman_full_table(protein, info_array, names_combined):
    """
    Plots a table of pairwise Spearman correlation and significance between features for a given protein.

    Args:
        protein (str): The name of the protein for which to plot the table.
        info_array (numpy.ndarray): A square numpy array containing the pairwise Spearman correlation and significance values as tuples.
        names_combined (list): A list of the names of the combined features used to calculate the Spearman correlation.

    Returns:
        None
    """
    table_dir = "table_pngs"
    os.makedirs(table_dir, exist_ok=True)

    # Flatten the third dimension of the info_array into a 2-dimensional array with tuples
    flattened_info_array = np.empty(info_array.shape[:2], dtype=object)
    for i in range(info_array.shape[0]):
        for j in range(info_array.shape[1]):
            flattened_info_array[i, j] = info_array[i, j]

    # Create the DataFrame without replacing -1 with "NA"
    spearman_info_df = pd.DataFrame(flattened_info_array, index=names_combined, columns=names_combined)

    # Create a boolean mask for the lower triangle of the DataFrame
    lower_triangle_mask = np.tril(np.ones(spearman_info_df.shape)).astype(bool)

    # Use the mask to keep only the lower triangle of the DataFrame and replace the upper triangle with NaN
    spearman_info_df = spearman_info_df.where(lower_triangle_mask)

    # Replace NaN values with "NA" in the DataFrame
    spearman_info_df = spearman_info_df.replace(np.nan, "NA")

    # Remove the last row and column from the DataFrame
    spearman_info_df = spearman_info_df.iloc[1:, :-1]

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.axis('tight')
    ax.axis('off')

    # Format the numbers in the DataFrame to have 2 decimal places for correlation and 3 decimal places for significance
    # Initialize an empty list to store the formatted values
    formatted_info_array = []

    # Iterate over each row in the spearman_info_df.values
    for row in spearman_info_df.values:
        # Initialize an empty list to store the formatted row
        formatted_row = []
        # Iterate over each element in the row
        for num in row:
            # Check if the element is an ndarray of shape (2,)
            if isinstance(num, np.ndarray) and num.shape == (2,):
                # Format the ndarray as a string with the correlation value rounded to 2 decimal places and the significance value rounded to 3 decimal places
                # formatted_num = f"{num[0]:.2f}, {num[1]:.3f}"
                formatted_num = f"{num[0]:.2f}"
            else:
                # If not an ndarray of shape (2,), leave the value as is
                formatted_num = num
            # Append the formatted value to the formatted row
            formatted_row.append(formatted_num)
        # Append the formatted row to the formatted_info_array
        formatted_info_array.append(formatted_row)
    
    # Create a DataFrame from the formatted_info_array
    formatted_df = pd.DataFrame(formatted_info_array, index=spearman_info_df.index, columns=spearman_info_df.columns)

    # Create a table
    table = plt.table(cellText=formatted_info_array,
                 rowLabels=spearman_info_df.index,
                 colLabels=spearman_info_df.columns,
                 cellLoc = 'center',
                 loc='center')

    # Set the font size of the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Set the table title
    plt.title(f'{protein} Pairwise Spearman Correlation and Significance Table')

    plt.tight_layout()

    # Increase the resolution of the PNG by setting a higher DPI
    # dpi = 300 # You can adjust this value as needed
    table_name = f'{protein}_table_Spearman_Correlation_Significance.png'
    table_name = os.path.join(table_dir, table_name)
    plt.savefig(table_name)#, dpi=dpi)

    # Save the DataFrame to an Excel file
    excel_name = f'{protein}_table_Spearman_Correlation_Significance.xlsx'
    excel_name = os.path.join(table_dir, excel_name)
    formatted_df.to_excel(excel_name, index=True)

def prepare_raw_supplementary_MSE_table(csv_file_path,csv_dir, output_dir, columns_to_keep = ["Study Name", "CV MSE", "Protein", "Model", "Output Metric", "Feature Set", "Test MSE", "Test_CV_MSE_Ratio", "Model Type", "Full Dataset R^2"]):
    """
    Load a CSV file into a pandas DataFrame and drop all columns except for the specified ones.

    Args:
        csv_file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The DataFrame with only the specified columns.
    """
    os.makedirs(output_dir, exist_ok=True)
    full_df_path = os.path.join(csv_dir, csv_file_path)
    
    df = pd.read_csv(full_df_path)

    # Drop all columns that are not in the list of columns to keep
    df = df[columns_to_keep]

    column_order = column_order = ["Study Name", "Protein", "Feature Set", "Output Metric", "Model", "Model Type", "CV MSE", "Test MSE", "Test_CV_MSE_Ratio", "Full Dataset R^2"]
    df = df.reindex(columns=column_order)
    
    df = df.sort_values(by='Test_CV_MSE_Ratio', ascending=True)
    df = df.rename(columns ={'Test_CV_MSE_Ratio':'Test MSE / CV MSE'})

    output_name = csv_file_path.split('.')[0] + '_raw_supplementTable.csv'
    full_output_path = os.path.join(output_dir, output_name)
    df.to_csv(full_output_path, index=False)

def prepare_final_supplementary_MSE_table(csv_file_path,csv_dir, output_dir):
    """
    This function loads a CSV file into a pandas DataFrame, drops the 'Study Name' column, and saves the resulting DataFrame as a CSV file.

    Args:
        csv_file_path (str): The path to the CSV file.
        csv_dir (str): The directory containing the CSV file.
        output_dir (str): The directory where the output CSV file should be saved.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    full_df_path = os.path.join(csv_dir, csv_file_path)
    df=pd.read_csv(full_df_path)
    # print(df.columns)

    df=df.drop('Study Name', axis=1)

    # Modify the 'Feature Set' column
    df['Feature Set'] = df['Feature Set'].apply(lambda x: ast.literal_eval(x)) # Convert string representation of list to list
    df['Feature Set'] = df['Feature Set'].apply(lambda x: [item.replace('Average', '') for item in x]) # Remove 'Average' from each item in the list
    df['Feature Set'] = df['Feature Set'].apply(lambda x: [item.replace("'", "") for item in x]) # Remove apostrophes from each item in the list
    df['Feature Set'] = df['Feature Set'].apply(lambda x: [item.replace('[', '').replace(']', '') for item in x]) # Remove brackets from each item in the list
    df['Feature Set'] = df['Feature Set'].apply(lambda x: ', '.join(x)) # Convert list back to string with commas
    print(df['Feature Set'])

    output_name = csv_file_path.split('.')[0] + '_FINAL_supplementTable.csv'
    full_output_path = os.path.join(csv_dir, output_dir, output_name)
    df.to_csv(full_output_path, index=False)
    
    excel_output_name = output_name.split('.')[0] + '.xlsx'
    # print(f'excel_output_name: {excel_output_name}')
    full_output_path = os.path.join(csv_dir, output_dir, excel_output_name)
    df.to_excel(full_output_path, index=False)









def sanitize_study_filename(filename):
    '''
    Take any filename (like a study name) and 
    shorten it + remove any unwanted problematic 
    ascii characters
    '''
    # print(f'start of sanitize function: {filename}')
    
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[<>:/\\|?*,()\[\]]', '_', filename)
    
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '')
    
    # Remove apostrophes
    sanitized = sanitized.replace("'", "")
    
    # Remove any single underscore at the end before the .db extension
    if sanitized.endswith('_'):
        sanitized = sanitized[:-1]
    
    # Limit the filename length to  255 characters
    sanitized = sanitized[:255]

    # Replace 3 or more consecutive underscores with a double underscore; this is a regular expression
    sanitized = re.sub(r'_{3,}', '__', sanitized)
    
    # print(f'at the very end of the sanitized function: {sanitized}')

    return sanitized





def initialize_connect_mysql_database(db_name):
        """
        Initialize the MySQL database by creating it if it doesn't exist.
        Connect to the MySQL database.
        """
        try:
            # Establish a connection to the MySQL server
            connection = mysql.connector.connect(
                host="localhost",
                user="root",  # MySQL username
                password="ad13K8p2VqA",  
            )

            # Create a cursor object
            cursor = connection.cursor()

            # Check if the database exists
            cursor.execute(f"SHOW DATABASES LIKE '{db_name}'")
            result = cursor.fetchone()

            # If the database does not exist, create it
            if result is None:
                cursor.execute(f"CREATE DATABASE {db_name}")
                print(f"Database '{db_name}' created successfully.")
            else:
                print(f"Database '{db_name}' already exists.")

            # Close the cursor and the connection
            cursor.close()
            connection.close()

        except Error as e:
            print(f"Error occurred: {e}")

#### Would have been ideal to implement this but abandoned since 
### I didn't save the dataset within the study object since I 
### didn't know that was even possible
# def load_dataset_from_study(study_name, db_name):
#     """
#     Load the dataset corresponding to a given study name.

#     Args:
#         study_name (str): The name of the study.
#         db_name (str): The name of the database file.

#     Returns:
#         pandas.DataFrame: The dataset corresponding to the study.
#     """
#     # Load the study from the database
#     study = optuna.load_study(storage=f'sqlite:///{db_name}.db', study_name=study_name)

#     # Check if the study has any trials
#     if len(study.trials) == 0:
#         print(f"No trials found for study '{study_name}'.")
#         return None

#     # Get the best trial
#     best_trial = study.best_trial

#     # Extract the dataset from the best trial's user attributes
#     # Assuming the dataset is stored in the 'dataset' user attribute
#     dataset = best_trial.user_attrs.get('dataset')

#     # If the dataset is not found in the user attributes, return None
#     if dataset is None:
#         print(f"Dataset not found in the user attributes of study '{study_name}'.")
#         return None

#     # Convert the dataset to a pandas DataFrame
#     dataset_df = pd.DataFrame(dataset)

#     return dataset_df