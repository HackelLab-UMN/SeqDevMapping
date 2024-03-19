import optuna

from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, r2_score, mean_squared_error

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.svm import NuSVR
from sklearn.ensemble import GradientBoostingRegressor

import pandas as pd
from itertools import chain, combinations

import time

import warnings
warnings.filterwarnings("ignore")

from inspect import signature

from architectures import*

import concurrent
from concurrent.futures import ProcessPoolExecutor
import os

import re

from utilities import *

import logging 


class ModelEvaluator:
    '''
    A class that automates model evaluation

    In: features, output_metric, feature scaler function, models, 
    and the hyperparameter grid for each of the models

    The key method of importance is evaluate_models(): 
    evaluate_models() loops over all output metrics, feature sets, 
    scalerizing input functions, and record the best
    performance and that best model's hyperparameters
    '''
    def __init__(self, protein, features, output_metrics, input_names, output_names, scalers, models, param_grids,db_name,num_trials=1000):
        self.protein = protein
        self.features = features
        self.output_metrics = output_metrics
        self.input_names = input_names
        self.output_names = output_names
        self.scalers = scalers # a list of scaler classes
        self.models = models # a list of model classes (from sklearn)
        self.param_grids = param_grids
        self.db_name = db_name
        self.final_sqlite_db_name = f'sqlite///{self.db_name}'
        
        # This is my mysql attempt: it failed because individual trials are short compared
        # to the cost/overhead of message passing
        # initialize_connect_mysql_database(db_name)
        # self.db_name = f"mysql+mysqlconnector://root:ad13K8p2VqA@localhost/{db_name}"
        
        self.num_trials = num_trials

        self.feature_sets = self.make_all_combinations(self.features)
        print(f'inside the init, self.feature_sets is length: {len(self.feature_sets)}')
        self.feature_set_names = self.make_all_name_combinations(self.input_names)
        # print(f'inside init: first feature set type={type(self.feature_sets[0])} with feature set={self.feature_sets[0]}')

        self.study_combinations = self.prepare_study_combinations()
        self.repository_path = self.make_repository_path()
    
    def format_data_general(self, output_metric, feature_set):
        '''
        Input:
        1) A single y metric DataFrame Series
        2) a feature set that is the X values

        Output:
        np arrays of the X and y features that have had NA rows/entries removed
        '''

        if isinstance(feature_set, list):
            X_df = pd.concat(feature_set, axis=1)
        elif isinstance(feature_set, pd.DataFrame):
            X_df = feature_set
        elif isinstance(feature_set, pd.Series):
            X_df = feature_set.to_frame()  # Convert the Series to a DataFrame
        else:
            raise ValueError(f'feature set is not a list, a dataframe, or a series: it is type={type(feature_set)}')

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
        return X_np, y_np
    
    def make_all_name_combinations(self,var_list):
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
    
    def make_all_combinations(self,series_list):
        '''
        Go over a list of pandas series, return a list of series
        of all combinations of pandas series
        '''
        all_combinations = []
        for r in range(1, len(series_list) +  1):
            combinations_object = combinations(series_list, r)
            all_combinations.extend(list(combo) for combo in combinations_object)
        return all_combinations
    
    def make_model_pipeline_search(self,trial,param_grid,scaler_instance,model_class):
        '''
        Summary: 
        Loop over all possible hyperparmeters for a given model, have optuna
        suggest which hyperparameters should be tried next

        Input: 
        1) trial object: this objects contains all of the hyperparameters in a current training run of an optuna study
        2) param_grid: this is a dictionary for a given model class containing all hyperparameters of interest
        3) scaler: this is a function that normalizes the X/y data
        4) model_class: this is the class of the model we wish to train/inspect

        Output: 
        1) an sklearn pipeline that contains an input scaling and optuna-hyperparameter suggested model
        '''
        # perform model initialization
        # model_params={'random_state':42}  # for reproducibility}
        model_params={}
        for param_name, param_range in param_grid.items():
            if param_range['type'] == 'categorical':
                model_params[param_name] = trial.suggest_categorical(param_name, param_range['values'])
            elif param_range['type'] == 'int':
                model_params[param_name] = trial.suggest_int(param_name, *param_range['range'])
            elif param_range['type'] == 'float':
                model_params[param_name] = trial.suggest_float(param_name, *param_range['range'])

        if 'random_state' in signature(model_class).parameters:
            model_params['random_state'] =  42  # for reproducibility

        pipeline = Pipeline([
        ('scaler', scaler_instance),  # Replace with your preferred scaler
        ('model', model_class(**model_params))
        ])
        
        return pipeline


    def objective_Kfold(self, trial, model_class, X_trainCV, y_trainCV, param_grid, scaler_class):
        '''
        Perform K-fold cross validation. Within a given fold, define a new model_pipeline
        including hyperparameters and input scalings that is trained
        with a unique combination of hyperparameters contained inside 'trial'

        Return the R^2 of the best hyperparameter combination for the model and input scaler

        We use this function instead of standard library methods like sklearn's cross_val_score()
        since we want to take the weighted average of the cross validation score with respect to the number
        of datapoints in a given validation set. cross_val_score() et al appear to assume that
        each CV fold has the same amount of data, which is problematic/incorrect for smaller datasets but
        acceptable for larger datasets.
        '''

        # perform k fold cross validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        weighted_scores_sum_r2 =  0
        weighted_scores_sum_mse =  0
        total_samples =  0
        for train_index, val_index in kf.split(X_trainCV):
            X_train_split, X_val_split = X_trainCV[train_index], X_trainCV[val_index]
            y_train_split, y_val_split = y_trainCV[train_index], y_trainCV[val_index]

            # Create a new instance of the scaler for each fold
            scaler_instance = scaler_class()
            
            # initialize the model
            pipeline_instance = self.make_model_pipeline_search(trial, param_grid, scaler_instance, model_class)

            # print(f"Scaler instance before fitting: {id(pipeline_instance.named_steps['scaler'])}")

            # Train your model with the current hyperparameters
            pipeline_instance.fit(X_train_split, y_train_split)

            # Access the fitted scaler from the pipeline's steps
            fitted_scaler = pipeline_instance.named_steps['scaler']

            # Print the memory address of the fitted scaler
            # print(f"Fitted scaler instance: {id(fitted_scaler)}")

            # Evaluate the model on the validation set
            y_pred = pipeline_instance.predict(X_val_split)
            r2_score_val = r2_score(y_val_split, y_pred)
            mse_score_val = mean_squared_error(y_val_split, y_pred)

            # Calculate the weighted score
            fold_samples = len(y_val_split)
            weighted_scores_sum_r2 += r2_score_val * fold_samples
            weighted_scores_sum_mse += mse_score_val * fold_samples
            total_samples += fold_samples

        # Return the weighted mean score
        weighted_mean_score_r2 = weighted_scores_sum_r2 / total_samples
        weighted_mean_score_mse = weighted_scores_sum_mse / total_samples

        return weighted_mean_score_r2, weighted_mean_score_mse

        ### If we had a large dataset size then I could simply assume all folds are equal and I could do the 3 line command below:
        # model_instance = self.make_model(trial,pipe)
        # cv_scores = cross_val_score(model, X_trainCV, y_trainCV, cv=5, scoring='r2')
        # weighted_mean_score = np.mean(cv_scores)

    def prepare_study_combinations(self):
        """
        Prepare the arguments for each evaluation.

        :return: A list of tuples containing the arguments for each evaluation.
        :rtype: list
        """

        # Prepare the arguments for each evaluation
        study_combinations = []
        for output_metric, output_name in zip(self.output_metrics, self.output_names):
            for feature_set, feature_set_name in zip(self.feature_sets, self.feature_set_names):
                for model, param_grid in zip(self.models, self.param_grids):
                    for scaler_class in self.scalers:
                        # Generate a unique database name for this combination with make_study_name()
                        study_name = self.make_study_name((feature_set, feature_set_name, output_metric, output_name, model, param_grid, scaler_class, self.db_name))
                        
                        # Prepare the arguments for this combination
                        study_combinations.append((feature_set, feature_set_name, output_metric, output_name, model, param_grid, scaler_class, self.db_name))
        
        return study_combinations

    def make_study_name(self, single_study_combination):
        """
        Create a unique name for a study based on the provided combination of arguments.

        Args:
            single_study_combination (tuple): A tuple containing the arguments for a single study.

        Returns:
            str: The unique name for the study.
        """
        
        feature_set, feature_set_name, output_metric, output_name, model, param_grid, scaler_class, db_name = single_study_combination
        
        # Create an instance of the model class, get the name, remove underscores
        model_instance = model()
        model_class_name = type(model_instance).__name__
        model_class_name = model_class_name.replace('_', '')

        # Use the model class name in the study name
        scaler_class_name = str(scaler_class).split('.')[-1]
        raw_study_name = f"study__{self.protein}__{scaler_class_name}__{str(model_class_name)}__{output_name}__{feature_set_name}"
        study_name = sanitize_study_filename(raw_study_name)
        
        # print(f'in make_study_name(), study_name = {study_name}')
        return study_name
    
    def run_study(self,single_study_combination):
        """
        Perform a hyperparameter optimization study for a single study combination.

        Args:
            single_study_combination (tuple): A tuple containing the arguments for a single study.

        Returns:
            None
        """
        print(f'inside run_study()')

        feature_set, feature_set_name, output_metric, output_name, model, param_grid, scaler_class, db_name = single_study_combination
        
        print(f'after single study combination assignment')
        # # Log the start of the study
        # logging.info(f"Starting study for {feature_set_name} with {model.__name__}")

        # Format the data
        X, y = self.format_data_general(output_metric, feature_set)
        # print(f'finished format data general')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # print(f'finished test train split')
        

        study_name = self.make_study_name(single_study_combination)
        # try: 
        #     study_name = self.make_study_name(single_study_combination)
        # except Exception as e:
        #     print(f'Exception occured: {e}')

        # print(f'inside run study: study_name = {study_name}')
        # print(f'inside run study: db_name = {self.db_name}')
        single_study_db_name = f"sqlite:///{os.path.join(self.db_name, study_name)}.db"
        
        print(f'inside run study: single study db name = {single_study_db_name}')

        study = optuna.create_study(storage=single_study_db_name, study_name=study_name, direction="minimize", load_if_exists=True)
  
        # try: 
        #     study = optuna.create_study(storage=single_study_db_name, study_name=study_name, direction="minimize", load_if_exists=True)
        # except Exception as e:
        #     print(f'Exception occured: {e}')

        # define an objective with just trial as an input to satisfy study.optimize()'s input format
        def objective(trial): 
            return self.objective_Kfold(trial, model, X_train, y_train, param_grid, scaler_class)[1] #[0] will optimize for r2, [1] will optimize for mse
        
        # enforce the same number of trials per combination study
        if len(study.trials) <  self.num_trials:
            study.optimize(objective, n_trials=self.num_trials-len(study.trials) , n_jobs = 1)
        else: 
            print(f"Study '{study_name}' already has {len(study.trials)} trials.")   
        
        # # Log the end of the study
        # logging.info(f"Finished study for {feature_set_name} with {model.__name__}")
    
    def evaluate_models_parallel(self):
        """
        Runs the hyperparameter optimization for each study combination in parallel using the ProcessPoolExecutor.

        This function is called by the inspect_db_and_report_models function.
        """
        # Ensure the directory exists before creating the combined database file
        os.makedirs(self.repository_path, exist_ok=True)
        
        # The default number of worker processes is the number of CPU cores
        default_num_workers = os.cpu_count() # For production use only
        # default_num_workers = 1 # For testing
        print(f"Default number of worker processes: {default_num_workers}")

        # Run the optimization in parallel for each study combination
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self.run_study, self.study_combinations)

    def make_repository_path(self):
        """
        Construct the path to the directory containing the single study databases.

        Returns:
            str: The full path to the repository directory.
        """
        # Construct the path to the directory containing the single study databases
        repository_path = os.path.join(os.getcwd(), self.db_name)
        return repository_path

    def make_combined_study_database(self):
        """
        Combine all single study databases into a single database containing all the studies.

        Args:
            self.repository_path (str): The path to the directory containing the single study databases.
            self.db_name (str): The name of the combined database file to be created.

        Returns:
            None
        """

        # Construct the full path to the combined database file
        combined_db_path = os.path.join(os.getcwd(), f'{self.db_name}.db')
        combined_db_path = f'sqlite:///{combined_db_path}'

        # Iterate over all the single study databases in the repository
        for filename in os.listdir(self.repository_path):
            if filename.endswith('.db'):
                # Construct the full path to the single study database file
                single_study_db_name = os.path.join(self.repository_path, filename)

                # Extract the study name from the filename
                study_name = os.path.splitext(filename)[0]


                # Load the single study
                single_study_db_name = f'sqlite:///{single_study_db_name}'
                single_study = optuna.load_study(storage=single_study_db_name, study_name=study_name)


                # Create a new study in the combined database with the same name and direction as the single study
                combined_study_name = single_study.study_name
                combined_study = optuna.create_study(storage=combined_db_path, study_name=combined_study_name, direction=single_study.direction, load_if_exists=True)


                # Add the trials from the single study to the corresponding new study in the combined database
                for trial in single_study.trials:
                    # Create a new trial with the same parameters and values as the single study trial
                    combined_study.add_trial(
                        optuna.trial.create_trial(
                            state=trial.state,
                            params=trial.params,
                            distributions=trial.distributions,
                            user_attrs=trial.user_attrs,
                            system_attrs=trial.system_attrs,
                            value=trial.value,
                            # datetime_start=trial.datetime_start,
                            # datetime_complete=trial.datetime_complete,
                            # duration=trial.duration
                        )
                    )

        print(f"Combined database created with {len(combined_study.trials)} trials.")

    

    def get_best_trial_and_study_dataframe(self, db_name, study_name):
        '''
        Load a single study. 

        Find that study's best trial, the trial attributes, 
        and the study statistics. 

        Return a DataFrame row that includes all this information
        '''
        
        # Load the study from the database
        print(f"Loading study with name: {study_name}")
        print(f"Database connection string: {db_name}")
        study = optuna.load_study(storage=f'sqlite:///{db_name}.db', study_name=study_name)

        # # Check if the study has any trials
        # if len(study.trials) == 0:
        #     print(f"No trials found for study '{study_name}'.")
        #     # Handle the case where there are no trials, e.g., return a default value or raise an exception
        #     return None, None

        # Get the best trial
        try: 
            best_trial = study.best_trial
        except ValueError as e: 
            if str(e) == "Record does not exist.":
                print(f"No trials found for study '{study_name}'.")
                # Handle the case where there are no trials, e.g., return a default value or raise an exception
                return None, None
            else:
                raise # Re-raise the exception if it's not the expected one

        # Create a DataFrame row with the best trial's metrics, values, and attributes
        best_trial_data = {
            'Trial Number': best_trial.number,
            # 'Trial Value': best_trial.value, # this is the same as the Best Trial Value = CV R^2
            'Trial Start Time': best_trial.datetime_start,
            'Trial Duration': best_trial.duration,
            'Trial Parameters': best_trial.params,
            'Trial User Attributes': best_trial.user_attrs,
            'Trial System Attributes': best_trial.system_attrs,
            'Trial State': best_trial.state
        }

        # Create a DataFrame row with the study's attributes, values, and statistics
        study_data = {
            'Study Name': study.study_name,
            'Study Direction': study.direction,
            'Study User Attributes': study.user_attrs,
            'Study System Attributes': study.system_attrs,
            'Number of Trials': len(study.trials_dataframe()),
            'CV R^2': study.best_trial.value,
            'Best Trial Number': study.best_trial.number
        }

        # Combine the best trial data and study data into a single DataFrame
        combined_data = {**best_trial_data, **study_data} # unpack and merge the two dictionaries
        combined_df = pd.DataFrame([combined_data])

        return combined_df, best_trial

    def inspect_db_and_report_models(self):
        '''
        Load an already-existing database (.db) file that was created
        by self.evaluate_models(); load each study, report the 
        aggregate and best individual trial performance per study.

        Save all of this information as a "raw" csv. Each row of the csv
        will be the best and aggregate trial performances in a single study.
        '''
        print(f'inside inspect: self.feature_sets length is: {len(self.feature_sets)} \n')
        results_list = []
        
        for output_metric, output_name in zip(self.output_metrics, self.output_names):
            print(f'self.features = {self.features} \n')
            print(f'self.features length = {len(self.features)}\n')
            print(f'self.feature_set_names = {self.feature_set_names}\n')
            print(f'self.feature_set_names length = {len(self.feature_set_names)}\n')
            for feature_set, feature_set_name in zip(self.feature_sets, self.feature_set_names):
                print(f'feature set name ={feature_set_name}')
                print(f'output_metric name= {output_name}')
                X, y = self.format_data_general(output_metric, feature_set)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                for model, param_grid in zip(self.models, self.param_grids):
                    for scaler_class in self.scalers:
                        
                        # Extract the model class name to save in the final pandas df
                        model_instance = model() # Create an instance of the model class
                        model_class_name = type(model_instance).__name__ # Get the name of the model class from the instance
                        model_class_name = model_class_name.replace('_', '') # Remove underscores from the model class name if there are
                        
                        # Get the study name
                        study_name = self.make_study_name((feature_set, feature_set_name, output_metric, output_name, model, param_grid, scaler_class, self.db_name))
                        print(f"Checking study with name: {study_name}")
                        
                        # Check if the study exists in the database
                        try:
                            study = optuna.load_study(storage=f'sqlite:///{self.db_name}.db', study_name=study_name)
                            print(f"Study '{study_name}' exists in the database.")
                        except optuna.exceptions.DuplicatedStudyError:
                            print(f"Study '{study_name}' already exists in the database.")
                        except optuna.exceptions.StorageInternalError:
                            print(f"Study '{study_name}' does not exist in the database.")
                        except Exception as e:
                            print(f"An error occurred while checking study '{study_name}': {e}")



                        trial_study_comb_df, best_trial = self.get_best_trial_and_study_dataframe(self.db_name,study_name)
                        if trial_study_comb_df is None or best_trial is None:
                            print(f"No trials found for study '{study_name}', skipping.")
                            continue # Skip this iteration and move to the next one
                        
                        # Instantiate a model with the best trial hyperparameters
                        best_model = model(**best_trial.params)

                        # Instantiate the scaler instance from the scaler class
                        scaler_instance=scaler_class()

                        # Instantiate the best pipeline (scaler + model)
                        best_pipeline =   Pipeline([
                            ('scaler', scaler_instance),  # Replace with your preferred scaler
                            ('model', best_model)
                        ])
                        
                        # Fit the model on the training set
                        best_pipeline.fit(X_train, y_train)
                        
                        # Predict on the test set
                        y_pred_test = best_pipeline.predict(X_test)
                        
                        # Calculate the test MSE and R^2
                        test_mse = mean_squared_error(y_test, y_pred_test)
                        test_r2 = r2_score(y_test, y_pred_test)
                        
                        # Retrain on the entire dataset
                        best_pipeline.fit(X, y)
                        
                        # Predict on the entire dataset
                        y_pred_full = best_pipeline.predict(X)
                        
                        # Calculate the entire dataset MSE and R^2
                        full_mse = mean_squared_error(y, y_pred_full)
                        full_r2 = r2_score(y, y_pred_full)

                        # Create a DataFrame row with the metrics
                        metrics_data = {
                        'Protein': self.protein,
                        'Scaler': str(scaler_class),
                        'Model': model_class_name,
                        'Output Metric': output_name,
                        'Feature Set': feature_set_name,
                        'Test MSE': test_mse,
                        'Full Dataset MSE': full_mse,
                        'Full Dataset R^2': full_r2
                    }
                        # Combine the metrics data with the trial and study data
                        metrics_df = pd.DataFrame([metrics_data])
                        combined_row = pd.merge(trial_study_comb_df, metrics_df, left_index=True, right_index=True)
                        results_list.append(combined_row)
        
        # Convert the list of DataFrames into a single DataFrame
        results_df = pd.concat(results_list, ignore_index=True)
        
        # Save the DataFrame to a CSV file
        results_df.to_csv(f'{self.protein}_raw_model_evaluation_results.csv', index=False)
