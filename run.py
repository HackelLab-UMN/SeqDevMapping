import sys
from ModelEvaluator import*
import os

def create_model_evaluator(protein):
    """
    Create  a ModelEvaluator object for the specified protein.

    Parameters
    ----------
    protein : str
        The name of the protein for which to create and evaluate the ModelEvaluator object.
        Supported proteins are "Aby" and "Fn".

    Returns
    -------
    An instance of the ModelEvaluator object.

    """
    
    scalers = [
        standard_scaler, 
        min_max_scaler, 
        robust_scaler
        ]
    
    models = [
        linear_regression, #linear models
        lasso, 
        ridge, 
        elastic_net, 
        random_forest, #nonlinear models
        svr, 
        mlp, 
        decision_tree, 
        ada_boost, 
        knn, 
        nu_svr, 
        gradient_boosting
        ]
    
    param_grids = [
        linear_regression_param_grid, 
        lasso_param_grid, 
        ridge_param_grid, 
        elastic_net_param_grid, 
        rf_param_grid, 
        svr_param_grid, 
        mlp_param_grid, 
        dt_param_grid, 
        ada_boost_param_grid, 
        knn_param_grid, 
        nu_svr_param_grid, 
        gradient_boosting_param_grid
        ]
    
    num_trials = 100

    if protein == "Aby":
        # Define parameters specific to protein "Aby"
        features = Aby_input_metrics
        output_metrics = Aby_output_metrics
        input_names = Aby_input_names
        output_names = Aby_output_names
        scalers = scalers.copy()
        models = models.copy()
        param_grids = param_grids.copy()
        db_name = 'Aby_studies_multithreaded'
        num_trials =  num_trials
    
    elif protein == "Fn":
        # Define parameters specific to protein "Fn"
        features = Fn_input_metrics
        output_metrics = Fn_output_metrics
        input_names = Fn_input_names
        output_names = Fn_output_names
        scalers = scalers.copy()
        models = models.copy()
        param_grids = param_grids.copy()
        db_name = 'Fn_studies_multithreaded'
        num_trials =  num_trials
    else:
        raise ValueError(f"Protein '{protein}' is not supported.")

    # Create and evaluate the ModelEvaluator object
    evaluator = ModelEvaluator(
        protein=protein,
        features=features,
        output_metrics=output_metrics,
        input_names=input_names,
        output_names=output_names,
        scalers=scalers,
        models=models,
        param_grids=param_grids,
        db_name=db_name,
        num_trials=num_trials
    )

    return evaluator

    # print(f'Starting evaluation of models for protein {protein}:')
    # # evaluator.evaluate_models()
    # evaluator.evaluate_models_parallel()
    # print(f'Finished evaluation of models for protein {protein}\n')

    # print(f'Starting inspection/reporting of models for protein {protein}:')
    # evaluator.inspect_db_and_report_models()
    # print(f'Finished inspection/reporting of models for protein {protein}\n')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run.py <protein>")
        print("Supported proteins are 'Aby' and 'Fn'.")
        sys.exit(1)

    protein = sys.argv[1]
    # protein = "Aby"  # Change this to "Fn" or any other protein as needed
    # protein = "Fn"

    print(f'Starting evaluation of models for protein {protein}:')
    
    start=time.time()
    evaluator = create_model_evaluator(protein)
    # evaluator.evaluate_models_parallel()
    evaluator.make_combined_study_database()
    evaluator.inspect_db_and_report_models()
    end = time.time()
    print(f'Time taken to create and evaluate models for protein {protein} is {end-start} seconds')