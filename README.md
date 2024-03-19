Sequence-developability mapping of affibody and fibronectin paratopes via library-scale variant characterization, Authors: Gregory Nielsen and Zachary Schmitz

Contact information: niels786@umn.edu

Hackel Lab, University of Minnesota

Overview: 
This project contains Python3 and bash scripts used to examine Fibronectin and Affibody library-scale assay measurements with gold standard output metrics.

To run our analytical pipeline, first unzip datasets located at ./datasets/. Then run run.py to train the model study combinations. Tools for analyzing the model study combinations and generating figures shown in our manuscript are in utilities.py; examples of use cases are in NonlinVsLin.py. Required packages are included in the conda_package_list.txt. See the quick start below for a more detailed introduction on using this project.

All Optuna studies containing hyperparameter optimization trials for each model can be found within ./studies/{protein}_studies_multithreaded.db.

File descriptions: 
1)	ModelEvaluator.py &#8594; defines a ModelEvaluator class that automates sklearn model (i.e., RandomForestRegression) evaluation. All possible combinations of outputs, feature combinations, models, and feature scalings are examined via grid search: we call one {output, feature combination, model, feature scaling} a “combination study”. The dataset for this combination study comprises all rows from the respective protein’s (i.e., Aby or Fn) raw csv that are populated for the features and output metric of interest. 

Then, a call to ModelEvaluator.evaluate_models() causes 1000 Optuna trials to be explored for the sklearn model’s hyperparameter optimization. Each trial constitutes a unique assignment for each model hyperparameter guided by Optuna’s default Bayesian optimization algorithm: tree-structured parzen estimator (TPE). Within a given trial, we perform k=5-fold cross validation on a consistent training subset of the study combination dataset: this training subset is 80% of the entire study combination dataset. These trials are stored as a combination study object within a sqlite database.

After completion of cross validation, ModelEvaluator.inspect_db_and_report_models() finds the best performing trial for a given study combination. This best model is retrained on the entire study combination dataset and the resulting performance is reported. Additional metadata over all trials for a given model is also recorded.

This process is then repeated for all possible study combinations. The study combination cross-validation metadata and best-trial performances are reported in a raw csv file. This csv file has each row constitute a single study-combination. For example, if 4 models, 3 features, 1 output metric, and 3 feature scalers are examined for protein Aby, there will be 4 [models] * sum(3choose1 + 3choose2 + 3choose3) [possible combinations of input features] * 1 [output metric] * 3 [feature scalers] = 4 * 7 * 1 * 3 = 84 study combination rows in the output csv.

2)	utilities.py &#8594; defines generally useful functions for ModelEvaluator.py and later processing scripts such as sanitization of study names and plotting.

3)	NonlinVsLin.py &#8594; handles various plotting routines to generate figures shown in our manuscript. Uncomment the relevant figure block you wish to generate.

4)	raw_data.py &#8594; defines input dataset loading regime and features to load into the model.

5)	architectures.py &#8594; handles definition of all inputs needed to initialize a ModelEvaluator object. This includes raw input data for a given protein of interest (e.g., Developability_scores_aby.csv), all sklearn models, and all sklearn hyperparameters with respective ranges.

Folder descriptions:
1)	./datasets/ &#8594; location of saved fibronectin and affibody library scale and gold-standard metric scores. You will need to unzip the datasets before running.
2)	./studies/ &#8594; location of sqlite databases of Oputna studies containing trials for each model examined during cross validation


Quickstart: 
1)	Create a conda environment and install core libraries including scikit-learn, optuna, and pandas. Either load the conda environment via pasting “conda install –file conda_package_list.txt” in terminal or copy and paste the following command into terminal: “conda create --name Dev2024 -c conda-forge optuna scikit-learn pandas”
2)	Modify architectures.py to define any additional models, input/output features, hyperparameters, etc. you want to examine
3)	Go to run.py. Initialize an instance of ModelEvaluator: modify any inputs passed as input to the evaluator = ModelEvaluator(…) call.
4)	Run ModelEvaluator.py via “python run.py”. You should see trials being initialized and examined by Optuna. Additionally, a series of .db sqlite database files should be created in a new “{protein}_studies_multithreaded” folder in the current directory for each model study combination that is generated.
5)	Once cross validation is completed for each study combination and the Sqlite .db studies are merged, call evaluator.inspect_db_and_report_models() to create a csv to summarize the best study combination for a given set of inputs on a single output of interest.
6)	Uncomment and run relevant sections of NonlinVsLin.py to examine and plot model study performances.

