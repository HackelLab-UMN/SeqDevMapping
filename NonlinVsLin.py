# Import the input and output metrics from raw_data.py
from raw_data import*
from utilities import*
plt.rcParams['figure.dpi'] = 1000

linear_model_list = ['LinearRegression', 'Lasso', 'Ridge', 'ElasticNet']
nonlinear_model_list = ['RandomForestRegressor', 'SVR','MLPRegressor','DecisionTreeRegressor','AdaBoostRegressor','KNeighborsRegressor','NuSVR','GradientBoostingRegressor']

Aby_results_filename = 'Aby_raw_model_evaluation_results.csv'
Fn_results_filename = 'Fn_raw_model_evaluation_results.csv'

Aby_best_rows_df = examine_all_combinations(Aby_results_filename,'Aby', Aby_output_names, Aby_input_names, linear_model_list, nonlinear_model_list, 'Full Dataset R^2', pd.Series.idxmax)
Fn_best_rows_df = examine_all_combinations(Fn_results_filename,'Fn', Fn_output_names, Fn_input_names, linear_model_list, nonlinear_model_list, 'Full Dataset R^2', pd.Series.idxmax)

Aby_df_name = 'Aby_best_combinations__DotBlotAvg_ug_mL__CDTm_degC__TestMSE.csv'
Fn_df_name = 'Fn_best_combinations__SDSPAGE_mg_mL__CDTm_degC__TestMSE.csv'

### Scatterplot function calls for best models, best feature sets:
# Testing/use of utilties functions for loading/using individual models/datasets:
study1 = 'study__Aby__StandardScaler__LinearRegression__CDTm_degC__AverageTL_AveragesGFP'
study2 = 'study__Aby__MinMaxScaler__AdaBoostRegressor__DotBlotAvg_ug_mL__AveragesGFP_NSBAverage'
study3 = 'study__Fn__MinMaxScaler__NuSVR__CDTm_degC__AverageTL'
Aby_db_name = 'Aby_studies_multithreaded'
Fn_db_name = 'Fn_studies_multithreaded'

# Scatterplot parent functions:
# best_Aby_CDTm_degC_model, _, _ = load_best_model_from_study_name(study1, Aby_db_name)
# best_Aby_DotBlot_model, _, _ = load_best_model_from_study_name(study2, Aby_db_name)
# best_Fn_CDTm_degC_model, _, _ = load_best_model_from_study_name(study3, Fn_db_name)
# X_np_study1, y_np_nstudy1, _, _ = load_dataset_from_study_name(Aby_df, study1)
# X_np_study2, y_np_study2, _, _ = load_dataset_from_study_name(Aby_df, study2)
# X_np_study3, y_np_study3, _, _ = load_dataset_from_study_name(Fn_df, study3)

# Making/saving general scatterplots (abandoned):
# make_scatterplot_from_model_and_dataset(Aby_df, study1, Aby_db_name)
# make_scatterplot_from_model_and_dataset(Aby_df, study2, Aby_db_name)
# make_scatterplot_from_model_and_dataset(Fn_df, study3, Fn_db_name)




# ### START OF ABY SCATTERPLOT ###
# def make_scatterplots():
    
    best_pipeline1, model_name1, scaler_name1 = load_best_model_from_study_name(study1, Aby_db_name)
    X1, y1, input_names1, output_name1 = load_dataset_from_study_name(Aby_df, study1) 
    output_name1 = output_name1.replace('Avg', '')
    input_names1 = [name.replace('Average', '') for name in input_names1]
    best_pipeline1.fit(X1,y1)
    y_pred1 = best_pipeline1.predict(X1)


    best_pipeline2, model_name2, scaler_name2 = load_best_model_from_study_name(study2, Aby_db_name)
    X2, y2, input_names2, output_name2 = load_dataset_from_study_name(Aby_df, study2) 
    output_name2 = output_name2.replace('Avg', '')
    input_names2 = [name.replace('Average', '') for name in input_names2]
    best_pipeline2.fit(X2,y2)
    y_pred2 = best_pipeline2.predict(X2)

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(ncols=2, figsize=(10, 10))

    # Plot for study1
    ax1 = axes[0]
    sns.scatterplot(x=y1, y=y_pred1, color='black', ax=ax1)
    ax1.set_xlabel('Actual ' + output_name1)
    ax1.set_ylabel('Predicted ' + output_name1)
    ax1.set_title(f'{output_name1} \n using {model_name1}')
    # ax1.axis('equal')

    # Set the limits of the y and x axes to the same range
    ax1.set_xlim(25,95)
    ax1.set_ylim(25,95)

    # Set the tick locations to the specified values
    ax1.set_xticks([30, 40, 50, 60, 70, 80, 90])
    ax1.set_yticks([30, 40, 50, 60, 70, 80, 90])

    # Calculate the correlation between y_pred1 and y1
    correlation1 = np.corrcoef(y1, y_pred1)[0, 1]

    # Add the number of data points and the correlation as labels within the box
    num_data_points1 = len(y1)
    ax1.text(0.05, 0.95, f'N={num_data_points1}, ρ={correlation1:.2f}', transform=ax1.transAxes, fontsize=12, verticalalignment='top')

    # Plot for study2
    ax2 = axes[1]
    sns.scatterplot(x=y2, y=y_pred2, color='black', ax=ax2)
    ax2.set_xlabel('Actual ' + output_name2)
    ax2.set_ylabel('Predicted ' + output_name2)
    ax2.set_title(f'{output_name2} \n using {model_name2}')
    # ax2.axis('equal')

    # Set the limits of the y and x axes to the same range
    ax2.set_xlim(-100,1450)
    ax2.set_ylim(-100,1450)

    # Set the tick locations to the specified values
    ax2.set_xticks([0, 200, 400, 600, 800, 1000, 1200, 1400])
    ax2.set_yticks([0, 200, 400, 600, 800, 1000, 1200, 1400])

    # Calculate the correlation between y_pred2 and y2
    correlation2 = np.corrcoef(y2, y_pred2)[0, 1]

    # Add the number of data points and the correlation as labels within the box
    num_data_points2 = len(y2)
    ax2.text(0.05, 0.95, f'N={num_data_points2}, ρ={correlation2:.2f}', transform=ax2.transAxes, fontsize=12, verticalalignment='top')

    # Adjust the layout
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    plt.tight_layout()

    # Create a directory to save the plot
    plot_dir = "scatterplot_pngs"
    os.makedirs(plot_dir, exist_ok=True)

    # Save the figure
    plot_name = f'Aby_scatterplot.png'
    plot_name = sanitize_study_filename(plot_name)
    full_output_path = os.path.join(plot_dir, plot_name)
    plt.savefig(full_output_path)

    # Beginning of Excel Formatting
    # Create separate DataFrames for each dataset
    df_dataset1 = pd.DataFrame({
        'Actual': y1,
        'Predicted': y_pred1
    })

    df_dataset2 = pd.DataFrame({
        'Actual': y2,
        'Predicted': y_pred2
    })

    # Define the directory path
    directory_path = "scatterplot_pngs"

    # Ensure the directory exists
    os.makedirs(directory_path, exist_ok=True)

    # Construct the full path to the Excel file
    full_path = os.path.join(directory_path, 'Aby_scatter.xlsx')

    # Use ExcelWriter to save the DataFrames to separate sheets within the same Excel file in the specified directory
    # print(f'outputname2 = {output_name2}')
    with pd.ExcelWriter(full_path) as writer:
        df_dataset1.to_excel(writer, sheet_name=output_name1, index=False)
        df_dataset2.to_excel(writer, sheet_name=' '.join(output_name2.split()[:2]), index=False) # have to remove the slash
    # ### END OF ABY SCATTERPLOT ###

    ### START of the FN SCATTERPLOT ###
    # Load the best model and dataset for study3
    best_pipeline3, model_name3, scaler_name3 = load_best_model_from_study_name(study3, Fn_db_name)
    X3, y3, input_names3, output_name3 = load_dataset_from_study_name(Fn_df, study3)
    input_names3 = [name.replace('Average', '') for name in input_names3]

    # Fit the model to the dataset
    best_pipeline3.fit(X3, y3)

    # Predict the output using the fitted model
    y_pred3 = best_pipeline3.predict(X3)

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the scatterplot
    sns.scatterplot(x=y3, y=y_pred3, color='black', ax=ax)
    ax.set_xlabel('Actual ' + output_name3)
    ax.set_ylabel('Predicted ' + output_name3)
    ax.set_title(f'{output_name3} \n using {model_name3}')

    # Set the limits of the y and x axes to the same range
    # Adjust these values based on the range of your data
    ax.set_xlim(25, 85)
    ax.set_ylim(25, 85)

    # Set the tick locations to the specified values
    # Adjust these values based on the range of your data
    ax.set_xticks([30, 40, 50, 60, 70, 80])
    ax.set_yticks([30, 40, 50, 60, 70, 80])

    # Calculate the correlation between y_pred3 and y3
    correlation3 = np.corrcoef(y3, y_pred3)[0, 1]

    # Add the number of data points and the correlation as labels within the box
    num_data_points3 = len(y3)
    ax.text(0.05, 0.95, f'N={num_data_points3}, ρ={correlation3:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    # Adjust the layout
    ax.set_aspect('equal')
    plt.tight_layout()

    # Create a directory to save the plot
    plot_dir = "scatterplot_pngs"
    os.makedirs(plot_dir, exist_ok=True)

    # Save the figure
    plot_name = f'Fn_scatterplot.png'
    plot_name = sanitize_study_filename(plot_name)
    full_output_path = os.path.join(plot_dir, plot_name)
    plt.savefig(full_output_path)

    # Create separate DataFrames for each dataset
    df_dataset3 = pd.DataFrame({
        'Actual': y3,
        'Predicted': y_pred3
    })

    # Define the directory path
    directory_path = "scatterplot_pngs"

    # Ensure the directory exists
    os.makedirs(directory_path, exist_ok=True)

    # Construct the full path to the Excel file
    full_path = os.path.join(directory_path, 'Fn_scatter.xlsx')

    # Use ExcelWriter to save the DataFrames to separate sheets within the same Excel file in the specified directory
    # print(f'outputname2 = {output_name2}')
    with pd.ExcelWriter(full_path) as writer:
        df_dataset3.to_excel(writer, sheet_name=output_name3, index=False)

### END of the FN SCATTERPLOT ###
# make_scatterplots()


# ### Affibody supplementary raw MSE table:
# csv_dir = 'collapsed_combination_csvs'
# supp_csv_dir = 'supp_MSE_tables'
# output_dir = 'supp_MSE_tables'
# final_output_dir = 'final_MSE_tables'
# prepare_raw_supplementary_MSE_table(Aby_df_name, csv_dir, output_dir)
# Aby_supp_raw_name = 'Aby_best_combinations__DotBlotAvg_ug_mL__CDTm_degC__TestMSE_raw_supplementTable.csv'
# prepare_final_supplementary_MSE_table(Aby_supp_raw_name, supp_csv_dir, final_output_dir) 
# ### Fibronectin supplementary MSE table:
# prepare_raw_supplementary_MSE_table(Fn_df_name, csv_dir, output_dir)
# Fn_supp_raw_name = r'Fn_best_combinations__SDSPAGE_mg_mL__CDTm_degC__TestMSE_raw_supplementTable.csv'
# prepare_final_supplementary_MSE_table(Fn_supp_raw_name, supp_csv_dir,final_output_dir)


# # ### Affibody Barcharts R^2:
# make_fullR2_barchart(Aby_best_rows_df, 'Aby', Aby_output_names[0], 'Aby_reporting_barchart')
# make_fullR2_barchart(Aby_best_rows_df, 'Aby', Aby_output_names[1], 'Aby_reporting_barchart')
# # ### Fibronectin Barcharts R^2:
# make_fullR2_barchart(Fn_best_rows_df, 'Fn', Fn_output_names[0], 'Fn_reporting_barchart')
# make_fullR2_barchart(Fn_best_rows_df, 'Fn', Fn_output_names[1], 'Fn_reporting_barchart')

# ### Refined Affibody Barcharts R^2 :
# Aby_df = Aby_best_rows_df[Aby_best_rows_df['Test_CV_MSE_Ratio'] <= 1.6]
# make_fullR2_barchart(Aby_df, 'Aby', Aby_output_names[0], 'Aby_reporting_barchart_refined')
# make_fullR2_barchart(Aby_df, 'Aby', Aby_output_names[1], 'Aby_reporting_barchart_refined')
# ### Refined Fibronectin Barcharts R^2 :
# Fn_df = Fn_best_rows_df[Fn_best_rows_df['Test_CV_MSE_Ratio'] <= 1.6]
# make_fullR2_barchart(Fn_df, 'Fn',Fn_output_names[0], 'Fn_reporting_barchart_refined')
# make_fullR2_barchart(Fn_df, 'Fn',Fn_output_names[1], 'Fn_reporting_barchart_refined')


# ### Affibody Barcharts test MSE (descending by num_features):
# make_mse_barchart(Aby_best_rows_df, 'Aby',Aby_output_names[0],'Aby_reporting_barchart')
# make_mse_barchart(Aby_best_rows_df, 'Aby',Aby_output_names[1],'Aby_reporting_barchart')
# ### Fibronectin Barcharts test MSE (descending by num_features):
# make_mse_barchart(Fn_best_rows_df, 'Fn',Fn_output_names[0],'Fn_reporting_barchart')
# make_mse_barchart(Fn_best_rows_df, 'Fn',Fn_output_names[1],'Fn_reporting_barchart')

# ### Affibody Barcharts test MSE (descending by TEST MSE):
# make_mse_barchart(Aby_best_rows_df, 'Aby',Aby_output_names[0],'Aby_reporting_barchart', sort_by='test_score')
# make_mse_barchart(Aby_best_rows_df, 'Aby',Aby_output_names[1],'Aby_reporting_barchart', sort_by='test_score')
# ### Fibronectin Barcharts test MSE (descending by TEST MSE):
# make_mse_barchart(Fn_best_rows_df, 'Fn',Fn_output_names[0],'Fn_reporting_barchart', sort_by='test_score')
# make_mse_barchart(Fn_best_rows_df, 'Fn',Fn_output_names[1],'Fn_reporting_barchart', sort_by='test_score')

# ### Affibody Barcharts test MSE (descending by RATIO of TEST to CV MSE):
# make_mse_ratio_barchart(Aby_best_rows_df, 'Aby',Aby_output_names[0],'Aby_reporting_barchart')
# make_mse_ratio_barchart(Aby_best_rows_df, 'Aby',Aby_output_names[1],'Aby_reporting_barchart')
# ### Fibronectin Barcharts test MSE (descending by RATIO of TEST to CV MSE):
# make_mse_ratio_barchart(Fn_best_rows_df, 'Fn',Fn_output_names[0],'Fn_reporting_barchart')
# make_mse_ratio_barchart(Fn_best_rows_df, 'Fn',Fn_output_names[1],'Fn_reporting_barchart')

# ### Triple Affibody Barcharts test MSE (descending by ratio, show Test/CV/Ratio together):
# make_mse_triple_barchart(Aby_best_rows_df, 'Aby',Aby_output_names[0],'Aby_reporting_triple_barchart')
# make_mse_triple_barchart(Aby_best_rows_df, 'Aby',Aby_output_names[1],'Aby_reporting_triple_barchart')
# ### Triple Fibronectin Barcharts test MSE (descending by RATIO of TEST to CV MSE):
# make_mse_triple_barchart(Fn_best_rows_df, 'Fn',Fn_output_names[0],'Fn_reporting_triple_barchart')
# make_mse_triple_barchart(Fn_best_rows_df, 'Fn',Fn_output_names[1],'Fn_reporting_triple_barchart')


### Double barchart testing for Aby and Fn
# Double ABY barchart: (abandoned)
# make_mse_double_barchart(Aby_best_rows_df, 'Aby',Aby_output_names[0],'Aby_reporting_double_barchart')
# make_mse_double_barchart(Aby_best_rows_df, 'Aby',Aby_output_names[1],'Aby_reporting_double_barchart')
# # Double Fn barchart:
# make_mse_double_barchart(Fn_best_rows_df, 'Fn',Fn_output_names[0],'Fn_reporting_double_barchart')
# make_mse_double_barchart(Fn_best_rows_df, 'Fn',Fn_output_names[1],'Fn_reporting_double_barchart')
### End of double barchart testing for Aby and Fn

### Double Barchart combined figure (using subfigures)
dfs = [Aby_best_rows_df, Aby_best_rows_df, Fn_best_rows_df, Fn_best_rows_df]
proteins = ['Aby', 'Aby', 'Fn', 'Fn']
output_names = [Aby_output_names[0], Aby_output_names[1], Fn_output_names[0], Fn_output_names[1]]
new_filename = 'combined_double_barchart'
make_stacked_barcharts(dfs, proteins, output_names, new_filename)