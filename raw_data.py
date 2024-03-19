import pandas as pd

import warnings
warnings.filterwarnings("ignore")

### Start of Greg Input data:
Aby_df = pd.read_csv('datasets/Developability_scores_aby.csv', encoding='unicode_escape')
Fn_df = pd.read_csv('datasets/Developability_scores_fn.csv', encoding='unicode_escape')

AbyTL = Aby_df['Average TL']
AbysGFP = Aby_df['Average sGFP']
AbyNSB = Aby_df['NSB Average']
AbyDB = Aby_df['Dot Blot Avg (ug/mL)']
AbyTM = Aby_df['CD Tm (degC)']
Aby_input_metrics = [AbyTL,AbysGFP,AbyNSB]
Aby_input_names=['Average TL','Average sGFP','NSB Average']
Aby_output_metrics = [AbyDB,AbyTM]
Aby_output_names=['Dot Blot Avg (ug/mL)','CD Tm (degC)']

FnTL = Fn_df['Average TL']
FnsGFP = Fn_df['Average sGFP']
FnNSB = Fn_df['NSB Average']
FnSDS = Fn_df['SDS PAGE (mg/mL)']
FnTM = Fn_df['CD Tm (degC)']
Fn_input_metrics=[FnTL,FnsGFP,FnNSB]
Fn_input_names=['Average TL','Average sGFP','NSB Average']
Fn_output_metrics=[FnSDS,FnTM]
Fn_output_names=['SDS PAGE (mg/mL)','CD Tm (degC)']
### End of Greg Input Data