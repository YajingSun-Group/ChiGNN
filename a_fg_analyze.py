import pandas as pd

column = 'IC'
chiral = False  # Set this to True if considering chirality
enantiomer = False
if enantiomer:
    result = pd.read_csv(f'/home/jiaxinyan/code_test/functional_group/enantiomer_fg_contribution_diff_{column}.csv')
else:   
    result = pd.read_csv(f'/home/jiaxinyan/code_test/functional_group/functional_group_contributions_{column}.csv')
fg_list = list(set(result['Functional_Group'].tolist()))
fg_list.sort()

if chiral:
    result = result[result['Chiral_Related'] == 'yes']

average_attribution_fg = pd.DataFrame()
mol_num_list = []
mean_att_list = []

for i, fg in enumerate(fg_list):
    result_fg = result[result['Functional_Group'] == fg]
    attribution_mean = result_fg['Contribution'].mean()
    print('**************************************************************************************')
    print("{} function group. number of mol: {}; attribution: {}".format(fg, len(result_fg), round(attribution_mean, 4)))
    print('**************************************************************************************')

    mol_num_list.append(len(result_fg))
    mean_att_list.append(round(attribution_mean, 4))

    print()

average_attribution_fg['Functional_Group'] = [fg for fg in fg_list]
average_attribution_fg['mol_num'] = mol_num_list
average_attribution_fg['attribution_mean'] = mean_att_list
average_attribution_fg.sort_values(by=['attribution_mean'], inplace=True)

if chiral:
    output_file_path = f'functional_group/A_average_attribution_summary_{column}_chiral.csv'
else:
    if enantiomer:
        output_file_path = f'functional_group/A_average_attribution_summary_enantiomer_{column}.csv'
    else:    
        output_file_path = f'functional_group/A_average_attribution_summary_{column}.csv'

average_attribution_fg.to_csv(output_file_path, index=False)
    
    