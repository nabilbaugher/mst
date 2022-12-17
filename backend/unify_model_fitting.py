import pandas as pd
import csv
from model_evaluation import DecisionModelRange, avg_log_likelihood_decisions, softmax_complement, decisions_to_subject_decisions, model_preference
from decisionmodel import random_heuristic, steps_cells_heuristic_with_memory, steps_cells_heuristic, blind_nodevalue_with_memory, blind_nodevalue_comb
from data_parser import get_csv, convert_data
import ast

# updated_data = {'0e9aebe0-7972-11ed-9421-5535258a0716': ('Combined_Memory', -0.43773071546593767, [0.0, 1.5, 0.75, 0.1, 10.0]), '14914210-7972-11ed-a3bb-674c56c9a5f7': ('Combined_Memory', -0.4492674779027006, [0.0, 1.0, 0.5, 0.1, 10.0]), '1af65fa0-7972-11ed-8e76-65fbafc77b31': ('Combined_Memory', -0.5141892059147661, [0.0, 0.5, 0.75, 0.1, 10.0]), '1c5fb3a0-7972-11ed-92f5-192da827d727': ('Combined_Memory', -0.4736891209910147, [0.0, 1.5, 0.25, 0.1, 10.0]), '1ed4f960-7972-11ed-abfe-53a0662f3357': ('Combined_Memory', -0.5765930680216199, [0.0, 0.5, 0.75, 0.1, 10.0]), '1fcc78c0-7972-11ed-89ce-d32c56bd82a2': ('Combined_Memory', -0.49089431921424864, [0.0, 1.5, 0.5, 0.1, 10.0]), '20ae4d40-7972-11ed-97e1-396710a0d358': ('Combined_Memory', -0.46897138543577044, [0.0, 1.0, 0.5, 0.1, 10.0]), '29cee150-7972-11ed-af98-919b1d80d094': ('Combined_Memory', -0.5502788252722078, [0.0, 0.5, 0.5, 0.1, 10.0]), '33782d60-7972-11ed-a05f-b374eea697ce': ('Combined_Memory', -0.39589500493818935, [0.0, 1.5, 0.75, 0.1, 10.0]), '3520d450-7972-11ed-9198-71d540724e0e': ('Combined_Memory', -0.5043509474345823, [0.0, 0.5, 0.75, 0.1, 10.0]), '379d1ef0-7972-11ed-8e63-5babb1808d0d': ('Combined_Memory', -0.4450749795703202, [0.0, 1.5, 0.5, 0.1, 10.0]), '3aef3020-7972-11ed-b7fe-a36af006f6b8': ('Combined_Memory', -0.5450402977324543, [0.0, 0.5, 0.5, 0.1, 10.0]), '3c3a26b0-7972-11ed-a45d-61844e9fbfe3': ('Combined_Memory', -0.4037026516033588, [0.0, 1.5, 0.25, 0.1, 10.0]), '3de369e0-7972-11ed-8f5e-a3c68d1aedd3': ('Combined_Memory', -0.5448434225463877, [0.0, 1.5, 0.25, 0.1, 10.0]), '3e3ea940-7972-11ed-93e3-8dbe2fd6e58a': ('Combined_Memory', -0.46086719987572905, [0.0, 1.5, 0.5, 0.1, 10.0]), '4466e4e0-7972-11ed-9b44-876234774939': ('Combined_Memory', -0.4639398031102234, [0.0, 1.5, 0.5, 0.1, 10.0]), '44da4020-7972-11ed-854c-03a7750b8ad6': ('Combined_Memory', -0.5117401832515842, [0.0, 1.0, 0.75, 0.1, 10.0]), '45cb7df0-7972-11ed-a9e1-0b8c78d81cd7': ('Combined_Memory', -0.4640190681029924, [0.0, 1.5, 0.5, 0.1, 10.0]), '4828a7d0-7972-11ed-ac6e-0b30e16b2513': ('Combined_Memory', -0.3751171424899279, [0.0, 1.5, 0.75, 0.1, 10.0]), '552898f0-7972-11ed-a8f6-c50b62b52bcd': ('Combined_Memory', -0.521398158062306, [0.0, 1.0, 0.75, 0.1, 10.0])}


def create_model_range(model_name, node_params_ranges, parent_params_ranges):
    raw_nodevalue_func = blind_nodevalue_comb
    if model_name == 'Combined_Memory':
        raw_nodevalue_func = blind_nodevalue_with_memory
    if model_name == 'Steps-Cells':
        raw_nodevalue_func = steps_cells_heuristic
    if model_name == 'Steps_Cells_Memory':
        raw_nodevalue_func = steps_cells_heuristic_with_memory
    if model_name == 'Random':
        raw_nodevalue_func = random_heuristic
    return DecisionModelRange(model_name=model_name,
                              evaluation_function=avg_log_likelihood_decisions,
                              raw_nodevalue_func=raw_nodevalue_func,
                              parent_nodeprob_func=softmax_complement,
                              node_params_ranges=node_params_ranges,
                              parent_params_ranges=parent_params_ranges
                             )


def unify_model_fitting_data_new(new_file, target_file):
    """
    Unify the results of multiple runs of model fitting.
    In this case we want to add the data from the steps-cells heuristic with memory.
    """
    target_data = pd.read_csv(target_file)
    target_dict = target_data.to_dict('index')
    prev_data = {}
    for index in target_dict:
        row = target_dict[index]
        prev_data[row['subject']] = (row['preferred_model'], row['evaluation'], row['parameters'])
        
    # get subject decisions
    path = get_csv("./data/prolific_data_sorted_tester_id_created_at")
    decisions = convert_data(path)
    subject_decisions = decisions_to_subject_decisions(decisions)
        
    # rerun model fitting to compare parameters from updated_data and from prev_data
    for subject in prev_data:
        prev_model = prev_data[subject][0]
        prev_eval = float(prev_data[subject][1])
        prev_params = prev_data[subject][2]
        
        new_model = 'Steps_Cells_Memory'
        new_node_params = ((0,2,5),(.25,.75,3),(.1,.5,5))
        new_parent_params = ((10,10,1),)
        
        model_to_test = create_model_range(new_model, new_node_params, new_parent_params)
        decision = {subject: subject_decisions[subject]}
        new_subject_preference = model_preference([model_to_test], decision, None, k=4)
        
        evaluation = new_subject_preference[subject][1]
        if evaluation > prev_eval:
            prev_data[subject] = (new_model, evaluation, new_subject_preference[subject][2])
        else:
            prev_data[subject] = (prev_model, prev_eval, prev_params)
        print('new datum')
        print(prev_data[subject])
    
    # write new data to file
    with open(new_file, 'w') as f:
        f.write(',subject,preferred_model,evaluation,parameters')
        writer = csv.writer(f)
        for i, subject in enumerate(prev_data):
            writer.writerow([i, subject, prev_data[subject][0], prev_data[subject][1], prev_data[subject][2]])
    
print(unify_model_fitting_data_new('data/new_file.csv', 'data/current_model_prefs.csv'))