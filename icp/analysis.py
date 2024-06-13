import pandas as pd

df = pd.read_csv('closest_poses.csv')

results = {'original_label': [], 'matching_instances': [], 'mismatch_instances': [], 'mean_mean_error': [], 'std_mean_error': []}

labels = ['jumping_in_air', 'one_hand_on_hip', 'arms_raised', 'sitting_on_ground', 
          'looking_over_shoulder', 'crossed_arms', 'hands_in_pockets', 'hands_clasped']

for label in labels:
    filtered_df = df[df['original_label'] == label]
    
    matching_instances = len(filtered_df[filtered_df['closest_label']==label])
    
    mismatch_instances = len(filtered_df) - matching_instances
    
    mean_mean_error = filtered_df['mean_error'].mean()
    std_mean_error = filtered_df['mean_error'].std()
    
    results['original_label'].append(label)
    results['matching_instances'].append(matching_instances)
    results['mismatch_instances'].append(mismatch_instances)
    results['mean_mean_error'].append(mean_mean_error)
    results['std_mean_error'].append(std_mean_error)

results_df = pd.DataFrame(results)

print(results_df)

results_df.to_csv("icp analysis.csv")