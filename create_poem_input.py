import numpy as np
import pandas as pd


pose_data_file = 'results/orig/out.csv'
output_csv = 'results/orig/poem_input.csv'


# Load the pose data file
pose_data = pd.read_csv(pose_data_file)

poses_2d_npzs = pose_data['pose_2d_path'].values

table_header = \
'''image/width,
image/height,
image/object/part/NOSE_TIP/center/x,
image/object/part/NOSE_TIP/center/y,
image/object/part/NOSE_TIP/score,
image/object/part/LEFT_SHOULDER/center/x,
image/object/part/LEFT_SHOULDER/center/y,
image/object/part/LEFT_SHOULDER/score,
image/object/part/RIGHT_SHOULDER/center/x,
image/object/part/RIGHT_SHOULDER/center/y,
image/object/part/RIGHT_SHOULDER/score,
image/object/part/LEFT_ELBOW/center/x,
image/object/part/LEFT_ELBOW/center/y,
image/object/part/LEFT_ELBOW/score,
image/object/part/RIGHT_ELBOW/center/x,
image/object/part/RIGHT_ELBOW/center/y,
image/object/part/RIGHT_ELBOW/score,
image/object/part/LEFT_WRIST/center/x,
image/object/part/LEFT_WRIST/center/y,
image/object/part/LEFT_WRIST/score,
image/object/part/RIGHT_WRIST/center/x,
image/object/part/RIGHT_WRIST/center/y,
image/object/part/RIGHT_WRIST/score,
image/object/part/LEFT_HIP/center/x,
image/object/part/LEFT_HIP/center/y,
image/object/part/LEFT_HIP/score,
image/object/part/RIGHT_HIP/center/x,
image/object/part/RIGHT_HIP/center/y,
image/object/part/RIGHT_HIP/score,
image/object/part/LEFT_KNEE/center/x,
image/object/part/LEFT_KNEE/center/y,
image/object/part/LEFT_KNEE/score,
image/object/part/RIGHT_KNEE/center/x,
image/object/part/RIGHT_KNEE/center/y,
image/object/part/RIGHT_KNEE/score,
image/object/part/LEFT_ANKLE/center/x,
image/object/part/LEFT_ANKLE/center/y,
image/object/part/LEFT_ANKLE/score,
image/object/part/RIGHT_ANKLE/center/x,
image/object/part/RIGHT_ANKLE/center/y,
image/object/part/RIGHT_ANKLE/score'''.replace('\n', '')

table_rows = []

for pose_2d_npz in poses_2d_npzs:
    pose_2d = np.load(pose_2d_npz)['pose_2d']
    pose_2d/=64.0
    
    row = []

    row.append(64)
    row.append(64)

    # NOSE_TIP: 9
    # LEFT_SHOULDER: 13
    # RIGHT_SHOULDER: 12
    # LEFT_ELBOW: 14
    # RIGHT_ELBOW: 11
    # LEFT_WRIST: 15
    # RIGHT_WRIST: 10
    # LEFT_HIP: 3
    # RIGHT_HIP: 2
    # LEFT_KNEE: 4
    # RIGHT_KNEE: 1
    # LEFT_ANKLE: 5
    # RIGHT_ANKLE: 0

    #score always 1
    indices = [9, 13, 12, 14, 11, 15, 10, 3, 2, 4, 1, 5, 0]
    for i in indices:
        row.append(pose_2d[i][0])
        row.append(pose_2d[i][1])
        row.append(1)
    
    table_rows.append(','.join(map(str, row)))


with open(output_csv, 'w') as f:
    f.write(table_header + '\n')
    f.write('\n'.join(table_rows))
