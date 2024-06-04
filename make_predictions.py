import numpy as np
import pandas as pd


data_file = 'results/out.csv'
embeddings_file = 'results/unnormalized_embeddings.csv'
output_file = 'results/comb_predictions.csv'
predictions_col = 'comb_pred'
use_combinatorial_embeddings = True

def get_combinatorial_embedding(file):
    pose_data = np.load(file)
    pose_3d = pose_data['pose_3d']
    
    # 1d embedding of distance between each pair of rows
    embedding = np.zeros((pose_3d.shape[0]*(pose_3d.shape[0]-1)//2))
    idx = 0
    for i in range(pose_3d.shape[0]):
        for j in range(i+1, pose_3d.shape[0]):
            embedding[idx] = np.linalg.norm(pose_3d[i]-pose_3d[j])
            idx += 1
    
    return embedding

# Load the data file
data = pd.read_csv(data_file)

# Load the embeddings file using numpy
if not use_combinatorial_embeddings:
    embeddings = np.loadtxt(embeddings_file, delimiter=',')
else:
    embeddings_files = data['pose_3d_path'].values
    embeddings = np.array([get_combinatorial_embedding(file) for file in embeddings_files])

assert len(data) == len(embeddings)

normalized_embeddings = embeddings/np.linalg.norm(embeddings, axis=1)[:, None]

predictions = np.argmax((normalized_embeddings@normalized_embeddings.T)*(1-np.eye(len(normalized_embeddings))), axis=1)

# predictions col is 'folder' of the corresponding image
data[predictions_col] = [data['folder'][i] for i in predictions]

data.to_csv(output_file, index=False)

