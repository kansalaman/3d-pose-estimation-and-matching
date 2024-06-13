import numpy as np
import pandas as pd
import os
from scipy.spatial import KDTree

def nearest_neighbor(src, dst):
        tree = KDTree(dst)
        _, indices = tree.query(src)
        return indices

def transform(A, B):
    assert A.shape == B.shape

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

def icp(A, B, max_iterations=50, tolerance=1e-6):
    src = np.ones((A.shape[0], 4))
    dst = np.ones((B.shape[0], 4))
    src[:, :3] = A
    dst[:, :3] = B

    prev_error = 0

    for i in range(max_iterations):
        indices = nearest_neighbor(src[:, :3], dst[:, :3])
        closest_points = dst[indices, :]

        T = transform(src[:, :3], closest_points[:, :3])

        src = (T @ src.T).T

        mean_error = np.mean(np.linalg.norm(src[:, :3] - closest_points[:, :3], axis=1))

        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    T = transform(A, src[:, :3])

    return T, mean_error

pose_info_path = 'pose_info/out.csv'
df = pd.read_csv(pose_info_path)

def load_3d_keypoints(file_path):
    file_path = "pose_info/" + file_path
    if os.path.exists(file_path):
        data = np.load(file_path)
        return data['pose_3d']
    
results = []

for idx, row in df.iterrows():
    target_pose_3d_path = row['pose_3d_path']
    if target_pose_3d_path == 'file_path':
        continue

    target_pose_3d = load_3d_keypoints(target_pose_3d_path)
    if target_pose_3d is None:
        continue

    best_match = None
    lowest_error = float('inf')

    for index, match_row in df.iterrows():
        if index == idx:
            continue
        
        pose_3d_path = match_row['pose_3d_path']
        if pose_3d_path == 'file_path':
            continue
        
        pose_3d = load_3d_keypoints(pose_3d_path)
        if pose_3d is not None:
            _, error = icp(target_pose_3d, pose_3d)
            if error < lowest_error:
                lowest_error = error
                best_match = match_row

    original_class = row['id']
    matched_class = best_match['id'] if best_match is not None else None

    results.append({
        'original_label': row['folder'],
        'closest_label': best_match['folder'] if best_match is not None else None,
        'original_id': row['id'],
        'closest_image_id': matched_class,
        'mean_error': lowest_error
    })

results_df = pd.DataFrame(results)

output_csv_path = 'closest_poses.csv'
results_df.to_csv(output_csv_path, index=False)

print("ICP Completed.")