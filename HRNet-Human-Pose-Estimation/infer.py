from copy import deepcopy
import torch
import cv2
import numpy as np
from lib.models import pose_hrnet
from lib.config import cfg, update_config

# Update configuration for the 256x256 HRNet model
cfg.merge_from_file("/home/user/DiffPose/HRNet-Human-Pose-Estimation/experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml")

# Create the HRNet model
model = pose_hrnet.get_pose_net(cfg, is_train=False)

# Load pre-trained weights
state_dict = torch.load("/home/user/DiffPose/data/preprocessing/pose_hrnet_w32_256x256.pth")
model.load_state_dict(state_dict)
model.eval()

# Load the input image
image = cv2.imread('/home/user/DiffPose/data/demo/arms-back.jpg')

# Preprocess the image
height, width = image.shape[:2]
image = cv2.resize(image, (256, 256))  # Resize to HRNet input resolution
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.transpose(image, (2, 0, 1))  # HWC to CHW
image = np.expand_dims(image, axis=0)  # Add batch dimension
image = image / 255.0  # Normalize to [0, 1]

# Run HRNet inference
with torch.no_grad():
    keypoints = model(torch.from_numpy(image).float())[0].detach().cpu().numpy()

keypoints = np.array([np.unravel_index(np.argmax(heatmap), heatmap.shape) for heatmap in keypoints])

# Post-process keypoints
keypoints_1 = deepcopy(keypoints)

keypoints[:, 0] = keypoints_1[:, 0] * height / 64  # Scale x coordinates
keypoints[:, 1] = keypoints_1[:, 1] * width / 64  # Scale y coordinates

keypoints [:, [0, 1]] = keypoints [:, [1, 0]]  # Swap x and y coordinates

# Draw the predicted skeleton on the input image
image = cv2.imread('/home/user/DiffPose/data/demo/arms-back.jpg')
for i in range(keypoints.shape[0]):
    x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
    cv2.circle(image, (x, y), 9, (0, 0, 255), -1)

# Display the image with predicted skeleton
cv2.imwrite('output_image.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()