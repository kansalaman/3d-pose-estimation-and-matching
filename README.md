# 3D Pose Analysis: Predict, Match & Search
## Authors: Megan Dass & Aman Kansal

### Setup
1. Clone and enter repo

2. Install dependencies using conda environment yaml
```
conda env create -f environment.yml
```

3. Activate conda environment
```
conda activate prob_pose
```

### DiffPose
The code for the DiffPose experiment is modified from the [original repository](3D Pose Analysis: Predict, Match & Search). 

To train the DiffPose model and generate the posefile, run: ```generate_posefile.py```


### ICP
To run the ICP algorithm:

```
cd icp
python icp.py
```

To analyze the pose estimations, run:
```python analysis.py```