import os
import sys

sys.path.append('.')

import pandas as pd
import torch
from torchvision import transforms
import argparse
import data.preprocessing.hrnet.model as hrnet
from yacs.config import CfgNode as CN
from PIL import Image
from experiments.default import get_config_file
from models.trainer import Trainer
from models.diffusion import GaussianDiffusion
from models.denoiser import get_denoiser
from models.condition_embedding import get_condition_embedding
from viz.plot_p3d import plot17j_multi, plot17j
from utils.data_utils import (
    reinsert_root_joint_torch, root_center_poses, mpii_to_h36m, mpii_to_h36m_covar, H36M_NAMES, MPII_NAMES
)
import numpy as np
import glob

from tqdm import tqdm


# 16 joints MPII skeleton used for the HRNet detections
MPII_NAMES = ['']*16
MPII_NAMES[0]  = 'RFoot'
MPII_NAMES[1]  = 'RKnee'
MPII_NAMES[2]  = 'RHip'
MPII_NAMES[3]  = 'LHip'
MPII_NAMES[4]  = 'LKnee'
MPII_NAMES[5]  = 'LFoot'
MPII_NAMES[6]  = 'Hip'
MPII_NAMES[7]  = 'Spine'
MPII_NAMES[8]  = 'Thorax'
MPII_NAMES[9]  = 'Head'
MPII_NAMES[10] = 'RWrist'
MPII_NAMES[11] = 'RElbow'
MPII_NAMES[12] = 'RShoulder'
MPII_NAMES[13] = 'LShoulder'
MPII_NAMES[14] = 'LElbow'
MPII_NAMES[15] = 'LWrist'


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def get_hrnet_config_file(config_file):
    _C = CN()

    _C.OUTPUT_DIR = ''
    _C.LOG_DIR = ''
    _C.DATA_DIR = ''
    _C.GPUS = (0,)
    _C.WORKERS = 4
    _C.PRINT_FREQ = 20
    _C.AUTO_RESUME = False
    _C.PIN_MEMORY = True
    _C.RANK = 0

    # Cudnn related params
    _C.CUDNN = CN()
    _C.CUDNN.BENCHMARK = True
    _C.CUDNN.DETERMINISTIC = False
    _C.CUDNN.ENABLED = True

    # common params for NETWORK
    _C.MODEL = CN()
    _C.MODEL.NAME = 'pose_hrnet'
    _C.MODEL.INIT_WEIGHTS = True
    _C.MODEL.PRETRAINED = ''
    _C.MODEL.NUM_JOINTS = 17
    _C.MODEL.TAG_PER_JOINT = True
    _C.MODEL.TARGET_TYPE = 'gaussian'
    _C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
    _C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
    _C.MODEL.SIGMA = 2
    _C.MODEL.EXTRA = CN(new_allowed=True)

    _C.LOSS = CN()
    _C.LOSS.USE_OHKM = False
    _C.LOSS.TOPK = 8
    _C.LOSS.USE_TARGET_WEIGHT = True
    _C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

    # DATASET related params
    _C.DATASET = CN()
    _C.DATASET.ROOT = ''
    _C.DATASET.DATASET = 'mpii'
    _C.DATASET.TRAIN_SET = 'train'
    _C.DATASET.TEST_SET = 'valid'
    _C.DATASET.DATA_FORMAT = 'jpg'
    _C.DATASET.HYBRID_JOINTS_TYPE = ''
    _C.DATASET.SELECT_DATA = False

    # training data augmentation
    _C.DATASET.FLIP = True
    _C.DATASET.SCALE_FACTOR = 0.25
    _C.DATASET.ROT_FACTOR = 30
    _C.DATASET.PROB_HALF_BODY = 0.0
    _C.DATASET.NUM_JOINTS_HALF_BODY = 8
    _C.DATASET.COLOR_RGB = False

    # train
    _C.TRAIN = CN()

    _C.TRAIN.LR_FACTOR = 0.1
    _C.TRAIN.LR_STEP = [90, 110]
    _C.TRAIN.LR = 0.001

    _C.TRAIN.OPTIMIZER = 'adam'
    _C.TRAIN.MOMENTUM = 0.9
    _C.TRAIN.WD = 0.0001
    _C.TRAIN.NESTEROV = False
    _C.TRAIN.GAMMA1 = 0.99
    _C.TRAIN.GAMMA2 = 0.0

    _C.TRAIN.BEGIN_EPOCH = 0
    _C.TRAIN.END_EPOCH = 140

    _C.TRAIN.RESUME = False
    _C.TRAIN.CHECKPOINT = ''

    _C.TRAIN.BATCH_SIZE_PER_GPU = 32
    _C.TRAIN.SHUFFLE = True

    # testing
    _C.TEST = CN()

    # size of images for each device
    _C.TEST.BATCH_SIZE_PER_GPU = 32
    # Test Model Epoch
    _C.TEST.FLIP_TEST = False
    _C.TEST.POST_PROCESS = False
    _C.TEST.SHIFT_HEATMAP = False

    _C.TEST.USE_GT_BBOX = False

    # nms
    _C.TEST.IMAGE_THRE = 0.1
    _C.TEST.NMS_THRE = 0.6
    _C.TEST.SOFT_NMS = False
    _C.TEST.OKS_THRE = 0.5
    _C.TEST.IN_VIS_THRE = 0.0
    _C.TEST.COCO_BBOX_FILE = ''
    _C.TEST.BBOX_THRE = 1.0
    _C.TEST.MODEL_FILE = ''

    # debug
    _C.DEBUG = CN()
    _C.DEBUG.DEBUG = False
    _C.DEBUG.SAVE_BATCH_IMAGES_GT = False
    _C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
    _C.DEBUG.SAVE_HEATMAPS_GT = False
    _C.DEBUG.SAVE_HEATMAPS_PRED = False

    _C.defrost()
    _C.merge_from_file(config_file)
    _C.freeze()

    return _C


image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
        transforms.Resize([255, 255])
    ])


@torch.inference_mode()
def infer(diffpose, hrnet, image, outfol, n_hypo=200, scale=1.):
    fn = image['file_path']

    img = Image.open(fn)

    img.show()

    import time
    t0 = time.time()
    heatmap = hrnet(image_transforms(img).unsqueeze(0))
    t1 = time.time()
    poses = diffpose.sample(heatmap*scale, n_hypotheses_to_sample=n_hypo)
    t2 = time.time()

    poses = reinsert_root_joint_torch(poses)
    poses = root_center_poses(poses) * 1000.

    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('Agg')

    cmap = color_map(17, normalized=True)
    overlayed_img = 0.
    pose_2d = np.zeros((16, 2))
    for i in range(heatmap.shape[1]):
        hm = heatmap[0, i, :, :, None].numpy()
        color = cmap[i + 1][None, None, :]
        colored_heatmap = color * hm * scale
        overlayed_img += colored_heatmap

        amax = np.argmax(hm.flatten())
        x, y = np.unravel_index(amax, hm.shape[:-1])

        # plt.scatter(y, x, c=color[0], label=MPII_NAMES[i])
        pose_2d[i] = [y, x]
        
    # plt.savefig('out.png')
    # plt.imshow(overlayed_img, vmin=0., vmax=1.)
    # plt.legend()
    # plt.savefig('heatmap.png')

    # save 2d pose array as npz and store path in a variable for returning
    pose_2d_path = f'{outfol}/2d_poses/pose_2d_{image["id"]}.npz'
    np.savez(pose_2d_path, pose_2d=pose_2d)

    pose_3d_path = plot17j_multi(
        poses[:5].numpy(),
        return_fig=False,
        save_path=f'{outfol}/3d_poses/pose_3d_{image["id"]}.npz'
    )

    return pose_2d_path, pose_3d_path


def get_pretrained_model(
        config_file='data/preprocessing/hrnet/mpii_hrnet_w32_255x255.yaml',
        model_weight_path='downloads/pretrained/fine_HRNet.pt'  # H36M finetuned
        # model_weights='downloads/pretrained/pose_hrnet_w32_256x256.pth'  # Original MPII
):
    cfg = get_hrnet_config_file(config_file)
    m = hrnet.get_pose_net(
        cfg, False
    )

    model_weights = torch.load(model_weight_path)

    if 'fine' in model_weight_path:
        model_weights = model_weights['net']

    m.load_state_dict(model_weights)
    
    m.eval()

    return m


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 360, 370, 705 works
    # parser.add_argument('--input', type=str, help='Path to image file to generate pose from', default='image_000018.png')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--hmodel', type=str, choices=['fine', 'original'], default='fine')

    args = parser.parse_args()

    if args.hmodel == 'fine':
        model_weights = 'downloads/pretrained/fine_HRNet.pt'
    else:
        model_weights = 'downloads/pretrained/pose_hrnet_w32_256x256.pth'

    cfg = get_config_file(args.config)

    hrnet = get_pretrained_model(model_weight_path=model_weights)
    hrnet.eval()

    denoiser = get_denoiser(cfg).to('cpu')
    conditioner = get_condition_embedding(cfg).to('cpu')
    diffusion = GaussianDiffusion(
        denoiser,
        conditioner,
        image_size=128,
        objective=cfg.MODEL.DIFFUSION_OBJECTIVE,
        timesteps=cfg.MODEL.TIMESTEPS,  # number of steps
        loss_type=cfg.LOSS.LOSS_TYPE,  # L1 or L2
        scaling_3d_pose=cfg.TRAIN.SCALE_3D_POSE,  # Pre-conditioning of target scale
        noise_scale=cfg.MODEL.NOISE_SCALE,
        cosine_offset=cfg.TRAIN.COSINE_OFFSET
    ).to('cpu')

    diffusion.load_state_dict(torch.load('downloads/pretrained/model-final.pt')['model'])

    data_file = 'results/data.csv'
    output_folder = 'results/orig'
    output_csv_file = f'{output_folder}/out.csv'

    df = pd.read_csv(data_file)
    os.makedirs(f'{output_folder}/2d_poses', exist_ok=True)
    os.makedirs(f'{output_folder}/3d_poses', exist_ok=True)

    # create results/out.csv with pose_2d_path and pose_3d_path as additional cols
    df['pose_2d_path'] = ''
    df['pose_3d_path'] = ''

    pb = tqdm(df.iterrows(), total=len(df))

    for i, row in df.iterrows():
        image = {
            'file_path': row['file_path'],
            'id': row['id']
        }

        try:
            pose_2d_path, pose_3d_path = infer(diffusion, hrnet, image, output_folder)
        except Exception as e:
            print(f'Error processing {image["file_path"]}: {e}')
            pose_2d_path = ''
            pose_3d_path = ''
            continue
        
        df.at[i, 'pose_2d_path'] = pose_2d_path
        df.at[i, 'pose_3d_path'] = pose_3d_path

        pb.update(1)
    
    df.to_csv(output_csv_file, index=False)



