import argparse

import torch
import torch.nn.parallel
import datasets
from utils import AverageMeter, img_cvt
import soft_renderer as sr
import soft_renderer.functional as srf
import models
import models_large
import time
import os
import imageio
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

BATCH_SIZE = 100
IMAGE_SIZE = 64
CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')

PRINT_FREQ = 5
SAVE_FREQ = 100

MODEL_DIRECTORY = 'results/models/checkpoint_0004999.pth.tar'
DATASET_DIRECTORY = 'data/datasets'

modelname = "checkpoint_0003998.pth.tar"
SIGMA_VAL = 0.01
IMAGE_PATH = ''

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('-eid', '--experiment-id', type=str)
parser.add_argument('-d', '--model-directory', type=str, default=MODEL_DIRECTORY)
parser.add_argument('-dd', '--dataset-directory', type=str, default=DATASET_DIRECTORY)

parser.add_argument('-cls', '--class-ids', type=str, default=CLASS_IDS_ALL)
parser.add_argument('-is', '--image-size', type=int, default=IMAGE_SIZE)
parser.add_argument('-bs', '--batch-size', type=int, default=BATCH_SIZE)
parser.add_argument('-img', '--image-path', type=str, default=IMAGE_PATH)

parser.add_argument('-sv', '--sigma-val', type=float, default=SIGMA_VAL)

parser.add_argument('-pf', '--print-freq', type=int, default=PRINT_FREQ)
parser.add_argument('-sf', '--save-freq', type=int, default=SAVE_FREQ)
parser.add_argument('--shading-model', action='store_true', help='test shading model')
args = parser.parse_args()

# setup model & optimizer
if args.shading_model:
    model = models_large.Model('data/obj/sphere/sphere_642.obj', args=args)
else:
    model = models.Model('data/obj/sphere/sphere_642.obj', args=args)
model = model.cuda()


print("model_directory: ", args.model_directory)
current_dir = os.path.dirname(os.path.realpath(__file__))
real_dir = os.path.join(current_dir, 'results', 'models', modelname)

state_dicts = torch.load(real_dir)
# import ipdb; ipdb.set_trace()
model.load_state_dict(state_dicts['model'], strict=True)
model.eval()

directory_output = 'data/results/test'

# directory_output = os.path.join(args.model_directory)
print("directory output: ", directory_output)
os.makedirs(directory_output, exist_ok=True)
directory_mesh = os.path.join(directory_output, 'ex55')
os.makedirs(directory_mesh, exist_ok=True)


def test():
   
    # load the image
    image = Image.open('internet_images/tplane.png')
    # convert image to numpy array
    # convert to float and normalize
    data = torch.from_numpy(np.asarray(image).astype('float32') / 255.)
    data = data.reshape(1,4,64,64)
    demo_image = torch.autograd.Variable(data).cuda()
    demo_v, demo_f = model.reconstruct(demo_image)

    
    demo_path = os.path.join('test_output_plane_4999.obj')
    srf.save_obj(demo_path, demo_v[0], demo_f[0])

    
    #plt.savefig('foo.png')


test()
