import argparse
import datasets
import models

import os


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


if __name__ == '__main__':
    IMAGE_SIZE = 64
    BATCH_SIZE = 64
    SIGMA_VAL = 1e-4
    DATASET_DIRECTORY = 'data/datasets'
    CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')

    PRINT_FREQ = 5
    SAVE_FREQ = 100

    parser = argparse.ArgumentParser()
    parser.add_argument('-is', '--image-size', type=int, default=IMAGE_SIZE)
    parser.add_argument('-b', '--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('-sv', '--sigma-val', type=float, default=SIGMA_VAL)
    parser.add_argument('-dd', '--dataset-directory', type=str, default=DATASET_DIRECTORY)
    
    parser.add_argument('-pf', '--print-freq', type=int, default=PRINT_FREQ)
    parser.add_argument('-sf', '--save-freq', type=int, default=SAVE_FREQ)

    args = parser.parse_args()

    model = models. Model('data/obj/sphere/sphere_642.obj', args=args)
    model = model.cuda()


    CLASS_IDS_EXAMPLE = (
    '02691156')
    CLASS_IDS_EXAMPLE_TEST = (
    '02691156')
    
    dataset_train = datasets.ShapeNet(DATASET_DIRECTORY, CLASS_IDS_EXAMPLE.split(','), 'train')

    images_a, images_b, viewpoints_a, viewpoints_b = dataset_train.get_random_batch(batch_size=BATCH_SIZE)


    # print sizes

    print("images_a shape: ", images_a.shape)
    print("images_b shape: ", images_b.shape)
    print("viewpoints_a shape: ", viewpoints_a.shape)
    print("viewpoints_b shape: ", viewpoints_b.shape)

    images_a = images_a.cuda()
    images_b = images_b.cuda()
    viewpoints_a = viewpoints_a.cuda()
    viewpoints_b = viewpoints_b.cuda()

    # soft render images
    render_images, laplacian_loss, flatten_loss = model([images_a, images_b],
                                                        [viewpoints_a, viewpoints_b],
                                                        task='train')

    
    print("========== rendered =========")
    print(render_images)

    
    # ===== test models ====
    test_model = models. Model('data/obj/sphere/sphere_642.obj', args=args)
    test_model = test_model.cuda()

    modelname = "checkpoint_0004999.pth.tar"
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(current_dir, 'results', 'models', modelname)

    state_dicts = torch.load(model_dir)
    test_model.load_state_dict(state_dicts['model'], strict=True)
    test_model.eval()

    dataset_val = datasets.ShapeNet(DATASET_DIRECTORY, CLASS_IDS_EXAMPLE_TEST.split(','), 'val')

    directory_output = 'data/results/test'
    print("directory output: ", directory_output)
    os.makedirs(directory_output, exist_ok=True)
    directory_mesh = os.path.join(directory_output, 'test2')
    os.makedirs(directory_mesh, exist_ok=True)

    end = time.time()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()

    iou_all = []

    for class_id, class_name in dataset_val.class_ids_pair:

        directory_mesh_cls = os.path.join(directory_mesh, class_id)
        os.makedirs(directory_mesh_cls, exist_ok=True)
        iou = 0

        for i, (im, vx) in enumerate(dataset_val.get_all_batches_for_evaluation(args.batch_size, class_id)):
            images = torch.autograd.Variable(im).cuda()
            voxels = vx.numpy()
            batch_iou, vertices, faces = test_model(images, voxels=voxels, task='test')
            iou += batch_iou.sum()

            batch_time.update(time.time() - end)
            end = time.time()

            # save demo images
            for k in range(vertices.size(0)):
                obj_id = (i * args.batch_size + k)
                if obj_id % args.save_freq == 0:
                    mesh_path = os.path.join(directory_mesh_cls, '%06d.obj' % obj_id)
                    input_path = os.path.join(directory_mesh_cls, '%06d.png' % obj_id)
                    srf.save_obj(mesh_path, vertices[k], faces[k])
                    imageio.imsave(input_path, img_cvt(images[k]))

            # print loss
            if i % args.print_freq == 0:
                print('Iter: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f}\t'
                      'IoU {2:.3f}\t'.format(i, ((dataset_val.num_data[class_id] * 24) // args.batch_size),
                                            batch_iou.mean(),
                                            batch_time=batch_time))

        iou_cls = iou / 24. / dataset_val.num_data[class_id] * 100
        iou_all.append(iou_cls)
        print('=================================')
        print('Mean IoU: %.3f for class %s' % (iou_cls, class_name))
        print('\n')

    print('=================================')
    print('Mean IoU: %.3f for all classes' % (sum(iou_all) / len(iou_all)))

