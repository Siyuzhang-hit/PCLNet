# coding=utf-8
import os
import torch
import time
from torch.autograd import Variable
import numpy as np
import cv2
from config_para import parse_option_test
from models.cnn_model import Net, LinearClassifier
from matplotlib import pylab
import matplotlib.pyplot as plt


def main():

    args = parse_option_test()
    torch.set_num_threads(args.set_num_threads)

    model = Net()
    classifier = LinearClassifier(args.layer)

    print('==> loading pre-trained model')
    ckpt_model = torch.load(args.model_path)
    model.load_state_dict(ckpt_model['model'])
    print("==> loaded checkpoint '{}' (epoch {})".format(args.model_path, ckpt_model['epoch']))
    print('==> done')

    print('==> loading linear classifier')
    ckpt_linear = torch.load(args.linear_path)
    classifier.load_state_dict(ckpt_linear['classifier'])
    print("==> loaded checkpoint '{}' (epoch {})".format(args.linear_path, ckpt_linear['epoch']))
    print('==> done')

    start_time = time.time()
    Half_patch = int((args.patch_size - 1) / 2)
    step = 0

    cha_image = np.zeros((args.character_num, args.char_height, args.char_width), dtype=np.uint8)
    for num in range(args.character_num):
        cha1_name = os.path.join(args.cha_path, os.listdir(args.cha_path)[num])
        cha1_image = cv2.imread(cha1_name, cv2.IMREAD_GRAYSCALE)
        cha_image[num, :, :] = cha1_image
    N = np.zeros((args.char_height, args.char_width))
    print('Doing...')
    for i in range(Half_patch, args.char_height - Half_patch, args.stride):
        for j in range(Half_patch, args.char_width - Half_patch, args.stride):
            step += 1
            patch_data = cha_image[:, i - Half_patch: i + Half_patch + 1, j - Half_patch: j + Half_patch + 1]
            patch_data = torch.from_numpy(np.array(patch_data, np.float32))
            patch_tensor = patch_data.unsqueeze(0)/255
            patch_tensor = Variable(patch_tensor)
            feat = model(patch_tensor, args.layer)
            out = classifier(feat)
            pred = torch.max(out, 1)[1]
            N[i, j] = pred.item() + 1
        print('row:{} finish!'.format(i))
    print('Finish!')
    print('Total images are ' + str(step) + ' .')
    end_time = time.time()
    dtime = end_time - start_time
    print('Doing classification time is :{}'.format(dtime))

    Fig = np.zeros((args.char_height, args.char_width, 3))
    for x in range(args.char_height):
        for y in range(args.char_width):
            for k in range(args.label + 1):
                if N[x][y] == k:
                    Fig[x][y] = args.color_value[k]
    Fig = Fig.astype(int)
    plt.imshow(Fig)
    plt.axis('off')
    pylab.show()


if __name__ == '__main__':
    main()
