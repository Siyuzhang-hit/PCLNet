# coding=utf-8
from __future__ import print_function
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class dataset(Dataset):

    def __init__(self, args, path, transform=None, two_crop=False):
        self.imgs, self.labs = self.get_images_and_labels_list(args, path)
        self.shape = (args.patch_size, args.patch_size, args.character_num)
        self.transform = transform
        self.two_crop = two_crop

    def __getitem__(self, index):

        fn = self.imgs[index]
        image = self.loader(fn, self.shape)
        label = self.labs[index]
        if self.transform is not None:
            img = self.transform(image)
            img = img.type(torch.FloatTensor)
        return img, label

    def loader(self, path, shape):
        img_con = np.empty(shape=shape,  dtype='uint8')
        for num in range(len(path)):
            img_con[:, :, num] = Image.open(path[num]).convert('L')
        return img_con


    def get_images_and_labels_list(self, args, path):
        total_images_list = []
        labels_list = []
        # Get the path to all images.
        for cha_file_name in os.listdir(path):
            images_list = []
            for image_name in os.listdir(path + cha_file_name):
                images_list.append(path + cha_file_name + '/' + image_name)
                # Judge label.
                for i in range(len(image_name)):
                    if image_name[i] == '_':
                        for j in range(len(args.label_name)):
                            if args.label_name[j] == image_name[:i]:
                                labels_list.append(j)
                                break
            total_images_list.append(images_list)
        # Average cutting label list
        equal = int(len(labels_list) / args.character_num)
        labels_list = labels_list[:equal]
        total_images_list = [[row[i] for row in total_images_list] for i in range(len(total_images_list[0]))]
        return total_images_list, labels_list

    def __len__(self):
        return len(self.imgs)


class self_supervised_dataset(Dataset):

    def __init__(self, args, path, transform=None, two_crop=True):
        self.imgs = self.get_images_and_labels_list(args, path)
        self.shape = (args.patch_size, args.patch_size, args.character_num)
        self.transform = transform
        self.two_crop = two_crop

    def __getitem__(self, index):

        fn = self.imgs[index]
        image = self.loader(fn, self.shape)
        if self.transform is not None:
            img = self.transform(image)
            img = img.type(torch.FloatTensor)
        if self.two_crop:
            # ToDo:resize, crop and flip
            image2 = np.flip(image, axis=0).copy()
            img2 = self.transform(image2)
            img2 = img2.type(torch.FloatTensor)
            img = torch.cat([img, img2], dim=0)

        return img

    def loader(self, path, shape):
        img_con = np.empty(shape=shape,  dtype='uint8')
        for num in range(len(path)):
            img_con[:, :, num] = Image.open(path[num]).convert('L')
        return img_con

    def get_images_and_labels_list(self, path):
        total_images_list = []
        # Get the path to all images.
        for cha_file_name in os.listdir(path):
            images_list = []
            for image_name in os.listdir(path + cha_file_name):
                images_list.append(path + cha_file_name + '/' + image_name)
            total_images_list.append(images_list)
        # Average cutting label list
        total_images_list = [[row[i] for row in total_images_list] for i in range(len(total_images_list[0]))]
        return total_images_list


    def __len__(self):
        return len(self.imgs)
