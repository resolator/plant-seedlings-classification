#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import argparse
import torchvision
import torchvision.transforms as transforms

from PIL import Image


DEVICE = "cpu"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(
        description='Demo application for plant-seedlings-classification.'
    )
    parser.add_argument('--model-path', required=True,
                        help='Path to trained model.')
    parser.add_argument('--img-path', required=True,
                        help='Path to image for predict.')

    return parser.parse_args()


def main():
    """Application entry point."""
    args = get_args()

    input_img_size = 224  # input size for ResNet
    valid_trans = transforms.Compose([
        transforms.Resize((input_img_size, input_img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    img = valid_trans(Image.open(args.img_path).convert('RGB')).to(DEVICE)
    img = img[None, :, :, :]

    ckpt = torch.load(args.model_path, map_location=DEVICE)
    model = torchvision.models.resnet18(
        num_classes=len(ckpt['classes_mp'].keys())
    )
    model.load_state_dict(ckpt['state_dict'])

    predict = torch.max(model(img), 1).indices.numpy()[0]
    id2class_name = {v: k for k, v in ckpt['classes_mp'].items()}

    print('\nPlant seedling on image:', id2class_name[predict])


if __name__ == '__main__':
    main()
