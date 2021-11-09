import argparse

import cv2
import numpy as np
import torch

from backbones import get_model


#计算余弦距离
def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

@torch.no_grad()
def compare(weight, name, img1,img2):
    if img1 is None:
        img1 = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img1 = cv2.imread(img1)
        img1 = cv2.resize(img1, (112, 112))

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = np.transpose(img1, (2, 0, 1))
    img1 = torch.from_numpy(img1).unsqueeze(0).float()
    img1.div_(255).sub_(0.5).div_(0.5)
    if img2 is None:
        img2 = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img2 = cv2.imread(img2)
        img2 = cv2.resize(img2, (112, 112))

    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = np.transpose(img2, (2, 0, 1))
    img2 = torch.from_numpy(img2).unsqueeze(0).float()
    img2.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=True)
    net.load_state_dict(torch.load(weight))
    #torch.save(net.state_dict(), 'weights/resnet18/resnet18_pth.pth', _use_new_zipfile_serialization=False)
    net.eval()
    feat1 = net(img1).numpy()
    feat2 = net(img2).numpy()
    print(feat1, feat2)
    #print(feat1.reshape(-1,1)*feat2)
    print(cosin_metric(feat1, feat2.reshape(-1,1)))

@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat = net(img).numpy()
    print(feat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r18', help='backbone network')
    parser.add_argument('--weight', type=str, default='./weights/ms1mv3_arcface_r18_fp16/resnet18_pth.pth')
    parser.add_argument('--img', type=str, default='curve/cut_faces/gzd_01_1.jpg')
    args = parser.parse_args()
    #inference(args.weight, args.network, args.img)
    compare(args.weight, args.network, 'curve/cut_faces/xz_01_1.jpg', 'curve/cut_faces/xz_01_1.jpg')
