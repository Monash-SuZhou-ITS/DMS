from __future__ import print_function
import os
import argparse
import torch
from torch.autograd import *
import torch.backends.cudnn as cudnn
import pickle
import numpy as np

from SSD import utils
from SSD.voc0712 import *
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from SSD.ssd_net_vgg import *
from SSD.detection import *
import SSD.Config as Config
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from models.resnet import resnet_face18
from torch.nn import DataParallel
from utils.box_utils import decode, decode_landm
import time

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--retinaface_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--arcface_model', default='./weights/resnet18_110.pth',
                    type=str, help='Trained arcface state_dict file path to open')
parser.add_argument('--retinaface_network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--arcface_network',default='mobile0.25', help='Backbone network mobile0.25 or resnet50')

parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.8, type=float, help='visualization_threshold')
parser.add_argument('--device', default='0', help='cuda device,  0 or 0,1,2,3 or cpu')
parser.add_argument('--show_cutting_image', action ="store_true", default =True, help = 'show_crop_images')
parser.add_argument('--save_folder', default='./curve/info', type=str, help='Dir to save results')
parser.add_argument('--face_images_path', default='./data/face_images', type=str, help='path to load face_images')
parser.add_argument('--face_features_save_path', default='./data/face_features/', type=str, help='path to save face_features')
parser.add_argument('--face_features_save_name', default='face_features', type=str, help=' save face_features name')

args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def feature_save(arcface_model, face_images_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_data = save_path + args.face_features_save_name
    doc = open(save_data, "ab+")
    persons_faces = []  # 建立人脸库的列表
    for person in os.listdir(face_images_path):  # 遍历每一个人脸文件夹
        person_face = []  # 用来同一个人的不同的人脸特征（一个人获取的可能不止一张照片）
        # persons_name.append(person)  # 存放人的名字
        for face in os.listdir(os.path.join(face_images_path, person)):  # 人脸照片转换为特征
            person_image = load_image(os.path.join(face_images_path, person, face))#transform(Image.open(os.path.join(file_path, person, face))).cuda()
            data = torch.from_numpy(person_image)
            data = data.to(torch.device("cuda"))
            #person_feature = net.encode(person_picture[None, ...])  # 获取编码后的每一个人的脸部特征
            output = arcface_model(data)  # 获取特征
            output = output.data.cpu().numpy()
            print(output.shape)

            fe_1 = output[::2]#正面特征
            fe_2 = output[1::2]#镜像特征
            # print("this",cnt)
            # print(fe_1.shape,fe_2.shape)
            person_feature = np.hstack((fe_1, fe_2))
            person_feature = person_feature.reshape(1024)
            #feature = person_feature.detach().cpu()  # 将脸部特征转到CPU上，节省GPU的计算量
            person_face.append(person_feature)  # 将同一个人脸的不同人脸特征存放到同一个列表中
        # persons_faces[person] = person_face  #
        persons_faces.append([person, torch.cat(person_face, dim=0)])  # 将不同人的名字、脸部特征存放到同一个列表中
    pickle.dump(persons_faces, doc)  # 按照列表形式存入文件

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def load_image(img_path):
    image = cv2.imread(img_path, 0)

    if image is None:
        return None
    #image = cv2_letterbox_image(image,128)
    image = cv2.resize(image, (128, 128))
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.dstack((image, np.fliplr(image)))

    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image

def image_format(image):
    if image is None:
        return None
    #image = cv2_letterbox_image(image,128)
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image
#计算余弦距离
def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def compare(load_path, pic1_path, pic2_path):
    # img1 = Image.open(picture1_path)
    # if img1.mode != 'L':
    #     img = img1.convert('L')
    # person1 = tf(img1).cuda()
    # person1_feature = arcface_model(person1[None, ...])[0]

    person1_image = load_image(pic1_path)
    data = torch.from_numpy(person1_image)
    data = data.to(torch.device("cuda"))
    output = arcface_model(data)  # 获取特征
    output = output.data.cpu().numpy()

    # 获取不重复图片 并分组
    fe_1 = output[::2]
    fe_2 = output[1::2]
    # print("this",cnt)
    # print(fe_1.shape,fe_2.shape)
    person1_feature = np.hstack((fe_1, fe_2))
    person1_feature = person1_feature.reshape(1024)
    #print(person1_feature.shape)

    # img2 = Image.open(picture2_path)
    # if img2.mode != 'L':
    #     img2 = img2.convert('L')
    # person2 = tf(img2).cuda()
    # person2_feature = arcface_model(person2[None, ...])[0]
    # person2_image = load_image(pic2_path)
    person2_image = load_image(pic2_path)
    data = torch.from_numpy(person2_image)
    data = data.to(torch.device("cuda"))
    output = arcface_model(data)  # 获取特征
    output = output.data.cpu().numpy()

    fe_1 = output[::2]
    fe_2 = output[1::2]
    person2_feature = np.hstack((fe_1, fe_2))
    person2_feature = person2_feature.reshape(1024)
    diff = cosin_metric(person1_feature, person2_feature)
    # siam = compare(person1_feature, person2_feature)
    # print(siam.item())



if __name__ == '__main__':
    # 检测cuda是否可用
    if torch.cuda.is_available():
        print('-----gpu mode-----')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print('-----cpu mode-----')
    colors_tableau = [(214, 39, 40), (23, 190, 207), (188, 189, 34), (188, 34, 188), (205, 108, 8), (150, 34, 188),
                      (105, 108, 8)]

    # 初始化目标检测网络
    net = SSD()
    net = torch.nn.DataParallel(net)
    net.train(mode=False)
    net.load_state_dict(
        torch.load('./weights/final_20200226_VOC_100000.pth', map_location=lambda storage, loc: storage))

    if torch.cuda.is_available():
        net = net.cuda()
        cudnn.benchmark = True
    img_mean = (104.0, 117.0, 123.0)




    torch.set_grad_enabled(False)
    cfg = None
    if args.retinaface_network == "mobile0.25":
        cfg = cfg_mnet
    elif args.retinaface_network == "resnet50":
        cfg = cfg_re50
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")

    # retinaface net and model
    retinaface_net = RetinaFace(cfg=cfg, phase = 'test')
    retinaface_model = load_model(retinaface_net, args.retinaface_model, args.cpu)
    retinaface_model.eval()
    retinaface_model = retinaface_model.to(device)
    print('Finished loading retinaface model!')
    #print(retinaface_net)

    #arcface net and model
    arcface_net = resnet_face18(False)
    # arcface_model = DataParallel(arcface_net)
    arcface_model = arcface_net
    arcface_model.load_state_dict(torch.load(args.arcface_model), strict=False)
    arcface_model.eval()
    arcface_model.to(device)
    print('Finished loading arcface model!')
    resize = 1

    # 存储已有人脸特征
    #feature_save(arcface_model,args.face_images_path, args.face_features_save_path)
    #exit()
    face_images_path = args.face_images_path
    # 加载已有人脸特征
    persons = []  # 建立人脸库的列表
    persons_name = []
    for person in os.listdir(face_images_path):  # 遍历每一个人脸文件夹
        person_faces = []  # 用来同一个人的不同的人脸特征（一个人获取的可能不止一张照片）
        persons_name.append(person)  # 存放人的名字
        for face in os.listdir(os.path.join(face_images_path, person)):  # 人脸照片转换为特征
            person_image = load_image(os.path.join(face_images_path, person,
                                                   face))  # transform(Image.open(os.path.join(file_path, person, face))).cuda()
            data = torch.from_numpy(person_image)
            data = data.to(torch.device("cuda"))
            # person_feature = net.encode(person_picture[None, ...])  # 获取编码后的每一个人的脸部特征
            output = arcface_model(data)  # 获取特征
            output = output.data.cpu().numpy()
            #print(output.shape)  # 2*512
            fe_1 = output[::2]  # 正面特征
            fe_2 = output[1::2]  # 镜像特征
            person_feature = []
            person_feature = np.hstack((fe_1, fe_2))
            person_feature = person_feature.reshape(1024)
            #print(person_feature)
            # feature = person_feature.detach().cpu()  # 将脸部特征转到CPU上，节省GPU的计算量
            person_faces.append(person_feature)  # 将同一个人脸的人脸特征存放到同一个列表中
        if(person == "gzd"):
            print(person,person_faces)
        # persons_faces[person] = person_face  #
        persons.append([person, person_faces])  # 将不同人的名字、脸部特征存放到同一个列表中
    #print(persons)

    # 保存检测结果的List
    # 眼睛和嘴巴都是，张开为‘1’，闭合为‘0’
    list_B = np.ones(15)  # 眼睛状态List,建议根据fps修改
    list_Y = np.zeros(50)  # 嘴巴状态list，建议根据fps修改
    list_Y1 = np.ones(10)  # 如果在list_Y中存在list_Y1，则判定一次打哈欠，同上，长度建议修改
    blink_count = 0  # 眨眼计数
    yawn_count = 0
    blink_start = time.time()  # 眨眼时间
    yawn_start = time.time()  # 打哈欠时间
    blink_freq = 0.5
    yawn_freq = 0
    time_ = time.time()
    point = []
    frag = True

    maxdiff = 0
    name = " "
    capture = cv2.VideoCapture(0)
    max_fps = 0
    fps = 0.0
    count = 1  # 用于跳帧检测

    while True:
        t1 = time.time()

        if count % 3 != 0:  # 每3帧检测一次
            count += 1
            continue
        for i in range(1):  # range(100)
            '''
            image_path = args.image_path  # "curve/raw_picture/gzd_02.jpg"
            image_name = image_path.split("/")[-1].split(".")[0]
            img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
            '''
            ref, img_raw = capture.read()  # 读取某一帧
            flag_B = True  # 是否闭眼的flag
            flag_Y = False
            num_rec = 0  # 检测到的眼睛的数量
            start = time.time()  # 计时
            img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)  # 格式转变，BGRtoRGB
            img = np.float32(img)
            x = cv2.resize(img_raw, (300, 300)).astype(np.float32)



            x -= img_mean
            x = x.astype(np.float32)
            x = x[:, :, ::-1].copy()
            x = torch.from_numpy(x).permute(2, 0, 1)
            xx = Variable(x.unsqueeze(0))
            if torch.cuda.is_available():
                xx = xx.cuda()
            y = net(xx)
            softmax = nn.Softmax(dim=-1)
            detect = Detect(Config.class_num, 0, 200, 0.01, 0.45)
            priors = utils.default_prior_box()

            loc, conf = y
            loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
            conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

            detections = detect(
                loc.view(loc.size(0), -1, 4),
                softmax(conf.view(conf.size(0), -1, Config.class_num)),
                torch.cat([o.view(-1, 4) for o in priors], 0)
            ).data
            labels = VOC_CLASSES
            top_k = 10

            # 将检测结果放置于图片上
            scale = torch.Tensor(img_raw.shape[1::-1]).repeat(2)
            for i in range(detections.size(1)):

                j = 0
                while detections[0, i, j, 0] >= 0.4:
                    score = detections[0, i, j, 0]
                    label_name = labels[i - 1]
                    if label_name == 'closed_eye':
                        flag_B = False
                    if label_name == 'open_mouth':
                        flag_Y = True
                    display_txt = '%s:%.2f' % (label_name, score)
                    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                    coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
                    color = colors_tableau[i]
                    cv2.rectangle(img_raw, (pt[0], pt[1]), (pt[2], pt[3]), color, 2)

                    cv2.putText(img_raw, display_txt, (int(pt[0]), int(pt[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 255, 255),
                                1, 8)
                    j += 1
                    num_rec += 1
            if num_rec > 0:
                if flag_B:
                    # print(' 1:eye-open')
                    list_B = np.append(list_B, 1)  # 睁眼为‘1’
                else:
                    # print(' 0:eye-closed')
                    list_B = np.append(list_B, 0)  # 闭眼为‘0’
                list_B = np.delete(list_B, 0)
                if flag_Y:
                    list_Y = np.append(list_Y, 1)
                else:
                    list_Y = np.append(list_Y, 0)
                list_Y = np.delete(list_Y, 0)
            else:
                print('nothing detected')
            # print(list)
            # 实时计算PERCLOS
            perclos = 1 - np.average(list_B)
            # print('perclos={:f}'.format(perclos))
            if list_B[13] == 1 and list_B[14] == 0:
                # 如果上一帧为’1‘，此帧为’0‘则判定为眨眼
                print('----------------眨眼----------------------')
                blink_count += 1
            blink_T = time.time() - blink_start
            if blink_T > 10:
                # 每10秒计算一次眨眼频率
                blink_freq = blink_count / blink_T
                blink_start = time.time()
                blink_count = 0
            # print('blink_freq={:f}'.format(blink_freq))
            # 检测打哈欠
            # if Yawn(list_Y,list_Y1):
            if (list_Y[len(list_Y) - len(list_Y1):] == list_Y1).all():
                print('----------------------打哈欠----------------------')
                yawn_count += 1
                list_Y = np.zeros(50)
            # 计算打哈欠频率
            yawn_T = time.time() - yawn_start
            if yawn_T > 60:
                yawn_freq = yawn_count / yawn_T
                yawn_start = time.time()
                yawn_count = 0
            print('yawn_freq={:.4f}'.format(yawn_freq))

            # 此处为判断疲劳部分
            '''
            想法1：最简单，但是太影响实时性
            if(perclos>0.4 or blink_freq<0.25 or yawn_freq>5/60):
                print('疲劳')
                if(blink_freq<0.25)
            else:
                print('清醒')
            '''
            # 想法2：
            if (perclos > 0.4):
                print('疲劳')
            elif (blink_freq < 0.25):
                print('疲劳')
                blink_freq = 0.5  # 如果因为眨眼频率判断疲劳，则初始化眨眼频率
            elif (yawn_freq > 5.0 / 60):
                print("疲劳")
                yawn_freq = 0  # 初始化，同上
            else:
                print('清醒')

            T = time.time() - start
            fps = 1 / T  # 实时在视频上显示fps
            if fps > max_fps:
                max_fps = fps
            fps_txt = 'fps:%.2f' % (fps)
            cv2.putText(img_raw, fps_txt, (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, 8)
            #cv2.imshow("ssd", img_raw)
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)  # 格式转变，BGRtoRGB
            img = np.float32(img_raw)
            #print(img.shape)
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)  # 对图片进行转置处理
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            tic = time.time()
            loc, conf, landms = retinaface_model(img)  # forward pass
            print('net forward time: {:.4f}'.format(time.time() - tic))

            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, args.nms_threshold)
            #keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:args.keep_top_k, :]
            landms = landms[:args.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)
            # 到这里为止我们已经利用Retinaface_pytorch的预训练模型检测完了人脸，并获得了人脸框和人脸五个特征点的坐标信息，全保存在dets中，接下来为人脸剪切部分

            for num, b in enumerate(dets):  # dets中包含了人脸框和五个特征点的坐标
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw,text+"_"+str(num), (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                # landms，在人脸上画出特征点，要是你想保存不显示特征点的人脸图，你可以把这里注释掉
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

                # 计算人脸框矩形大小
                Height = b[3] - b[1]
                Width = b[2] - b[0]

                '''
                # 显示人脸矩阵大小
                print("人脸数 / faces in all:", str(num + 1), "\n")
                print("窗口大小 / The size of window:"
                    , '\n', "高度 / height:", Height
                    , '\n', "宽度 / width: ", Width)
                '''

                # 根据人脸框大小，生成空白的图片
                img_blank = np.zeros((Height, Width, 3), np.uint8)
                # 将人脸填充到空白图片
                for h in range(Height):
                    for w in range(Width):
                        img_blank[h][w] = img_raw[b[1] + h][b[0] + w]

                #image_input就是要输入arcface的图片

                image_input = cv2.cvtColor(img_blank, cv2.COLOR_BGR2GRAY)
                image_input = image_format(img_blank)
                data = torch.from_numpy(image_input)
                data = data.to(torch.device("cuda"))
                output = arcface_model(data)  # 获取特征
                output = output.data.cpu().numpy()
                #print(output.shape)
                fe_1 = output[::2]  # 正面特征
                fe_2 = output[1::2]  # 镜像特征
                person_feature_x = []
                person_feature_x = np.hstack((fe_1, fe_2))
                #print(person_feature_x.shape)
                person_feature_x = person_feature_x.reshape(1024)
                #print(person_feature_x.shape)
                #得到当前人脸特征person_feature_x

                #与已有人脸特征比较
                maxdiff = 0
                persons_similarity = []
                for person_faces in persons:
                    person_name = person_faces[0]
                    #print(person_name)
                    person_features = person_faces[1]
                    for person_feature in person_features:
                        diff = cosin_metric(person_feature_x,person_feature)
                        if(diff>maxdiff):
                            maxdiff = diff
                            name = person_name
                            print(name, maxdiff)
                            persons_similarity.append([person_name, diff])

                #     siam = compare(person_feature, personal_features)
                #     sia = max(siam[0]).item()
                #     persons_similarity.append([personal_name, sia])
                # data = pd.DataFrame(persons_similarity)
                # data = data.sort_values(by=1, ascending=False)
                # obj_name = data.iloc[0][0]
        print(name,maxdiff)
        # cv2.putText(img_raw, (" 相似度:%.2f" % maxdiff), (60, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0))  # 相似度
        cv2.putText(img_raw, str("who :" + name), (500, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0))  # 展示who
        t7 = time.time()
        fps = 1 / (t7 - t1)
        cv2.putText(img_raw, str("fps :%.2f" % fps), (0, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0))  # 展示帧率
        # print("fps    :", 1 / (t7 - t1))
        img_raw = cv2.cvtColor(np.asarray(img_raw), cv2.COLOR_RGB2BGR)
        cv2.imshow("detect_face", np.uint8(img_raw))

        # 点击小写字母q 退出程序
        if cv2.waitKey(1) == ord('q'):
            break

        # 点击窗口关闭按钮退出程序
        if cv2.getWindowProperty('detect_face', cv2.WND_PROP_AUTOSIZE) < 1:
            break
    capture.release()
    cv2.destroyAllWindows()

    '''
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 格式转变，BGRtoRGB
        frame, boxes = retinaface.detect_image(frame)  # 接收返回的变量
    frame = Image.fromarray(np.uint8(frame))

        for box in boxes:  # 提取检测到的人脸的四个坐标值
            box = list(map(int, box))
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            frame = Image.fromarray(np.uint8(frame))  # 从numpy转为PIL类型
            cropped = frame.crop((x1, y1, x2, y2))

            #person1 = tf(cropped).cuda()  # 将MTCNN裁剪出来的图片归一化并且传入cuda
            #person1_feature = net.encode(person1[None, ...])  # 获取到处理后的视频人脸的特征

            siam_last = 0
            name = 0
            for i in range(num):
                person2_feature = featuress[i].cuda()
                siam = compare(person1_feature, person2_feature)
                if siam > siam_last:  # 如果此时的当前的相似度大于上一个特征的相似度，则从字典中取出当前对应的人的名字（按所有特征中的相似度最大的那个算）
                    siam_last = siam
                    name = dic[featuress[i]]

            frame = np.asarray(frame)  # 从PIL转为numpy格式
            cv2.putText(frame, name + str(float("%.2f" % siam_last.detach().cpu())), (x1, y1 + 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            count += 1

        t7 = time.time()
        fps = 1 / (t7 - t1)
        cv2.putText(frame, str("fps :%.2f" % fps), (0, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0))  # 展示帧率
        # print("fps    :", 1 / (t7 - t1))

        frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        cv2.imshow("video", np.uint8(frame))
        c = cv2.waitKey(1) & 0xff
        
    '''

