from __future__ import print_function
import os
import argparse
import torch
from torch.autograd import *
import torch.backends.cudnn as cudnn
import pickle
import numpy as np
import cv2

import time
from SSD import utils
from SSD.voc0712 import *
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from SSD.ssd_net_vgg import *
from SSD.detection import *
import SSD.Config as Config

from utils.nms.py_cpu_nms import py_cpu_nms

from models.retinaface import RetinaFace
from models.resnet import resnet_face18
from torch.nn import DataParallel
from utils.box_utils import decode, decode_landm


from models.fsanet.FSANET_model import *
from keras.layers import Average
from math import cos, sin

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--retinaface_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--arcface_model', default='./weights/resnet18_110.pth',#./weights/resnet18_110.pth./weights/ms1mv3_arcface_r18_fp16/backbone.pth
                    type=str, help='Trained arcface state_dict file path to open')
parser.add_argument('--retinaface_network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--arcface_network',default='mobile0.25', help='Backbone network mobile0.25 or resnet50')

parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.4, type=float, help='confidence_threshold')#default=0.02
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
parser.add_argument('--face_features_save_name', default='/face_features_0', type=str, help=' save face_features name')
parser.add_argument('--store_face', action ="store_true", default =False, help = 'store_images')

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
    # doc = open(save_data, "rb")
    # persons_faces = pickle.load(doc)  # ??????????????????
    doc = open(save_data, "ab+")
    persons = []
    persons_faces = []  # ????????????????????????
    for person in os.listdir(face_images_path):  # ??????????????????????????????
        person_faces = []  # ??????????????????????????????????????????????????????????????????????????????????????????
        # persons_name.append(person)  # ??????????????????
        for face in os.listdir(os.path.join(face_images_path, person)):  # ???????????????????????????
            person_image = load_image(os.path.join(face_images_path, person, face))  # transform(Image.open(os.path.join(file_path, person, face))).cuda()
            data = torch.from_numpy(person_image)
            #data = data.to(???cpu???)#torch.device("cuda")
            # person_feature = net.encode(person_picture[None, ...])  # ?????????????????????????????????????????????
            output = arcface_model(data)  # ????????????
            output = output.data.cpu().numpy()
            # print(output.shape)  # 2*512
            fe_1 = output[::2]  # ????????????
            fe_2 = output[1::2]  # ????????????
            person_feature = []
            person_feature = np.hstack((fe_1, fe_2))
            person_feature = person_feature.reshape(1024)
            # print(person_feature)
            # feature = person_feature.detach().cpu()  # ?????????????????????CPU????????????GPU????????????
            person_faces.append(person_feature)  # ????????????????????????????????????????????????????????????
        # persons_faces[person] = person_face  #
        persons.append([person, person_faces])  # ???????????????????????????????????????????????????????????????
    for person in os.listdir(face_images_path):  # ??????????????????????????????
        person_face = []  # ??????????????????????????????????????????????????????????????????????????????????????????
        # persons_name.append(person)  # ??????????????????
        for face in os.listdir(os.path.join(face_images_path, person)):  # ???????????????????????????
            person_image = load_image(os.path.join(face_images_path, person, face))#transform(Image.open(os.path.join(file_path, person, face))).cuda()
            data = torch.from_numpy(person_image)
            data = data.to(torch.device("cpu" if args.cpu else "cuda"))
            #person_feature = net.encode(person_picture[None, ...])  # ?????????????????????????????????????????????
            output = arcface_model(data)  # ????????????
            output = output.data.cpu().numpy()
            #print(output.shape)

            fe_1 = output[::2]#????????????
            fe_2 = output[1::2]#????????????
            # print(fe_1.shape,fe_2.shape)
            person_feature = []
            person_feature = np.hstack((fe_1, fe_2))
            person_feature = person_feature.reshape(1024)
            #feature = person_feature.detach().cpu()  # ?????????????????????CPU????????????GPU????????????
            person_face.append(person_feature)  # ??????????????????????????????????????????????????????????????????
        # persons_faces[person] = person_face  #
        #persons_faces.append([person, torch.cat(person_face, dim=0)])  # ???????????????????????????????????????????????????????????????
        persons_faces.append([person, person_face])  # ???????????????????????????????????????????????????????????????
    print(persons_faces)
    print(persons)
    pickle.dump(persons_faces, doc)  # ??????????????????????????????
    doc = open(save_data, "rb")
    persons_faces = pickle.load(doc)  # ??????????????????
    print(persons_faces)

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
    image = cv2.imread(img_path)
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

def image_format(image):
    if image is None:
        return None
    #image = cv2_letterbox_image(image,128)
    #print('image:', image)
    print('image.shape:' ,image.shape)
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image
#??????????????????
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
    output = arcface_model(data)  # ????????????
    output = output.data.cpu().numpy()

    # ????????????????????? ?????????
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
    output = arcface_model(data)  # ????????????
    output = output.data.cpu().numpy()

    fe_1 = output[::2]
    fe_2 = output[1::2]
    person2_feature = np.hstack((fe_1, fe_2))
    person2_feature = person2_feature.reshape(1024)
    diff = cosin_metric(person1_feature, person2_feature)
    print("diff:", diff)
    # siam = compare(person1_feature, person2_feature)
    # print(siam.item())

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 50):
    print("yaw:", yaw)
    print("roll:", roll)
    print("pitch:", pitch)
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img


def draw_headpose(detected, input_img, faces, ad, img_size, img_w, img_h, model):
    # loop over the detections
    if detected.shape[2] > 0:
        for i in range(0, detected.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detected[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the face and extract the face ROI
                (h0, w0) = input_img.shape[:2]
                box = detected[0, 0, i, 3:7] * np.array([w0, h0, w0, h0])
                (startX, startY, endX, endY) = box.astype("int")
                # print((startX, startY, endX, endY))
                x1 = startX
                y1 = startY
                w = endX - startX
                h = endY - startY

                x2 = x1 + w
                y2 = y1 + h

                xw1 = max(int(x1 - ad * w), 0)
                yw1 = max(int(y1 - ad * h), 0)
                xw2 = min(int(x2 + ad * w), img_w - 1)
                yw2 = min(int(y2 + ad * h), img_h - 1)

                faces[i, :, :, :] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                faces[i, :, :, :] = cv2.normalize(faces[i, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

                face = np.expand_dims(faces[i, :, :, :], axis=0)
                p_result = model.predict(face)

                face = face.squeeze()
                img = draw_axis(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], p_result[0][0], p_result[0][1], p_result[0][2])

                # input_img[yw1:yw2 + 1, xw1:xw2 + 1, :] = img
    return input_img  # ,time_network,time_plot

def draw_headpose_det(dets, input_img, faces, ad, img_size, img_w, img_h, model):
    for i, det in enumerate(dets):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        # compute the (x, y)-coordinates of the bounding box for
        # the face and extract the face ROI
        (h0, w0) = input_img.shape[:2]
        det = list(map(int, det))
        (x1, y1, x2, y2) = det[0], det[1], det[2], det[3]
        # print((startX, startY, endX, endY))
        w = x2 - x1
        h = y2 - y1

        xw1 = max(int(x1 - ad * w), 0)
        yw1 = max(int(y1 - ad * h), 0)
        xw2 = min(int(x2 + ad * w), img_w - 1)
        yw2 = min(int(y2 + ad * h), img_h - 1)

        faces[i, :, :, :] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
        faces[i, :, :, :] = cv2.normalize(faces[i, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        face = np.expand_dims(faces[i, :, :, :], axis=0)
        p_result = model.predict(face)

        face = face.squeeze()
        img = draw_axis(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], p_result[0][0], p_result[0][1], p_result[0][2])
        yaw_txt = 'yaw:%.2f' % p_result[0][0]
        roll_txt = 'roll:%.2f' % p_result[0][1]
        pitch_txt = 'pitch:%.2f' % p_result[0][2]
        cx = x2
        cy = y1
        cv2.putText(img_raw, yaw_txt, (cx-70, cy+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, 8)
        cv2.putText(img_raw, roll_txt, (cx-70, cy+24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, 8)
        cv2.putText(img_raw, pitch_txt, (cx-70, cy+36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, 8)
        # input_img[yw1:yw2 + 1, xw1:xw2 + 1, :] = imglengt
    return input_img  # ,time_network,time_plot


if __name__ == '__main__':
    # # ??????cuda????????????
    if torch.cuda.is_available():
        print('-----gpu mode-----')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print('-----cpu mode-----')
    colors_tableau = [(214, 39, 40), (23, 190, 207), (188, 189, 34), (188, 34, 188), (205, 108, 8), (150, 34, 188), (105, 108, 8)]

    # # arcface net and model
    # device = torch.device("cpu" if args.cpu else "cuda")
    # arcface_net = resnet_face18(False)
    # #arcface_model = DataParallel(arcface_net)
    # arcface_model = arcface_net
    # arcface_model.load_state_dict(torch.load(args.arcface_model), strict=False)
    # arcface_model.to(device)
    # arcface_model.eval()
    #
    # print('Finished loading arcface model!')
    #
    # face_images_path = args.face_images_path
    # # ????????????????????????
    # persons = []  # ????????????????????????
    # persons_name = []
    # for person in os.listdir(face_images_path):  # ??????????????????????????????
    #     person_faces = []  # ??????????????????????????????????????????????????????????????????????????????????????????
    #     # persons_name.append(person)  # ??????????????????
    #     for face in os.listdir(os.path.join(face_images_path, person)):  # ???????????????????????????
    #         person_image = load_image(os.path.join(face_images_path, person,
    #                                                face))  # transform(Image.open(os.path.join(file_path, person, face))).cuda()
    #
    #         data = torch.from_numpy(person_image)
    #         data = data.to(torch.device("cpu" if args.cpu else "cuda"))
    #         # person_feature = net.encode(person_picture[None, ...])  # ?????????????????????????????????????????????
    #         output = arcface_model(data)  # ????????????
    #         output = output.data.cpu().numpy()
    #         # print(output.shape)  # 2*512
    #         fe_1 = output[::2]  # ????????????
    #         fe_2 = output[1::2]  # ????????????
    #         # person_feature = []
    #         person_feature = np.hstack((fe_1, fe_2))
    #         person_feature = person_feature.reshape(1024)
    #         # print(person_feature)
    #         # feature = person_feature.detach().cpu()  # ?????????????????????CPU????????????GPU????????????
    #         person_faces.append(person_feature)  # ????????????????????????????????????????????????????????????
    #     # persons_faces[person] = person_face  #
    #     persons.append([person, person_faces])  # ???????????????????????????????????????????????????????????????
    # print(persons)
    # exit()


    # ?????????????????????????????????
    # load model and weights
    img_size = 64 #
    stage_num = [3, 3, 3]
    lambda_local = 1
    lambda_d = 1
    img_idx = 0
    detected = ''  # make this not local variable
    time_detection = 0
    time_network = 0
    time_plot = 0
    skip_frame = 5  # every 5 frame do 1 detection and network forward propagation
    ad = 0.6

    # Parameters
    num_capsule = 3
    dim_capsule = 16
    routings = 2
    stage_num = [3, 3, 3]
    lambda_d = 1
    num_classes = 3
    image_size = 64
    num_primcaps = 7 * 3 # ??????Scoring Function ???????????????7???c?????????
    m_dim = 5
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model1 = FSA_net_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    model2 = FSA_net_Var_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

    num_primcaps = 8 * 8 * 3 # ?????????Scoring Function
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model3 = FSA_net_noS_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

    print('Loading models ...')

    weight_file1 = './weights/fsanet_pre-trained/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
    model1.load_weights(weight_file1)
    print('Finished loading model 1.')

    weight_file2 = './weights/fsanet_pre-trained/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
    model2.load_weights(weight_file2)
    print('Finished loading model 2.')

    weight_file3 = './weights/fsanet_pre-trained/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
    model3.load_weights(weight_file3)
    print('Finished loading model 3.')

    inputs = Input(shape=(64, 64, 3))
    x1 = model1(inputs)  # 1x1  1*1??????
    x2 = model2(inputs)  # var  ??????
    x3 = model3(inputs)  # w/o  ???
    avg_model = Average()([x1, x2, x3])
    model = Model(inputs=inputs, outputs=avg_model)

    # # load our serialized face detector from disk
    # print("[INFO] loading face detector...")
    # protoPath = os.path.sep.join(["weights",
    #                               "fasnet_face_detector",
    #                               "deploy.prototxt"])
    # modelPath = os.path.sep.join(["weights",
    #                               "fasnet_face_detector",
    #                               "res10_300x300_ssd_iter_140000.caffemodel"])
    # fsa_facedetector_net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # ???????????????????????????
    SSD_net = SSD()
    SSD_net = torch.nn.DataParallel(SSD_net)
    SSD_net.train(mode=False)
    SSD_net.load_state_dict(
        torch.load('./weights/final_20200226_VOC_100000.pth', map_location=lambda storage, loc: storage))
    if torch.cuda.is_available():
        SSD_net = SSD_net.cuda()
        cudnn.benchmark = True
    img_mean = (104.0, 117.0, 123.0)
    print('Finished loading SSD model!')

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
    resize = 1



    #arcface net and model
    arcface_net = resnet_face18(False)
    #arcface_model = DataParallel(arcface_net)
    arcface_model = arcface_net
    arcface_model.load_state_dict(torch.load(args.arcface_model), strict=False)
    arcface_model.to(device)
    arcface_model.eval()


    print('Finished loading arcface model!')



    face_images_path = args.face_images_path
    # ????????????????????????
    persons = []  # ????????????????????????
    persons_name = []
    for person in os.listdir(face_images_path):  # ??????????????????????????????
        person_faces = []  # ??????????????????????????????????????????????????????????????????????????????????????????
        #persons_name.append(person)  # ??????????????????
        for face in os.listdir(os.path.join(face_images_path, person)):  # ???????????????????????????
            person_image = load_image(os.path.join(face_images_path, person,
                                                   face))  # transform(Image.open(os.path.join(file_path, person, face))).cuda()

            data = torch.from_numpy(person_image)
            data = data.to(torch.device("cpu" if args.cpu else "cuda"))
            # person_feature = net.encode(person_picture[None, ...])  # ?????????????????????????????????????????????
            output = arcface_model(data)  # ????????????
            output = output.data.cpu().numpy()
            #print(output.shape)  # 2*512
            fe_1 = output[::2]  # ????????????
            fe_2 = output[1::2]  # ????????????
            #person_feature = []
            person_feature = np.hstack((fe_1, fe_2))
            person_feature = person_feature.reshape(1024)
            #print(person_feature)
            # feature = person_feature.detach().cpu()  # ?????????????????????CPU????????????GPU????????????
            person_faces.append(person_feature)  # ????????????????????????????????????????????????????????????
        # persons_faces[person] = person_face  #
        persons.append([person, person_faces])  # ???????????????????????????????????????????????????????????????
    print(persons)
    # ????????????????????????
    # feature_save(arcface_model, args.face_images_path, args.face_features_save_path)
    # exit()
    '''
    [['gzd', [array([-0.02775349, -0.01238428, -0.00944023, ...,  0.08134101,
       -0.03249797, -0.12846223], dtype=float32), array([-0.0286944 , -0.03361573,  0.00755915, ...,  0.07978991,
       -0.03391185, -0.0696403 ], dtype=float32), array([ 0.00073069, -0.04353086, -0.00838107, ...,  0.0480972 ,
        0.0194932 , -0.10387032], dtype=float32), array([-0.00602166, -0.09596573, -0.04411527, ...,  0.06135169,
       -0.00455882, -0.10979676], dtype=float32)]], ['xz', [array([-0.01870626,  0.00946164,  0.01530436, ...,  0.13674383,
       -0.02859906, -0.12919042], dtype=float32)]], ['ztb', [array([-0.0344049 , -0.01772376, -0.0315088 , ...,  0.09661521,
        0.02863654, -0.0538289 ], dtype=float32), array([ 0.00725023, -0.02575168, -0.05472682, ...,  0.07009535,
        0.02719895, -0.10241977], dtype=float32), array([-0.01226433, -0.06378665, -0.03950522, ...,  0.09895502,
       -0.01279964, -0.09516567], dtype=float32)]]]
    [['gzd', [array([-0.18953685,  0.25127172,  0.44318914, ...,  0.26041853,
        0.00556347, -0.09722402], dtype=float32), array([-0.08858509,  0.40954152,  0.33340368, ...,  0.2609753 ,
       -0.22737163, -0.09730398], dtype=float32), array([ 0.06697167,  0.1673652 ,  0.4466713 , ...,  0.3534699 ,
       -0.02927328, -0.15480004], dtype=float32), array([ 0.02910094,  0.26345947,  0.16575542, ...,  0.21317184,
       -0.15785839, -0.10570662], dtype=float32)]], ['xz', [array([ 1.2904107e-04,  1.9026779e-01,  9.7813323e-02, ...,
        4.9653128e-02, -2.2928578e-01,  5.6012046e-02], dtype=float32)]], ['ztb', [array([ 0.09449671,  0.25402218,  0.19888894, ...,  0.1691326 ,
        0.07427573, -0.18209465], dtype=float32), array([ 0.08949346,  0.07748014,  0.20857541, ...,  0.0628132 ,
        0.04524595, -0.25275803], dtype=float32), array([ 0.11823017,  0.20543633,  0.24783547, ...,  0.20919316,
        0.06901397, -0.07464402], dtype=float32)]]]
    [['gzd', [array([-0.18953685,  0.25127172,  0.44318914, ...,  0.26041853,
        0.00556347, -0.09722402], dtype=float32), array([-0.08858509,  0.40954152,  0.33340368, ...,  0.2609753 ,
       -0.22737163, -0.09730398], dtype=float32), array([ 0.06697167,  0.1673652 ,  0.4466713 , ...,  0.3534699 ,
       -0.02927328, -0.15480004], dtype=float32), array([ 0.02910094,  0.26345947,  0.16575542, ...,  0.21317184,
       -0.15785839, -0.10570662], dtype=float32)]], ['xz', [array([ 1.2904107e-04,  1.9026779e-01,  9.7813323e-02, ...,
        4.9653128e-02, -2.2928578e-01,  5.6012046e-02], dtype=float32)]], ['ztb', [array([ 0.09449671,  0.25402218,  0.19888894, ...,  0.1691326 ,
        0.07427573, -0.18209465], dtype=float32), array([ 0.08949346,  0.07748014,  0.20857541, ...,  0.0628132 ,
        0.04524595, -0.25275803], dtype=float32), array([ 0.11823017,  0.20543633,  0.24783547, ...,  0.20919316,
        0.06901397, -0.07464402], dtype=float32)]]]
    [['gzd', [array([ 0.01101855,  0.01160288, -0.03547082, ..., -0.12146265,
        0.02234442,  0.00050762], dtype=float32), array([-0.02148875,  0.0030162 , -0.01124092, ..., -0.02582328,
       -0.00773485,  0.01521705], dtype=float32), array([ 0.02365636,  0.02082416, -0.04609686, ..., -0.0935097 ,
       -0.0038598 , -0.01387273], dtype=float32), array([-0.01511925, -0.01434538,  0.02279602, ..., -0.08121396,
        0.00682687,  0.03347363], dtype=float32)]], ['xz', [array([ 0.02226446, -0.02818697, -0.01003379, ..., -0.07193639,
        0.07014387, -0.03628271], dtype=float32)]], ['ztb', [array([-0.00616407, -0.00902649, -0.03098095, ..., -0.0565353 ,
        0.00480253,  0.02953284], dtype=float32), array([-0.00192584,  0.01550587, -0.00236001, ..., -0.07215291,
        0.0133029 , -0.02598327], dtype=float32), array([-0.02350364, -0.00817247, -0.01091156, ..., -0.06054008,
        0.01472112,  0.00472352], dtype=float32)]]]
    [['gzd', [array([ 0.09051416, -0.01777817,  0.04139188, ..., -0.07996829,
       -0.16045465, -0.04303788], dtype=float32), array([ 0.03992381, -0.01211053, -0.00352772, ..., -0.04422708,
       -0.08078621, -0.02579249], dtype=float32), array([ 0.045525  , -0.02676153,  0.01660296, ..., -0.02378191,
       -0.05913594,  0.04412578], dtype=float32), array([ 0.04933872,  0.01073836,  0.05539575, ..., -0.06883282,
       -0.07300425, -0.03862544], dtype=float32)]], ['xz', [array([ 0.09568006, -0.02096574, -0.00904692, ..., -0.0383738 ,
       -0.09290529,  0.00380711], dtype=float32)]], ['ztb', [array([ 0.07488263,  0.00200886,  0.04021835, ..., -0.06697144,
       -0.12475453,  0.00546473], dtype=float32), array([ 0.06989961,  0.02497304,  0.0296509 , ..., -0.08306047,
       -0.12402482, -0.0111357 ], dtype=float32), array([ 0.07987466,  0.01392394,  0.02962763, ..., -0.06169514,
       -0.11828618, -0.02213287], dtype=float32)]]]
    [['gzd', [array([ 0.04213404, -0.0253459 , -0.04876185, ...,  0.03942836,
       -0.05263235, -0.02335209], dtype=float32), array([ 0.03830254, -0.01540274, -0.04397659, ...,  0.0516267 ,
       -0.00858263, -0.00116498], dtype=float32), array([ 0.01451938, -0.01980229,  0.00374373, ...,  0.06936742,
       -0.01634569,  0.00850963], dtype=float32), array([ 0.05374377,  0.00201504, -0.05847294, ...,  0.04271624,
       -0.04611531,  0.00365958], dtype=float32)]], ['xz', [array([ 0.04851374,  0.00360618, -0.05070179, ...,  0.10088143,
       -0.06129668, -0.02319437], dtype=float32)]], ['ztb', [array([ 0.06156676, -0.01057883, -0.04503519, ...,  0.08676384,
       -0.05934704, -0.00422611], dtype=float32), array([ 0.05925316, -0.00801994, -0.06470278, ...,  0.04954056,
       -0.05348888, -0.01057992], dtype=float32), array([ 0.05257639,  0.02088886, -0.03644251, ...,  0.07505848,
       -0.08138286, -0.00753461], dtype=float32)]]]
    '''
    # ?????????????????????List
    # ????????????????????????????????????1??????????????????0???
    list_B = np.ones(15)  # ????????????List,????????????fps??????
    list_Y = np.zeros(50)  # ????????????list???????????????fps??????
    list_Y1 = np.ones(10)  # ?????????list_Y?????????list_Y1?????????????????????????????????????????????????????????
    blink_count = 0  # ????????????
    yawn_count = 0
    blink_start = time.time()  # ????????????
    yawn_start = time.time()  # ???????????????
    blink_freq = 0.5
    yawn_freq = 0
    time_ = time.time()
    point = []
    frag = True

    maxdiff = 0
    name = " "
    # videopath = './data/test_video/ztb.mp4'
    videopath = 0
    capture = cv2.VideoCapture(videopath)
    # capture = cv2.VideoCapture(videopath, cv2.CAP_DSHOW)
    #capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,  1024* 1)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 768* 1)
    max_fps = 0
    fps = 0.0
    count = 0  # ??????????????????

    while True:
        start_time = time.time()
        count += 1
        if count % 3 != 0:  # ???3???????????????
            continue
        for i in range(1):  # range(100)
            '''
            image_path = args.image_path  # "curve/raw_picture/gzd_02.jpg"
            image_name = image_path.split("/")[-1].split(".")[0]
            img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
            '''
            ref, img_raw = capture.read()  # ???????????????
            if(img_raw.shape == 0):
                continue


            #????????????--------------------------------------------------------------------------
            # T = time.time() - start
            # fps = 1 / T  # ????????????????????????fps
            # if fps > max_fps:
            #     max_fps = fps
            # fps_txt = 'fps:%.2f' % (fps)
            #cv2.putText(img_raw, fps_txt, (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, 8)
            #cv2.imshow("ssd", img_raw)
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)  # ???????????????BGRtoRGB
            img = np.float32(img_raw)
            #print(img.shape)
            img_height, img_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)  # ???????????????????????????
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            tic = time.time()
            loc, conf, landms = retinaface_model(img)  # forward pass
            #print('net forward time: {:.4f}'.format(time.time() - tic))

            priorbox = PriorBox(cfg, image_size=(img_height, img_width))
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
            # ?????????????????????????????????Retinaface_pytorch??????????????????????????????????????????????????????????????????????????????????????????????????????????????????dets????????????????????????????????????
            print("dets:", dets)







            #????????????---------------------------------------------------------------------------------------------
            for num, b in enumerate(dets):  # dets????????????????????????????????????????????????
                if b[4] < args.vis_thres:
                    continue;
                vis_thres_text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                # ???????????????????????????
                Height = b[3] - b[1]
                Width = b[2] - b[0]
                '''
                # ????????????????????????
                print("????????? / faces in all:", str(num + 1), "\n")
                print("???????????? / The size of window:"
                    , '\n', "?????? / height:", Height
                    , '\n', "?????? / width: ", Width)
                '''
                # ?????????????????????????????????????????????
                img_cut = np.zeros((Height, Width, 3), np.uint8)
                # ??????????????????????????????
                # for h in range(Height):
                #     for w in range(Width):
                #         img_cut[h][w] = img_raw[b[1] + h][b[0] + w]
                img_cut = img_raw[b[1]:b[3], b[0]:b[2]]
                cut_path = "gzdstore_face/"  # ??????????????????????????????????????????????????? ???????????????????????????
                if args.store_face:
                    if not os.path.exists(cut_path):
                        os.makedirs(cut_path)
                    if cv2.waitKey(1) == ord('p'):
                        cv2.imwrite(cut_path + 'storeface' + "_" + str(count + 1) + ".jpg",
                                    img_cut)  # ???????????????????????????????????????
                        print("Save into:", cut_path + 'storeface' + "_" + str(count + 1) + ".jpg")




                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw,vis_thres_text+"_"+str(num), (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                # landms???????????????????????????????????????????????????????????????????????????????????????????????????????????????
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)



                #image_input???????????????arcface?????????

                #image_input = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)
                image_input = image_format(img_cut)

                data = torch.from_numpy(image_input)
                data = data.to(torch.device("cpu" if args.cpu else "cuda"))
                output = arcface_model(data)  # ????????????
                output = output.data.cpu().numpy()
                #print(output.shape)
                fe_1 = output[::2]  # ????????????
                fe_2 = output[1::2]  # ????????????
                person_feature_x = []
                person_feature_x = np.hstack((fe_1, fe_2))
                print('person_feature_x.shape:', person_feature_x.shape)
                person_feature_x = person_feature_x.reshape(1024)
                print('person_feature_x.shape:', person_feature_x.shape)
                #????????????????????????person_feature_x
                #exit()

                #???????????????????????????
                maxdiff = 0
                persons_similarity = []
                save_data = args.face_features_save_path + args.face_features_save_name
                # doc = open(save_data, "rb")
                # person_s = pickle.load(doc)  # ??????????????????
                # print(person_s)
                # print(persons)
                for person_faces in persons:
                    person_name = person_faces[0]
                    #print(person_name)
                    person_features = person_faces[1]
                    for person_feature in person_features:
                        diff = cosin_metric(person_feature_x,person_feature)
                        # print(person_name, diff)
                        if(diff>maxdiff):
                            maxdiff = diff
                            name = person_name
                            persons_similarity.append([person_name, diff])
            # print(name, maxdiff)
            # cv2.putText(img_raw, (" ?????????:%.2f" % maxdiff), (60, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0))  # ?????????

            flag_B = True  # ???????????????flag
            flag_Y = False
            num_rec = 0  # VOC_CLASSES?????????????????????????????????
            start = time.time()  # ??????
            x = cv2.resize(img_raw, (300, 300)).astype(np.float32)

            x -= img_mean
            x = x.astype(np.float32)
            x = x[:, :, ::-1].copy()
            x = torch.from_numpy(x).permute(2, 0, 1)
            xx = Variable(x.unsqueeze(0))
            if torch.cuda.is_available():
                xx = xx.cuda()
            y = SSD_net(xx)
            # print(y)
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
            # print(detections)
            # ?????????????????????????????????
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
                    list_B = np.append(list_B, 1)  # ????????????1???
                else:
                    # print(' 0:eye-closed')
                    list_B = np.append(list_B, 0)  # ????????????0???
                list_B = np.delete(list_B, 0)  # ???????????????????????????list
                if flag_Y:
                    list_Y = np.append(list_Y, 1)  # ????????????1???
                else:
                    list_Y = np.append(list_Y, 0)  # ????????????0???
                list_Y = np.delete(list_Y, 0)  # ???????????????????????????list
            else:
                print('nothing detected')
            # print(list)
            # ????????????PERCLOS
            perclos = 1 - np.average(list_B)
            # print('perclos={:f}'.format(perclos))
            if list_B[13] == 1 and list_B[14] == 0:
                # ?????????????????????1??????????????????0?????????????????????
                print('----------------??????----------------------')
                blink_count += 1
            blink_T = time.time() - blink_start
            if blink_T > 10:
                # ???10???????????????????????????
                blink_freq = blink_count / blink_T
                blink_start = time.time()
                blink_count = 0
            # print('blink_freq={:f}'.format(blink_freq))
            # ???????????????
            # if Yawn(list_Y,list_Y1):
            if (list_Y[len(list_Y) - len(list_Y1):] == list_Y1).all():
                print('----------------------?????????----------------------')
                yawn_count += 1
                list_Y = np.zeros(50)
            # ?????????????????????
            yawn_T = time.time() - yawn_start
            if yawn_T > 60:
                yawn_freq = yawn_count / yawn_T
                yawn_start = time.time()
                yawn_count = 0
            # print('yawn_freq={:.4f}'.format(yawn_freq))

            # ??????????????????-------------------------------------------------------------
            if (perclos > 0.4):
                print('??????')
            elif (blink_freq < 0.25):
                print('??????')
                blink_freq = 0.5  # ???????????????????????????????????????????????????????????????
            elif (yawn_freq > 5.0 / 60):
                print("??????")
                yawn_freq = 0  # ??????????????????
            else:
                print('??????')
            # ??????????????????-------------------------------------------------------------




            # ??????????????????-----------------------------------------------------------------------
            print('Start detecting pose ...')
            print("dets.shape[0]:", dets.shape[0])
            faces = np.empty((dets.shape[0], img_size, img_size, 3))  # detected.shape[2]????????????????????? dets.shape[2]
            draw_headpose_det(dets, img_raw, faces, ad, img_size, img_width, img_height, model)
            # ??????????????????-----------------------------------------------------------------------

            # # ??????????????????-----------------------------------------------------------------------
            # print('Start detecting pose ...')
            # detected_pre = np.empty((1, 1, 1))
            # blob = cv2.dnn.blobFromImage(cv2.resize(img_raw, (300, 300)), 1.0,
            #                              (300, 300), (104.0, 177.0, 123.0))
            # fsa_facedetector_net.setInput(blob)
            # detected = fsa_facedetector_net.forward()
            # print("detected:", detected)
            # if detected_pre.shape[2] > 0 and detected.shape[2] == 0:
            #     detected = detected_pre
            #
            # faces = np.empty((detected.shape[2], img_size, img_size, 3))
            # draw_headpose(detected, img_raw, faces, ad, img_size, img_width, img_height, model)
            # # ??????????????????-----------------------------------------------------------------------



            cv2.putText(img_raw, str("who :" + name), (500, 40), cv2.FONT_HERSHEY_DUPLEX, 1,
                        (255, 255, 255))  # ??????who
            end_time = time.time()
            T = end_time-start_time
            print('T:', T)
            fps = 1 / T  # ????????????????????????fps
            if fps > max_fps:
                max_fps = fps
            fps_txt = 'fps:%.2f' % (fps)
            cv2.putText(img_raw, str("fps :%.2f" % fps), (0, 40), cv2.FONT_HERSHEY_DUPLEX, 1,
                        (255, 0, 0))  # ????????????
            # print("fps    :", 1 / (t7 - t1))
            img_raw = cv2.cvtColor(np.asarray(img_raw), cv2.COLOR_RGB2BGR)
            cv2.imshow("detect_video", np.uint8(img_raw))

                #     siam = compare(person_feature, personal_features)
                #     sia = max(siam[0]).item()
                #     persons_similarity.append([personal_name, sia])
                # data = pd.DataFrame(persons_similarity)
                # data = data.sort_values(by=1, ascending=False)
                # obj_name = data.iloc[0][0]


        # ??????????????????q ????????????
        if cv2.waitKey(1) == ord('q'):
            break

        # ????????????????????????????????????
        if cv2.getWindowProperty('detect_video', cv2.WND_PROP_AUTOSIZE) < 1:
            break
    capture.release()
    cv2.destroyAllWindows()

    '''
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # ???????????????BGRtoRGB
        frame, boxes = retinaface.detect_image(frame)  # ?????????????????????
    frame = Image.fromarray(np.uint8(frame))

        for box in boxes:  # ??????????????????????????????????????????
            box = list(map(int, box))
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            frame = Image.fromarray(np.uint8(frame))  # ???numpy??????PIL??????
            cropped = frame.crop((x1, y1, x2, y2))

            #person1 = tf(cropped).cuda()  # ???MTCNN??????????????????????????????????????????cuda
            #person1_feature = net.encode(person1[None, ...])  # ??????????????????????????????????????????

            siam_last = 0
            name = 0
            for i in range(num):
                person2_feature = featuress[i].cuda()
                siam = compare(person1_feature, person2_feature)
                if siam > siam_last:  # ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
                    siam_last = siam
                    name = dic[featuress[i]]

            frame = np.asarray(frame)  # ???PIL??????numpy??????
            cv2.putText(frame, name + str(float("%.2f" % siam_last.detach().cpu())), (x1, y1 + 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            count += 1

        t7 = time.time()
        fps = 1 / (t7 - t1)
        cv2.putText(frame, str("fps :%.2f" % fps), (0, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0))  # ????????????
        # print("fps    :", 1 / (t7 - t1))

        frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        cv2.imshow("video", np.uint8(frame))
        c = cv2.waitKey(1) & 0xff
        
    '''

