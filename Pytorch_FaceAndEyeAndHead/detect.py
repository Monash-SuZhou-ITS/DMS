from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--retinaface_network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False
                    , help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.7, type=float, help='visualization_threshold')


parser.add_argument('--show_cutting_image', action ="store_true", default =True, help = 'show_crop_images')
parser.add_argument('--save_folder', default='./curve/info', type=str, help='Dir to save results')

parser.add_argument('--image_path', default='./curve/raw_picture/gzd_09.jpg', type=str, help='picture path to detect')
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
if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.retinaface_network == "mobile0.25":
        cfg = cfg_mnet
    elif args.retinaface_network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    #print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1




    # testing begin
    for i in range(1):#range(100)
        image_path = args.image_path#"curve/raw_picture/gzd_02.jpg"
        image_name=image_path.split("/")[-1].split(".")[0]
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        #print(img_raw)
        img = np.float32(img_raw)
        #print(img.shape)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)#???????????????????????????
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
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
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        # ?????????????????????????????????Retinaface_pytorch??????????????????????????????????????????????????????????????????????????????????????????????????????????????????dets????????????????????????????????????
        # ??????????????????????????????????????????

        cut_path = "./curve/cut_faces/"  # ??????????????????????????????????????????????????? ???????????????????????????
        mark_path = "./curve/mark_faces/"#??????????????????mark
        # ????????????
        if args.show_cutting_image:
            if not os.path.exists(cut_path):
                os.makedirs(cut_path)
            for num, b in enumerate(dets):  # dets????????????????????????????????????????????????
                if b[4] < args.vis_thres:
                    continue
                b = list(map(int, b))

                # landms???????????????????????????????????????????????????????????????????????????????????????????????????????????????
                # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

                # ???????????????????????????
                Height = b[3] - b[1]
                Width = b[2] - b[0]

                # ????????????????????????
                print("????????? / faces in all:", str(num + 1), "\n")
                print("???????????? / The size of window:"
                    , '\n', "?????? / height:", Height
                    , '\n', "?????? / width: ", Width)

                # ?????????????????????????????????????????????
                img_blank = np.zeros((Height, Width, 3), np.uint8)
                # ??????????????????????????????
                for h in range(Height):
                    for w in range(Width):
                        img_blank[h][w] = img_raw[b[1] + h][b[0] + w]

                # cv2.namedWindow(image_name + "_"+str(num + 1))  # , 2)
                # cv2.imshow(image_name + "_"+str(num + 1), img_blank)  # ????????????
                cv2.imwrite(cut_path + image_name + "_"+str(num + 1) + ".jpg", img_blank)  # ???????????????????????????????????????
                print("Save into:", cut_path + image_name + "_"+str(num + 1) + ".jpg")
                cv2.waitKey(0)

        # ?????????????????????????????????????????????txt???
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        fw = open(os.path.join(args.save_folder, '__dets.txt'), 'w')  # ???????????????????????????????????????????????????__dets???txt??????
        if args.save_folder:
            fw.write('{:s}\n'.format(image_name))  # ???txt?????????????????????
            for k in range(dets.shape[0]):  # ??????dets?????????????????????dets??????????????????????????????????????????x,y???????????????????????????x,y???????????????????????????scores?????????????????????x,y????????????15????????????
                xmin = dets[k, 0]
                ymin = dets[k, 1]
                xmax = dets[k, 2]
                ymax = dets[k, 3]
                score = dets[k, 4]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                landms1_x = dets[k, 5]
                landms1_y = dets[k, 6]
                landms2_x = dets[k, 7]
                landms2_y = dets[k, 8]
                landms3_x = dets[k, 9]
                landms3_y = dets[k, 10]
                landms4_x = dets[k, 11]
                landms4_y = dets[k, 12]
                landms5_x = dets[k, 13]
                landms5_y = dets[k, 14]
                # ???????????????????????????????????????????????????????????????????????????txt?????????
                fw.write('{:d} {:d} {:d} {:d} {:.10f} {:d} {:d} {:d} {:d} {:d} {:d} {:d} {:d} {:d} {:d}\n'
                         .format(int(xmin),int(ymin),int(w),int(h),score,
                                 int(landms1_x),int(landms1_y),int(landms2_x),
                                 int(landms2_y),int(landms3_x),int(landms3_y),
                                 int(landms4_x),int(landms4_y),int(landms5_x),
                                 int(landms5_y)))
            # ?????????????????????txt??????
            fw.close()



        # show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image
            if not os.path.exists(mark_path):
                os.makedirs(mark_path)
            cv2.imwrite(mark_path+image_name+".jpg", img_raw)

