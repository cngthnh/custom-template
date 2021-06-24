from classification.utils.getter import *
from detection.utils.getter import *
import argparse
import os
import cv2
import matplotlib.pyplot as plt 
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from detection.utils.utils import draw_boxes_v2
from detection.utils.postprocess import box_fusion, postprocessing, change_box_order
import pandas as pd
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from detection.augmentations.transforms import get_resize_augmentation
from detection.augmentations.transforms import MEAN, STD
import numpy as np

parser = argparse.ArgumentParser(description='Classify an image / folder of images')
parser.add_argument('--detector_weight', type=str ,help='trained weight')
parser.add_argument('--classifier_weight', type=str ,help='trained weight')
parser.add_argument('--input_path', type=str, help='path to an image to inference')
parser.add_argument('--output_path', type=str, help='path to save csv result file')
parser.add_argument('--visualized', type=str, default='/content/visualized', help='path to save visualized images')
parser.add_argument('--gpus', type=str, default='0', help='path to save inferenced image')
parser.add_argument('--min_conf', type=float, default= 0.15, help='minimum confidence for an object to be detect')
parser.add_argument('--min_iou', type=float, default=0.5, help='minimum iou threshold for non max suppression')
parser.add_argument('--tta', action='store_true', help='whether to use test time augmentation')
parser.add_argument('--tta_ensemble_mode', type=str, default='wbf', help='tta ensemble mode')
parser.add_argument('--tta_conf_threshold', type=float, default=0.01, help='tta confidence score threshold')
parser.add_argument('--tta_iou_threshold', type=float, default=0.9, help='tta iou threshold')
parser.add_argument('--expand_top', type=float, default=1.0, help='bbox expansion')
parser.add_argument('--expand_btm', type=float, default=0.5, help='bbox expansion')
parser.add_argument('--expand_left', type=float, default=0.5, help='bbox expansion')
parser.add_argument('--expand_right', type=float, default=0.5, help='bbox expansion')
parser.add_argument('--no_visualization', dest='visualization', action='store_false')
parser.set_defaults(visualization=True)

class ClassificationTestset():
    def __init__(self, config, img_list, transforms=None):
        self.transforms = transforms
        self.image_size = config.image_size
        self.load_images(img_list)

    def get_batch_size(self):
        num_samples = len(self.img_list)

        # Temporary
        return 1

    def load_images(self, img_list):
        self.img_list = img_list

    def __getitem__(self, idx):
        img = self.img_list[idx]
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        return {
            'img': img,
        }

    def collate_fn(self, batch):
        imgs = torch.stack([s['img'] for s in batch])  
        return {
            'imgs': imgs,
        }

    def __len__(self):
        return len(self.img_list)

    def __str__(self):
        return f"Number of found images: {len(self.img_list)}"
  
def classify(args, config, img_list):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    num_gpus = len(config.gpu_devices.split(','))
    devices_info = get_devices_info(config.gpu_devices)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    test_transforms = A.Compose([
        get_resize_augmentation(config.image_size, keep_ratio=config.keep_ratio),
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=1.0, p=1.0),
        ToTensorV2(p=1.0)
    ])

    # if config.tta:
    #     config.tta = TTA(
    #         min_conf=config.tta_conf_threshold, 
    #         min_iou=config.tta_iou_threshold, 
    #         postprocess_mode=config.tta_ensemble_mode)
    # else:
    #     config.tta = None

    testset = ClassificationTestset(
        config, 
        img_list=img_list,
        transforms=test_transforms)

    testloader = DataLoader(
        testset,
        batch_size=testset.get_batch_size(),
        num_workers=2,
        pin_memory=True,
        collate_fn=testset.collate_fn
    )

    if args.classifier_weight is not None:
        class_names, num_classes = get_class_names(args.classifier_weight)


    net = BaseTimmModel(
        name=config.model_name, 
        num_classes=num_classes)

    model = Classifier( model = net,  device = device)

    model.eval()
    if args.classifier_weight is not None:                
        load_checkpoint(model, args.classifier_weight)

    pred_list = []
    prob_list = []

    with torch.no_grad():
        for idx, batch in enumerate(testloader):  
            preds, probs = model.inference_step(batch, return_probs=True)

            for idx, (pred, prob) in enumerate(zip(preds, probs)):
                pred_list.append(class_names[pred])
                prob_list.append(prob)
                
    return pred_list, prob_list

class DetectionTestset():
    def __init__(self, config, input_path, transforms=None):
        self.input_path = input_path # path to image folder or a single image
        self.transforms = transforms
        self.image_size = config.image_size
        self.load_images()

    def get_batch_size(self):
        num_samples = len(self.all_image_paths)

        # Temporary
        return 1

    def load_images(self):
        self.all_image_paths = []   
        if os.path.isdir(self.input_path):  # path to image folder
            paths = sorted(os.listdir(self.input_path))
            for path in paths:
                self.all_image_paths.append(os.path.join(self.input_path, path))
        elif os.path.isfile(self.input_path): # path to single image
            self.all_image_paths.append(self.input_path)

    def __getitem__(self, idx):
        image_path = self.all_image_paths[idx]
        image_name = os.path.basename(image_path)
        img = cv2.imread(image_path)
        image_w, image_h = self.image_size
        ori_height, ori_width, c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        ori_img = img.copy()
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        return {
            'img': img,
            'img_name': image_name,
            'ori_img': ori_img,
            'image_ori_w': ori_width,
            'image_ori_h': ori_height,
            'image_w': image_w,
            'image_h': image_h,
        }

    def collate_fn(self, batch):
        imgs = torch.stack([s['img'] for s in batch])   
        ori_imgs = [s['ori_img'] for s in batch]
        img_names = [s['img_name'] for s in batch]
        image_ori_ws = [s['image_ori_w'] for s in batch]
        image_ori_hs = [s['image_ori_h'] for s in batch]
        image_ws = [s['image_w'] for s in batch]
        image_hs = [s['image_h'] for s in batch]
        img_scales = torch.tensor([1.0]*len(batch), dtype=torch.float)
        img_sizes = torch.tensor([imgs[0].shape[-2:]]*len(batch), dtype=torch.float)
        return {
            'imgs': imgs,
            'ori_imgs': ori_imgs,
            'img_names': img_names,
            'image_ori_ws': image_ori_ws,
            'image_ori_hs': image_ori_hs,
            'image_ws': image_ws,
            'image_hs': image_hs,
            'img_sizes': img_sizes, 
            'img_scales': img_scales
        }

    def __len__(self):
        return len(self.all_image_paths)

    def __str__(self):
        return f"Number of found images: {len(self.all_image_paths)}"

def detect(args, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    num_gpus = len(args.gpus.split(','))
    devices_info = get_devices_info(args.gpus)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    test_transforms = A.Compose([
        get_resize_augmentation(config.image_size, keep_ratio=config.keep_ratio),
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=1.0, p=1.0),
        ToTensorV2(p=1.0)
    ])

    if args.tta:
        args.tta = TTA(
            min_conf=args.tta_conf_threshold, 
            min_iou=args.tta_iou_threshold, 
            postprocess_mode=args.tta_ensemble_mode)
    else:
        args.tta = None

    testset = DetectionTestset(
        config, 
        args.input_path,
        transforms=test_transforms)
    testloader = DataLoader(
        testset,
        batch_size=testset.get_batch_size(),
        num_workers=2,
        pin_memory=True,
        collate_fn=testset.collate_fn
    )
    
    if args.detector_weight is not None:
        class_names, num_classes = get_class_names(args.detector_weight)
    class_names.insert(0, 'Background')
    net = get_model(args, config, device, num_classes=num_classes)

    model = Detector(model = net, device = device)
    model.eval()
    if args.detector_weight is not None:                
        load_checkpoint(model, args.detector_weight)

    if os.path.isdir(args.input_path):
        if not os.path.exists(args.visualized):
            os.makedirs(args.visualized)

    #load classfier' configs
    classifier_config = get_config(args.classifier_weight)
    if classifier_config is None:
        print("Config not found. Load configs from classification/configs/configs.yaml")
        classifier_config = Config(os.path.join('classification/configs','configs.yaml'))
    else:
        print("Load configs from weight")

    ## Print info
    print(config)
    print(classifier_config)
    print(testset)
    print(f"Nubmer of gpus: {num_gpus}")
    print(devices_info)

    labels_list = []
    prob_list = []
    img_list = []
    box_list = []

    empty_imgs = 0
    with tqdm(total=len(testloader)) as pbar:
        with torch.no_grad():
            for idx, batch in enumerate(testloader):
                if args.tta is not None:
                    preds = args.tta.make_tta_predictions(model, batch)
                else:
                    preds = model.inference_step(batch)

                for idx, outputs in enumerate(preds):
                    img_name = batch['img_names'][idx]
                    ori_img = batch['ori_imgs'][idx]
                    img_w = batch['image_ws'][idx]
                    img_h = batch['image_hs'][idx]
                    img_ori_ws = batch['image_ori_ws'][idx]
                    img_ori_hs = batch['image_ori_hs'][idx]
                    
                    outputs = postprocessing(
                        outputs, 
                        current_img_size=[img_w, img_h],
                        ori_img_size=[img_ori_ws, img_ori_hs],
                        min_iou=args.min_iou,
                        min_conf=args.min_conf,
                        max_dets=config.max_post_nms,
                        keep_ratio=config.keep_ratio,
                        output_format='xywh',
                        mode=config.fusion_mode)

                    boxes = outputs['bboxes'] 
                    labels = outputs['classes']  
                    scores = outputs['scores']

                    if len(boxes) == 0:
                        empty_imgs += 1
                        boxes = None

                    _croppedImgs = []

                    if boxes is not None:
                        for box in boxes:
                            box[0]-=args.expand_left*box[2]
                            box[1]-=args.expand_top*box[3]
                            box[2]+=(args.expand_left + args.expand_right)*box[2]
                            box[3]+=(args.expand_top + args.expand_btm)*box[3]
                            uly = int(box[1])
                            ulx = int(box[0])
                            lrx = int(box[0]+box[2])
                            lry = int(box[1]+box[3])
                            if (uly<0): uly = 0
                            if (ulx<0): ulx = 0
                            _img = ori_img[uly:lry, ulx:lrx]
                            _img = img_resize(_img, classifier_config.image_size[0])

                            box_list.append(box)
                            _croppedImgs.append(_img)
                            img_list.append(img_name)
                        
                        _labels, _probs = classify(args,classifier_config,img_list=_croppedImgs)

                        labels_list += _labels

                        prob_list += _probs

                        if (args.visualization):
                            if os.path.isdir(args.input_path):
                                out_path = os.path.join(args.visualized, img_name)
                            else:
                                out_path = args.visualized
                            draw_boxes_v2(out_path, ori_img , boxes, _labels, _probs, class_names)

                pbar.update(1)
                pbar.set_description(f'Empty images: {empty_imgs}')

    result_df = pd.DataFrame({
        'img_name':img_list,
        'predictions': labels_list,
        'probalities': prob_list,
        'bbox': box_list
    })

    result_df.to_csv(args.output_path, index=False)

    print(f"Result file is saved to {args.output_path}")


def img_resize(img, size, interpolation = cv2.INTER_AREA):
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w: return cv2.resize(img, (size, size), interpolation)
    if h > w: dif = h
    else:     dif = w
    x_pos = int((dif - w)/2.)
    y_pos = int((dif - h)/2.)
    if c is None:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    return cv2.resize(mask, (size, size), interpolation)

if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config(args.detector_weight)
    if config is None:
        print("Config not found. Load configs from detection/configs/configs.yaml")
        config = Config(os.path.join('detection/configs','configs.yaml'))
    else:
        print("Load configs from weight")
    detect(args, config)
    