import traceback
from torchvision.datasets import VOCDetection
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor, to_pil_image
import torchvision.transforms as transforms
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# import albumentations as A
# from albumentations.pytorch import ToTensor
import os 
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import parse
import cv2
from typing import Dict, Any
from collections import OrderedDict
import collections
import tensorflow as tf


path2data = 'pascal_VOC'
if not os.path.exists(path2data):
    os.mkdir(path2data)


# VOC class names
classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]


class myVOCDetection(VOCDetection):
    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]).convert('RGB'))  # 여기서 image를 resize해줘야 할거 같은데 아닌가?
        img_2_ten = T.ToTensor()(img)
        re_img = T.Resize(size = (224,224))(img_2_ten)
        img_np = re_img.numpy()
        img = tf.convert_to_tensor(img_np)
        
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot()) # xml파일 분석하여 dict으로 받아오기

        targets = [] # 바운딩 박스 좌표
        labels = [] # 바운딩 박스 클래스

        # 바운딩 박스 정보 받아오기
        for t in target['annotation']['object']:
            label = np.zeros(5)
            label[:] = t['bndbox']['xmin'], t['bndbox']['ymin'], t['bndbox']['xmax'], t['bndbox']['ymax'], classes.index(t['name'])

            targets.append(list(label[:4])) # 바운딩 박스 좌표
            labels.append(label[4])         # 바운딩 박스 클래스

        if self.transforms:
            augmentations = self.transforms(image=img, bboxes=targets)
            img = augmentations['image']
            targets = augmentations['bboxes']

        return img, targets, labels

    def parse_voc_xml(self, node: ET.Element) -> Dict[str, Any]:    # xml 파일을 dictionary로 반환
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
    
    
    
train_ds = myVOCDetection(path2data, year='2007', image_set='train', download=True)
# img, target, label = train_ds[2]
# print(img.dtype)  >  (3, 244, 244)
test_ds = myVOCDetection(path2data, year='2007', image_set='test', download=True)












