import os.path as osp
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
import torch
from PIL import ImageEnhance

THIS_PATH = osp.dirname(__file__)
PATH = '/autodl-tmp/COCO2017'
 
transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)


class CocoSet(Dataset):
    """ Usage:
    """
    def __init__(self, path, split,class_type, args, manualSeed=1, aug=False,transform=None):
        

        self.aug = aug
        self.class_type=class_type
        manualSeed = 8601#random.randint(1, 10000)
        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
        PATH=osp.join(path,args['dataset'])
        dataset_time=args['dataset'][-4:]
        if split=='train':
            path=osp.join(PATH, 'train'+dataset_time)
            self.coco=COCO(osp.join(PATH, f'annotations/instances_train{dataset_time}.json'))
        else:
            path=osp.join(PATH, 'val'+dataset_time)
            self.coco=COCO(osp.join(PATH, f'annotations/instances_val{dataset_time}.json'))
        
        self.id2label_map, self.label2class_map = self.get_coco_labels()
        self.image_dir=path
        if self.class_type=='base':
            label_sets=[0,1,2,3,5,7,9,13,15,16,19,21,22,23,24,25,26,27,28,31,32,34,35,38,39,41,42,43,44,45,53,55,56,57,58,60,62,63,64,65,66,67,69,70,71,72,73,74,75,78]
        elif self.class_type=='val':
            label_sets=[4,11,29,30,33,37,49,51,52,79]
        else:
            label_sets=[6,8,10,12,14,17,18,20,36,40,46,47,48,50,54,59,61,68,76,77]
        self.label_sets=label_sets
        self.data,self.label,self.label_str = self.load_paths_multilabels(label_sets)
        self.ids=list(range(len(self.data)))
        self.label_ids=self.assign_label_to_id(self.data,self.label)
        # Transformation
        if args['modeltype'] == 'ConvNet':
            self.image_size = 84
            print('use convnet transform!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args['modeltype'] == 'ResNet':
            self.image_size = 224
            print('use ResNet transform!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            self.transform = transform
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')

        self.normalize_param = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225])
        self.jitter_param = dict(Brightness=0.2, Contrast=0.2, Color=0.2)

        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        self.transform_aug = transforms.Compose(transform_funcs)

    def get_coco_labels(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        classes = {}
        id2label_map = {}

        for cat in categories:
            id2label_map[cat['id']] = len(classes)#catid:label_idx,90 to 79
            classes[len(classes)] = cat['name']#label_idx:label_name,79 to str

        return id2label_map, classes


    def load_paths_multilabels(self,label_sets): 
        label_sets=set(label_sets)
        i=0
        imgs_paths, imgs_labels,imgs_classnames = [], [], []
        for img_id, img_data in self.coco.imgs.items():
            img_anns_ids = self.coco.getAnnIds(imgIds=[img_id])
            img_anns = self.coco.loadAnns(img_anns_ids)

            if self.class_type=='base':
                img_labels = {
                    self.id2label_map[ann["category_id"]] for ann in img_anns
                }
            else:
                img_labels = {
                    self.id2label_map[ann["category_id"]] for ann in img_anns
                }
            if not img_labels:
                continue
            if not img_labels.issubset(label_sets):
                continue
            if not len(img_labels)>=2:
                i+=1
                continue   
            img_labels=list(img_labels)
            imgs_paths.append(osp.join(self.image_dir,img_data["file_name"]))
            imgs_labels.append(img_labels)
            img_classnames=','.join([str(label) for label in img_labels])
            imgs_classnames.append(img_classnames)
        print(i)
        print('%d images in total'%len(imgs_paths))
        return imgs_paths, imgs_labels,imgs_classnames


    def assign_label_to_id(self, data, label):

        out = {}

        for i in range(len(data)):
            label_int = label[i]
            for j in range(len(label_int)):
                num=(label_int[j])
                if out.get(num) == None:
                    out[num] = []
                out[num].append(i)

        return out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label_i = self.data[i], self.label_str[i]
        img_temp = Image.open(path).convert('RGB')

        if self.aug == True:
            image = self.transform_aug(img_temp)
        else:
            image = self.transform(img_temp)
        return (image, label_i)


    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


    def jitter(self, img):
        transforms = [(transformtypedict[k], self.jitter_param[k]) for k in self.jitter_param]
        out = img
        randtensor = torch.rand(len(transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out

    def parse_transform(self,  transform_type):
        if transform_type=='ImageJitter':
            method = ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size)
        elif transform_type=='CenterCrop':
            return method(self.image_size)
        elif transform_type=='Resize':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()