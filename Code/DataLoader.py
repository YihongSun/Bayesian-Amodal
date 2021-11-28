import torch
from configs import data_dir, categories
from configs import *
from pycocotools.coco import COCO
from PIL import Image, ImageOps
from torchvision import transforms
import pycocotools.mask as maskUtils
from util import *
import json

def resize_bbox(img, bboxs, short=224, single=False, interp=False):
    h, w  = img.shape[0:2]

    if single:
        box = bboxs
        factor = short / min(box[2] - box[0], box[3] - box[1])
    else:
        short_side = []
        for box in bboxs:
            short_side.append(min(box[2] - box[0], box[3] - box[1]))
        factor = short / np.min(short_side)

    resized_bboxs = bboxs * factor
    if interp:
        resized_img = cv2.resize(img, (int(w * factor), int(h * factor)), interpolation = cv2.INTER_NEAREST)
    else:
        resized_img = cv2.resize(img, (int(w * factor), int(h * factor)))
    return resized_img, resized_bboxs

def resize_scale(img, scale=1, interp=False):
    h, w  = img.shape[0:2]
    factor = scale

    if interp:
        resized_img = cv2.resize(img, (int(w * factor), int(h * factor)), interpolation = cv2.INTER_NEAREST)
    else:
        resized_img = cv2.resize(img, (int(w * factor), int(h * factor)))
    return resized_img


def get_pascal3d_data(cats, train=True, single_obj=True):

    image_files = []
    mask_files = []
    labels = []
    bboxs = []

    for category in cats:
        if train:
            filelist = data_dir + 'PASCAL3D+/PASCAL3D+_release1.1/Image_sets/{}_imagenet_train.txt'.format(category)
            mask_dir_mod = data_dir + 'Occluded_Vehicles/training/annotations/{}_raw_mod/'.format(category)
        else:
            filelist = data_dir + 'PASCAL3D+/PASCAL3D+_release1.1/Image_sets/{}_imagenet_val.txt'.format(category)
            mask_dir_mod = data_dir + 'Occluded_Vehicles/testing/annotations/{}_raw_mod/'.format(category)

        img_dir = data_dir + 'PASCAL3D+/PASCAL3D+_release1.1/Images/{}_imagenet/'.format(category)
        anno_dir = data_dir + 'PASCAL3D+/PASCAL3D+_release1.1/Annotations/{}_imagenet/'.format(category)

        with open(filelist, 'r') as fh:
            contents = fh.readlines()
        fh.close()

        img_list = [cc.strip() for cc in contents]
        label = categories['train'].index(category)

        for img_path in img_list:

            if img_path == 'n03790512_11192' and category == 'motorbike':
                continue
            img_file = img_dir + img_path + '.JPEG'
            mask_file = mask_dir_mod + img_path + '.npz'
            bbox = np.load(anno_dir + '{}.npy'.format(img_path))

            image_files.append(img_file)
            mask_files.append(mask_file)
            labels.append(label)
            if single_obj:
                bboxs.append(bbox[0])
            else:
                bboxs.append(bbox)

    return image_files, mask_files, labels, bboxs


def get_coco_data(cats, dataType='train2017', single_obj=True):
    image_files = []
    mask_files = []
    labels = []
    bboxs = []

    if single_obj:
        mask_type = '_single'
    else:
        print('Not yet implemented.')

    img_dir = data_dir + 'COCO/{}/'.format(dataType)
    mask_dir = data_dir + 'COCO/mask_{}{}/'.format(dataType, mask_type)
    annFile = data_dir + 'COCO/annotations/instances_{}.json'.format(dataType)
    coco = COCO(annFile)

    for category in cats:
        if category == 'aeroplane':
            catIds = coco.getCatIds(catNms='airplane')
        elif category == 'motorbike':
            catIds = coco.getCatIds(catNms='motorcycle')
        else:
            catIds = coco.getCatIds(catNms=category)

        filelist = data_dir + 'COCO/file_lists/{}_ZERO.txt'.format(category)
        with open(filelist, 'r') as fh:
            contents = fh.readlines()
        fh.close()

        img_list = [cc.strip() for cc in contents]

        label = categories['train'].index(category)

        for img_path in img_list:
            img_id, obj_id, d_type = img_path.split('_')
            if d_type != dataType:
                continue

            img = coco.loadImgs(int(img_id))[0]
            img_file = img_dir + img['file_name']
            if mask_type == '_single':
                mask_file = mask_dir + '{}_{}_{}.jpg'.format(category, img_id, obj_id)
            else:
                mask_file = mask_dir + '{}.jpg'.format(img_id)

            if not os.path.exists(mask_file):
                continue

            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            ann = coco.loadAnns(annIds)[int(obj_id)]
            bbox_ = ann['bbox']
            bbox = bbox_.copy()
            bbox[0] = bbox_[1]
            bbox[1] = bbox_[0]
            bbox[2] = bbox_[1] + bbox_[3]
            bbox[3] = bbox_[0] + bbox_[2]

            image_files.append(img_file)
            mask_files.append(mask_file)
            labels.append(label)
            bboxs.append(np.array(bbox))        # x1, y1, x2, y2

    return image_files, mask_files, labels, bboxs


class Single_Object_Loader():
    def __init__(self, image_files, mask_files, labels, bboxs, resize=True, ss_length=224, crop_img=True, crop_padding=48, crop_central=False, demo_img_return=True, return_true_pad=False):
        self.image_files = image_files
        self.mask_files = mask_files
        self.labels = labels
        self.bboxs = bboxs

        self.resize_bool = resize                   #boolean:   resize image
        self.resize_side = ss_length                #int:       resize side length
        self.crop_bool = crop_img                   #boolean:   crop image
        self.crop_pad = crop_padding                #int:       crop padding
        self.crop_central = crop_central            #boolean:   same padding on all 4 sides
        self.demo_bool = demo_img_return            #boolean:   return demo image corresponding to the float tensor
        self.return_true_pad = return_true_pad      #boolean:   return true pad length

    def __getitem__(self, index):

        img_path = self.image_files[index]
        mask_path = self.mask_files[index]
        label = self.labels[index]
        bbox = self.bboxs[index]

        input_image = Image.open(img_path)
        sz = input_image.size                   # W, H

        mask = np.ones((sz[1], sz[0], 3))
        if os.path.exists(mask_path):
            annotation = np.load(mask_path)
            mask[:, :, 0] = annotation['mask']

        demo_img = []

        if self.demo_bool:
            demo_img = cv2.imread(img_path)


        if self.resize_bool:
            short_side = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
            if short_side < 3:
                print('Bad Bbox Annotation:', index, img_path, bbox)
                bbox = np.array([0, 0, sz[1], sz[0]])
                short_side = min(bbox[2] - bbox[0], bbox[3] - bbox[1])

            input_image = input_image.resize((np.asarray(sz) * (self.resize_side / short_side)).astype(int), Image.ANTIALIAS)
            sz = input_image.size
            mask, _ = resize_bbox(mask, bbox, single=True, interp=True, short=self.resize_side)

            if self.demo_bool:
                demo_img, _ = resize_bbox(demo_img, bbox, single=True, short=self.resize_side)

            bbox = (bbox * (self.resize_side / short_side)).astype(int)

        pad = self.crop_pad
        if self.crop_bool:

            box = bbox

            if self.crop_central:
                box[0] = max(box[0], 0)
                box[1] = max(box[1], 0)
                box[2] = min(box[2], sz[1])
                box[3] = min(box[3], sz[0])
                pad = min(box[0] - 0, box[1] - 0, sz[1] - box[2], sz[0] - box[3], self.crop_pad)

            left = max(0, box[1] - pad)
            top = max(0, box[0] - pad)
            right = min(sz[0], box[3] + pad)
            bottom = min(sz[1], box[2] + pad)

            input_image = input_image.crop((left, top, right, bottom))
            mask = (mask[top:bottom, left:right, 0] > 127).astype(float)

            if self.demo_bool:
                demo_img = demo_img[top:bottom, left:right, :]

        rgbimg = Image.new("RGB", input_image.size)
        rgbimg.paste(input_image)

        preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        input_tensor = preprocess(rgbimg)

        if np.sum(mask) == 0:
            mask = 1 - mask

        if self.return_true_pad:
            return input_tensor, label, bbox, mask, demo_img, img_path, pad

        return input_tensor, label, bbox, mask, demo_img, img_path

    def __len__(self):
        return len(self.image_files)


class Multi_Object_Loader():
    def __init__(self, image_files, mask_files, labels, bboxs, resize=True, min_size=99999, max_size=0, demo_img_return=True):
        self.image_files = image_files
        self.mask_files = mask_files
        self.labels = labels
        self.bboxs = bboxs

        self.resize_bool = resize                   #boolean:   resize image
        self.demo_bool = demo_img_return            #boolean:   return demo image corresponding to the float tensor
        self.max_size = max_size
        self.min_size = min_size

    def __getitem__(self, index):

        img_path = self.image_files[index]
        mask_path = self.mask_files[index]
        label = self.labels[index]
        bbox = self.bboxs[index]

        input_image = Image.open(img_path)
        sz = input_image.size

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path)
        else:
            mask = np.zeros((sz[1], sz[0], 3))

        demo_img = []

        if self.demo_bool:
            demo_img = cv2.imread(img_path)


        if self.resize_bool:
            box = bbox[0]
            short_side = min(box[2] - box[0], box[3] - box[1])
            if short_side < 3:
                bbox = np.array([[0, 0, sz[1], sz[0]]])
                box = bbox[0]
                short_side = min(box[2] - box[0], box[3] - box[1])

            input_image = input_image.resize((np.asarray(sz) * (224 / short_side)).astype(int), Image.ANTIALIAS)
            sz = input_image.size
            mask, _ = resize_bbox(mask, box, single=True, interp=True)

            if self.demo_bool:
                demo_img, _ = resize_bbox(demo_img, box, single=True)

            bbox = (bbox * (224 / short_side)).astype(int)

        if sz[0] > self.max_size or sz[1] > self.max_size:
            scale = self.max_size / max(sz[0], sz[1])
            input_image = input_image.resize((np.asarray(sz) * scale).astype(int), Image.ANTIALIAS)
            mask = resize_scale(mask, scale=scale, interp=False)
            if self.demo_bool:
                demo_img = resize_scale(demo_img, scale=scale, interp=False)

            bbox = (bbox * scale).astype(int)
        else:
            scale = 1.

        if sz[0] < self.min_size or sz[1] < self.min_size:
            scale = self.min_size / min(sz[0], sz[1])
            input_image = input_image.resize((np.asarray(sz) * scale).astype(int), Image.ANTIALIAS)
            mask = resize_scale(mask, scale=scale, interp=False)
            if self.demo_bool:
                demo_img = resize_scale(demo_img, scale=scale, interp=False)

            bbox = (bbox * scale).astype(int)
        else:
            scale = 1.

        rgbimg = Image.new("RGB", input_image.size)
        rgbimg.paste(input_image)

        preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        input_tensor = preprocess(rgbimg)

        return input_tensor, label, bbox, mask, scale, demo_img, img_path

    def __len__(self):
        return len(self.image_files)


class Occluded_Classification_Dataset():
    def __init__(self, cats, occ_level='ONE', occ_type='_white', demo_img_return=True):
        self.image_files = []
        self.labels = []

        self.demo_bool = demo_img_return            #boolean:   return demo image corresponding to the float tensor


        for category in cats:

            filelist = data_dir + 'PASCAL3D+/PASCAL3D+_occ/occ_img_cropped/{}_imagenet_occ.txt'.format(category)

            img_dir = data_dir + 'PASCAL3D+/PASCAL3D+_occ/occ_img_cropped/{}LEVEL{}{}/'.format(category, occ_level, occ_type)

            with open(filelist, 'r') as fh:
                contents = fh.readlines()
            fh.close()

            img_list = [cc.strip() for cc in contents]
            label = categories['train'].index(category)

            for img_path in img_list:
                img_file = img_dir + img_path + '.JPEG'

                self.image_files.append(img_file)
                self.labels.append(label)

    def __getitem__(self, index):

        img_path = self.image_files[index]
        label = self.labels[index]

        input_image = Image.open(img_path)
        sz = input_image.size

        short_side = min(sz)

        demo_img = []

        input_image = input_image.resize((np.asarray(sz) * (224 / short_side)).astype(int), Image.ANTIALIAS)

        if self.demo_bool:
            demo_img = cv2.imread(img_path)
            demo_img = cv2.resize(demo_img, (int(sz[0] * (224 / short_side)), int(sz[1] * (224 / short_side))) )

        rgbimg = Image.new("RGB", input_image.size)
        rgbimg.paste(input_image)

        preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        input_tensor = preprocess(rgbimg)

        return input_tensor, label, -1, -1, demo_img, img_path

    def __len__(self):
        return len(self.image_files)


class Occ_Veh_Dataset():
    def __init__(self, cats, dataType='train', train_types=(None), fg_level=1, bg_level=1, single_obj=True, resize=True, crop_img=True, crop_padding=48, data_range=(0, 1), crop_central=False, demo_img_return=True):
        self.image_files = []
        self.annotation_files = []

        self.resize_bool = resize                   #boolean:   resize image
        self.crop_bool = crop_img                   #boolean:   crop image
        self.crop_pad = crop_padding                #int:       crop padding
        self.crop_central = crop_central            #boolean:   same padding on all 4 sides
        self.demo_bool = demo_img_return            #boolean:   return demo image corresponding to the float tensor

        self.artifical_occ = (fg_level + bg_level > 0)

        tag = ''
        if dataType == 'train':
            fg_level = -1
            bg_level = -1
            assert 'raw' in train_types or 'occluded' in train_types
        elif dataType == 'test':
            train_types = [None]
            assert fg_level >= 0 and bg_level >= 0
        else:
            print('dataType not recognized')


        for train_type in train_types:

            if dataType == 'train':
                tag = '_' + train_type

            if dataType == 'test':
                tag = 'FGL{}_BGL{}'.format(fg_level, bg_level)

            for category in cats:

                filelist = data_dir + 'Occluded_Vehicles/{}ing/lists/{}{}.txt'.format(dataType, category, tag)

                img_dir = data_dir + 'Occluded_Vehicles/{}ing/images/{}{}/'.format(dataType, category, tag)
                anno_dir = data_dir + 'Occluded_Vehicles/{}ing/annotations/{}{}/'.format(dataType, category, tag)

                with open(filelist, 'r') as fh:
                    contents = fh.readlines()
                fh.close()

                img_list = [cc.strip() for cc in contents]

                N = len(img_list)
                np.random.seed(0)
                index_list = np.random.permutation(N)[int(data_range[0] * N):int(data_range[1] * N)]

                for ri in index_list:
                    self.image_files.append(img_dir + img_list[ri][:-5] + '.JPEG')
                    self.annotation_files.append(anno_dir + img_list[ri][:-5] + '.npz')

    def __getitem__(self, index):

        img_path = self.image_files[index]
        anno_path = self.annotation_files[index]

        try:
            annotation = np.load(anno_path)
        except:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, True

        gt_labels = np.array([categories['train'].index(str(annotation['category']))])
        bbox = annotation['box'][0:4]
        gt_amodal_bbox = np.array([bbox]).astype(int)

        sz = [annotation['box'][5], annotation['box'][4]]

        for i in range(gt_amodal_bbox.shape[0]):
            gt_amodal_bbox[i][0] = max(0, gt_amodal_bbox[i][0])
            gt_amodal_bbox[i][1] = max(0, gt_amodal_bbox[i][1])
            gt_amodal_bbox[i][2] = min(sz[1], gt_amodal_bbox[i][2])
            gt_amodal_bbox[i][3] = min(sz[0], gt_amodal_bbox[i][3])
        try:
            obj_mask = maskUtils.decode(maskUtils.merge(maskUtils.frPyObjects(annotation['mask'].tolist(), sz[1], sz[0])))
        except:
            # mask error
            obj_mask = np.zeros((sz[1], sz[0]))

        if self.artifical_occ:
            occluder_mask = maskUtils.decode(maskUtils.merge(maskUtils.frPyObjects(annotation['occluder_mask'].tolist(), sz[1], sz[0])))
        else:
            occluder_mask = np.zeros(obj_mask.shape)
        occ_obj_mask = np.array(obj_mask * occluder_mask).astype(float)

        demo_img = None
        if self.demo_bool:
            demo_img = cv2.imread(img_path)

        if self.resize_bool:
            short_side = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
            if short_side < 3:
                bbox = np.array([[0, 0, sz[1], sz[0]]])
                short_side = min(bbox[2] - bbox[0], bbox[3] - bbox[1])

            input_image = input_image.resize((np.asarray(sz) * (224 / short_side)).astype(int), Image.ANTIALIAS)
            sz = input_image.size
            obj_mask, _ = resize_bbox(obj_mask, bbox, single=True, interp=True)
            occ_obj_mask, _ = resize_bbox(occ_obj_mask, bbox, single=True, interp=True)

            if self.demo_bool:
                demo_img, _ = resize_bbox(demo_img, bbox, single=True)

            bbox = (bbox * (224 / short_side)).astype(int)

        if self.crop_bool:

            box = bbox
            pad = self.crop_pad

            if self.crop_central:
                pad = min(box[0] - 0, box[1] - 0, sz[1] - box[2], sz[0] - box[3], self.crop_pad)

            left = max(0, box[1] - pad)
            top = max(0, box[0] - pad)
            right = min(sz[0], box[3] + pad)
            bottom = min(sz[1], box[2] + pad)

            input_image = input_image.crop((left, top, right, bottom))
            obj_mask = (obj_mask[top:bottom, left:right] > 0.5).astype(float)
            occ_obj_mask = (occ_obj_mask[top:bottom, left:right] > 0.5).astype(float)

            if self.demo_bool:
                demo_img = demo_img[top:bottom, left:right, :]

        rgbimg = Image.new("RGB", sz)
        rgbimg.paste(Image.open(img_path))

        preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        input_tensor = preprocess(rgbimg)

        inmodal_seg = (obj_mask - occ_obj_mask).astype(int)
        gt_inmodal_segentation = [inmodal_seg]
        gt_amodal_segentation = [obj_mask]
        gt_occ = [ np.sum(occ_obj_mask) / np.sum(obj_mask) ]

        if np.sum(inmodal_seg) > 0 and self.artifical_occ:
            gt_inmodal_bbox = np.array([ [ max(gt_amodal_bbox[0][0], np.min(np.where(inmodal_seg > 0.3)[0])), max(gt_amodal_bbox[0][1], np.min(np.where(inmodal_seg > 0.3)[1])), min(gt_amodal_bbox[0][2], np.max(np.where(inmodal_seg > 0.3)[0])), min(gt_amodal_bbox[0][3], np.max(np.where(inmodal_seg > 0.3)[1])) ] ])
        else:
            gt_inmodal_bbox = gt_amodal_bbox

        return input_tensor, gt_labels, gt_inmodal_bbox, gt_amodal_bbox, gt_inmodal_segentation, gt_amodal_segentation, gt_occ, demo_img, img_path, False

    def __len__(self):
        return len(self.image_files)

#Major Dataset for this project, includes both inmodal and amodal segmentation mask for analysis
class KINS_Dataset():
    def __init__(self, category_list, dataType='train', occ=(0,1), height_thrd=50, amodal_height=True, data_range=(0, 1), demo_img_return=True):

        self.src_data_path = data_dir + 'kitti/{}ing/'.format(dataType)
        self.image_ids = []
        self.obj_ids = []
        self.demo_bool = demo_img_return
        occ_lb = occ[0]
        occ_ub = occ[1]
        assert occ_lb <= occ_ub

        cat_kins = []
        for category in category_list:
            cat_kins.append(categories['kins'].index(category))

        filelist = '{}list.txt'.format(self.src_data_path)

        with open(filelist, 'r') as fh:
            contents = fh.readlines()
        fh.close()
        img_list = [cc.strip() for cc in contents]
        N = len(img_list)
        img_list = img_list[int(data_range[0] * N):int(data_range[1] * N)]
        for ii, img_id in enumerate(img_list):

            if ii % 10 == 0:
                print('Loading Data: {}/{}'.format(ii, N), end='\r')

            annotation = np.load('{}annotations/{}.npz'.format(self.src_data_path, img_id))
            obj_ids = annotation['obj_ids']
            labels = annotation['labels']
            occlusion_fractions = annotation['occluded_percentage']
            if amodal_height:
                bboxes = annotation['amodal_bbox']  # dim = (N, 4) --> (y1, x1, y2, x2)
            else:
                bboxes = annotation['inmodal_bbox']


            obj_ids_per_img = []

            for i in range(obj_ids.shape[0]):
                box = bboxes[i]           #inmodal_bboxes, amodal_bboxes

                if labels[i] in cat_kins and occlusion_fractions[i] >= occ_lb and occlusion_fractions[i] <= occ_ub and box[3] - box[1] >= height_thrd:
                    obj_ids_per_img.append(obj_ids[i])

            if len(obj_ids_per_img) > 0:
                self.image_ids.append(img_id)
                self.obj_ids.append(obj_ids_per_img)


    def __getitem__(self, index):

        img_id = self.image_ids[index]
        obj_id = self.obj_ids[index]

        img_path = self.src_data_path + 'images/{}.png'.format(img_id)

        input_image = Image.open(img_path)
        demo_img = []
        if self.demo_bool:
            demo_img = cv2.imread(img_path)

        annotation = np.load('{}annotations/{}.npz'.format(self.src_data_path, img_id), allow_pickle=True)

        obj_ids = annotation['obj_ids']
        inmodal_bbox = annotation['inmodal_bbox']
        amodal_bbox = annotation['amodal_bbox']

        labels = annotation['labels']
        occlusion_fractions = annotation['occluded_percentage']

        # dim = [ encode, encode, encode ]
        inmodal_masks_ = annotation['inmodal_mask']
        amodal_masks_ = annotation['amodal_mask']

        gt_inmodal_bbox = []
        gt_amodal_bbox = []
        gt_labels = []
        gt_occ = []
        gt_inmodal_segentation = []
        gt_amodal_segentation = []

        for id in obj_id:
            index = np.where(obj_ids == id)[0][0]

            box = inmodal_bbox[index]
            gt_inmodal_bbox.append(np.array([box[1], box[0], box[3], box[2]]))

            box = amodal_bbox[index]
            gt_amodal_bbox.append(np.array([box[1], box[0], box[3], box[2]]))

            if categories['kins'][labels[index]] in categories['train']:
                gt_labels.append( categories['train'].index( categories['kins'][labels[index]] ) )
            else:
                gt_labels.append(-1)

            gt_occ.append(occlusion_fractions[index])
            gt_inmodal_segentation.append(maskUtils.decode(inmodal_masks_[index])[:, :, np.newaxis].squeeze())
            gt_amodal_segentation.append(maskUtils.decode(amodal_masks_[index][0]).squeeze())

        gt_inmodal_bbox = np.array(gt_inmodal_bbox)
        gt_amodal_bbox = np.array(gt_amodal_bbox)
        gt_labels = np.array(gt_labels)
        gt_occ = np.array(gt_occ)
        gt_inmodal_segentation = np.array(gt_inmodal_segentation)
        gt_amodal_segentation = np.array(gt_amodal_segentation)

        rgbimg = Image.new("RGB", input_image.size)
        rgbimg.paste(input_image)

        preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        input_tensor = preprocess(rgbimg)

        return input_tensor, gt_labels, gt_inmodal_bbox, gt_amodal_bbox, gt_inmodal_segentation, gt_amodal_segentation, gt_occ, demo_img, img_path, False


    def __len__(self):
        return len(self.image_ids)

# Dataset based on the PASCAL3D+ Dataset with artificial generated occlusions -- Currently Active
class Occ_Veh_Dataset_NOT_ACTIVE():
    def __init__(self, cats, dataType='train', train_types=(None), fg_level=1, bg_level=1, single_obj=True, resize=True, crop_img=True, crop_padding=48, data_range=(0, 1), crop_central=False, demo_img_return=True):
        self.image_files = []
        self.mask_files = []
        self.labels = []
        self.bboxs = []

        self.resize_bool = resize                   #boolean:   resize image
        self.crop_bool = crop_img                   #boolean:   crop image
        self.crop_pad = crop_padding                #int:       crop padding
        self.crop_central = crop_central            #boolean:   same padding on all 4 sides
        self.demo_bool = demo_img_return            #boolean:   return demo image corresponding to the float tensor

        self.artifical_occ = (fg_level + bg_level > 0)

        tag = ''
        tag_mod = ''
        if dataType == 'train':
            fg_level = -1
            bg_level = -1
            assert 'raw' in train_types or 'occluded' in train_types
        elif dataType == 'test':
            train_types = [None]
            assert fg_level >= 0 and bg_level >= 0
        else:
            print('dataType not recognized')


        for train_type in train_types:

            if dataType == 'train':
                tag = '_' + train_type
                tag_mod = '_raw'

            if dataType == 'test':
                tag = 'FGL{}_BGL{}'.format(fg_level, bg_level)
                tag_mod = 'FGL0_BGL0'

            for category in cats:

                filelist = data_dir + 'Occluded_Vehicles/{}ing/lists/{}{}.txt'.format(dataType, category, tag)

                img_dir = data_dir + 'Occluded_Vehicles/{}ing/images/{}{}/'.format(dataType, category, tag)
                mask_dir = data_dir + 'Occluded_Vehicles/{}ing/annotations/{}{}/'.format(dataType, category, tag)
                mask_dir_mod = data_dir + 'Occluded_Vehicles/{}ing/annotations/{}{}_mod/'.format(dataType, category, tag_mod)

                anno_dir = data_dir + 'PASCAL3D+/PASCAL3D+_release1.1/Annotations/{}_imagenet/'.format(category)

                with open(filelist, 'r') as fh:
                    contents = fh.readlines()
                fh.close()

                img_list = [cc.strip() for cc in contents]
                label = categories['train'].index(category)

                N = len(img_list)
                np.random.seed(0)
                index_list = np.random.permutation(N)[int(data_range[0] * N):int(data_range[1] * N)]

                for ri in index_list:
                    img_path = img_list[ri][:-5]
                    img_file = img_dir + img_path + '.JPEG'
                    mask_file = [mask_dir_mod + img_path + '.npz', mask_dir + img_path + '.npz']
                    bbox = np.load(anno_dir + img_path + '.npy')

                    self.image_files.append(img_file)
                    self.mask_files.append(mask_file)
                    self.labels.append(label)

                    if single_obj:
                        self.bboxs.append(bbox[0])
                    else:
                        self.bboxs.append(bbox)

    def __getitem__(self, index):

        img_path = self.image_files[index]
        mask_path_mod, mask_path = self.mask_files[index]
        gt_labels = np.array([self.labels[index]])
        gt_amodal_bbox = np.array([self.bboxs[index]]).astype(int)

        input_image = Image.open(img_path)
        sz = input_image.size

        for i in range(gt_amodal_bbox.shape[0]):
            gt_amodal_bbox[i][0] = max(0, gt_amodal_bbox[i][0])
            gt_amodal_bbox[i][1] = max(0, gt_amodal_bbox[i][1])
            gt_amodal_bbox[i][2] = min(sz[1], gt_amodal_bbox[i][2])
            gt_amodal_bbox[i][3] = min(sz[0], gt_amodal_bbox[i][3])
        try:
            annotation = np.load(mask_path_mod)
        except:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, True

        obj_mask = (annotation['mask'] > 177).astype(float)
        if self.artifical_occ:
            annotation = np.load(mask_path)
            occluder_mask = annotation['occluder_mask'].T.T
        else:
            occluder_mask = np.zeros(obj_mask.shape)

        occ_obj_mask = np.array(obj_mask * occluder_mask).astype(float)

        demo_img = None
        if self.demo_bool:
            demo_img = cv2.imread(img_path)

        if self.resize_bool:
            short_side = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
            if short_side < 3:
                bbox = np.array([[0, 0, sz[1], sz[0]]])
                short_side = min(bbox[2] - bbox[0], bbox[3] - bbox[1])

            input_image = input_image.resize((np.asarray(sz) * (224 / short_side)).astype(int), Image.ANTIALIAS)
            sz = input_image.size
            obj_mask, _ = resize_bbox(obj_mask, bbox, single=True, interp=True)
            occ_obj_mask, _ = resize_bbox(occ_obj_mask, bbox, single=True, interp=True)

            if self.demo_bool:
                demo_img, _ = resize_bbox(demo_img, bbox, single=True)

            bbox = (bbox * (224 / short_side)).astype(int)

        if self.crop_bool:

            box = bbox
            pad = self.crop_pad

            if self.crop_central:
                pad = min(box[0] - 0, box[1] - 0, sz[1] - box[2], sz[0] - box[3], self.crop_pad)

            left = max(0, box[1] - pad)
            top = max(0, box[0] - pad)
            right = min(sz[0], box[3] + pad)
            bottom = min(sz[1], box[2] + pad)

            input_image = input_image.crop((left, top, right, bottom))
            obj_mask = (obj_mask[top:bottom, left:right] > 0.5).astype(float)
            occ_obj_mask = (occ_obj_mask[top:bottom, left:right] > 0.5).astype(float)

            if self.demo_bool:
                demo_img = demo_img[top:bottom, left:right, :]


        rgbimg = Image.new("RGB", input_image.size)
        rgbimg.paste(input_image)

        preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        input_tensor = preprocess(rgbimg)

        inmodal_seg = (obj_mask - occ_obj_mask).astype(int)
        gt_inmodal_segentation = [inmodal_seg]
        gt_amodal_segentation = [obj_mask]
        gt_occ = [( np.sum(occ_obj_mask) - np.sum(obj_mask) ) / np.sum(obj_mask) ]

        inmodal_seg = (obj_mask - occ_obj_mask).astype(float)

        if np.sum(inmodal_seg) > 0 and self.artifical_occ:
            gt_inmodal_bbox = np.array([ [ max(int(self.bboxs[index][0]), np.min(np.where(inmodal_seg > 0.3)[0])), max(int(self.bboxs[index][1]), np.min(np.where(inmodal_seg > 0.3)[1])), min(int(self.bboxs[index][2]), np.max(np.where(inmodal_seg > 0.3)[0])), min(int(self.bboxs[index][3]), np.max(np.where(inmodal_seg > 0.3)[1])) ] ])
        else:
            gt_inmodal_bbox = gt_amodal_bbox

        return input_tensor, gt_labels, gt_inmodal_bbox, gt_amodal_bbox, gt_inmodal_segentation, gt_amodal_segentation, gt_occ, demo_img, img_path, False

    def __len__(self):
        return len(self.image_files)


class COCO_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        # self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = []
        for cat in ['airplane', 'bicycle', 'bus', 'car', 'motorcycle']:
            catIds = self.coco.getCatIds(catNms=[cat])
            self.ids += self.coco.getImgIds(catIds=catIds)
        self.ids = list(set(self.ids))
        self.max_size = 1800
        self.min_size = 400


    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        input_image = Image.open(os.path.join(self.root, path))
        sz = input_image.size
        demo = cv2.imread(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([ymin, xmin, ymax, xmax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        # img_id = torch.tensor([img_id])
        # # Size of bbox (Rectangular)
        # areas = []
        # for i in range(num_objs):
        #     areas.append(coco_annotation[i]['area'])
        # areas = torch.as_tensor(areas, dtype=torch.float32)
        # # Iscrowd
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # # Annotation is in dictionary format
        # my_annotation = {}
        # my_annotation["boxes"] = boxes
        # my_annotation["labels"] = labels
        # my_annotation["image_id"] = img_id
        # my_annotation["area"] = areas
        # my_annotation["iscrowd"] = iscrowd

        if sz[0] > self.max_size or sz[1] > self.max_size:
            scale = self.max_size / max(sz[0], sz[1])
            input_image = input_image.resize((np.asarray(sz) * scale).astype(int), Image.ANTIALIAS)
            demo = resize_scale(demo, scale=scale, interp=False)

            boxes = boxes * scale


        if sz[0] < self.min_size or sz[1] < self.min_size:
            scale = self.min_size / min(sz[0], sz[1])
            input_image = input_image.resize((np.asarray(sz) * scale).astype(int), Image.ANTIALIAS)
            demo = resize_scale(demo, scale=scale, interp=False)

            boxes = boxes * scale


        rgbimg = Image.new("RGB", input_image.size)
        rgbimg.paste(input_image)

        preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        input_tensor = preprocess(rgbimg)

        return input_tensor, boxes, labels, demo

    def __len__(self):
        return len(self.ids)

#Major Dataset for this project, includes both inmodal and amodal segmentation mask for analysis
class KINS_Compnet_Train_Dataset():
    def __init__(self, category, height_thrd=50, frac=1.0, pad=0, demo_img_return=True):
        self.src_data_path = data_dir + 'kitti/compnet_training/'

        self.files = []
        self.bbox = []
        self.demo_bool = demo_img_return

        with open('{}{}_annotations.pickle'.format(self.src_data_path, category), 'rb') as fh:
            annotations = pickle.load(fh)
        fh.close()

        for anno in annotations:
            if anno['short_side'] >= height_thrd:
                if pad <= 0:
                    self.files.append(anno['file'])
                else:
                    self.files.append(anno['org_file'])
                    self.bbox.append(anno['inmodal_bbox'])

        self.files = self.files[0:int(frac * len(self.files))]
        self.pad = pad
        np.random.seed(0)
        self.random_transform = np.random.permutation(len(self.files))

        if pad > 0:
            self.src_data_path = data_dir + 'kitti/training/'

    def __getitem__(self, index):
        index = self.random_transform[index]
        img_path = self.src_data_path + 'images/{}.png'.format(self.files[index])
        input_image = Image.open(img_path)
        pad = 0

        if self.pad > 0:
            bbox = self.bbox[index]

            factor = 224 / (bbox[2] - bbox[0])

            input_image = input_image.resize((np.asarray(input_image.size) * factor).astype(int))
            bbox = (np.array(bbox) * factor).astype(int)
            sz = input_image.size
            pad = min(bbox[0] - 0, bbox[1] - 0, sz[1] - bbox[2], sz[0] - bbox[3], self.pad)

            top = max(0, bbox[0] - pad)
            left = max(0, bbox[1] - pad)
            bottom = min(sz[1], bbox[2] + pad)
            right = min(sz[0], bbox[3] + pad)

            input_image = input_image.crop((left, top, right, bottom))

        demo_img = []
        if self.demo_bool:
            demo_img = np.asarray(input_image)

        rgbimg = Image.new("RGB", input_image.size)
        rgbimg.paste(input_image)

        preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        input_tensor = preprocess(rgbimg)

        return input_tensor, demo_img, img_path, pad


    def __len__(self):
        return len(self.files)


class COCOA_Dataset():
    def __init__(self, cats, dataTypes=['train', 'val'], fg_level=1, resize=True, crop_img=True, crop_padding=48, data_range=(0, 1), crop_central=False, demo_img_return=True):
        self.image_files = []
        self.annotation_files = []

        self.resize_bool = resize                   #boolean:   resize image
        self.crop_bool = crop_img                   #boolean:   crop image
        self.crop_pad = crop_padding                #int:       crop padding
        self.crop_central = crop_central            #boolean:   same padding on all 4 sides
        self.demo_bool = demo_img_return            #boolean:   return demo image corresponding to the float tensor
        
        occ_levels = {0: [0, 0], 1: [0.0001, 0.2], 2: [0.2, 0.4], 3: [0.4, 0.7]}
        occ_lbound, occ_ubound = occ_levels[fg_level]

        self.image_files = []
        self.annotations = []


        for data_type in dataTypes:
            img_dir = data_dir + 'COCO/{}2014/'.format(data_type)
            with open(data_dir + 'COCO/COCOA_annotations_detectron/COCO_amodal_{}2014_with_classes.json'.format(data_type), 'r') as fh:
                annotation = json.load(fh)

            image_obj_pair = dict()

            count = 0
            for anno in annotation['annotations']:
                if anno['category_id'] < len(categories['coco']) and categories['coco'][anno['category_id']] in cats and anno['occlude_rate'] >= occ_lbound and anno['occlude_rate'] <= occ_ubound:
                    try:
                        image_obj_pair[anno['image_id']].append(anno)
                    except:
                        image_obj_pair[anno['image_id']] = [anno]
                count += 1
                if count > data_range[1] * len(annotation['annotations']):
                    break

            for image_anno in annotation['images']:
                try:
                    obj_annotations = image_obj_pair[image_anno['id']]
                except:
                    continue
                self.image_files.append(img_dir + image_anno['file_name'])
                self.annotations.append(obj_annotations)

    def __getitem__(self, index):

        img_path = self.image_files[index]
        annotations = self.annotations[index]

        gt_labels = []
        gt_occ = []
        gt_amodal_bbox = []
        gt_inmodal_bbox = []
        gt_inmodal_segmentation = []
        gt_amodal_segmentation = []


        for anno in annotations:
            inmodal_mask = maskUtils.decode(anno['visible_mask'])[:, :, np.newaxis].squeeze()

            try:
                occ_mask = maskUtils.decode(anno['invisible_mask'])[:, :, np.newaxis].squeeze()
            except:
                occ_mask = np.zeros(inmodal_mask.shape)

            amodal_mask = (inmodal_mask + occ_mask > 0.5).astype(int)

            inmodal_box = [ np.min(np.where(inmodal_mask > 0.5)[0]), np.min(np.where(inmodal_mask > 0.5)[1]), np.max(np.where(inmodal_mask > 0.5)[0]), np.max(np.where(inmodal_mask > 0.5)[1]) ]
            amodal_box = [ np.min(np.where(amodal_mask > 0.5)[0]), np.min(np.where(amodal_mask > 0.5)[1]), np.max(np.where(amodal_mask > 0.5)[0]), np.max(np.where(amodal_mask > 0.5)[1]) ]

            gt_labels.append(categories['train'].index( categories['coco'][anno['category_id']] ))
            gt_occ.append(anno['occlude_rate'])
            gt_inmodal_segmentation.append(inmodal_mask)
            gt_amodal_segmentation.append(amodal_mask)
            gt_inmodal_bbox.append(inmodal_box)
            gt_amodal_bbox.append(amodal_box)

        gt_labels = np.array(gt_labels)
        gt_occ = np.array(gt_occ)
        gt_inmodal_bbox = np.array(gt_inmodal_bbox)
        gt_amodal_bbox = np.array(gt_amodal_bbox)
        gt_inmodal_segmentation = np.array(gt_inmodal_segmentation)
        gt_amodal_segmentation = np.array(gt_amodal_segmentation)

        demo_img = cv2.imread(img_path)

        sz = demo_img.shape[0:2][::-1]
        rgbimg = Image.new("RGB", sz)
        rgbimg.paste(Image.open(img_path))

        preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        input_tensor = preprocess(rgbimg)

        return input_tensor, gt_labels, gt_inmodal_bbox, gt_amodal_bbox, gt_inmodal_segmentation, gt_amodal_segmentation, gt_occ, demo_img, img_path, False

    def __len__(self):
        return len(self.image_files)