import random
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.crops import functional as F

class BBoxOneSafeRandomCrop(A.BBoxSafeRandomCrop):
    """Crop a random part of the input so that it contains at least one of the bboxes."""

    def __init__(self, height, width, always_apply=False, p=1.0):
        super(BBoxOneSafeRandomCrop, self).__init__(erosion_rate=0.0, always_apply=always_apply, p=p)
        self.height = height
        self.width = width

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        if len(params["bboxes"]) == 0:
            return {
                "h_start": random.random(),
                "w_start": random.random(),
                "crop_height": self.height,
                "crop_width": self.width
            }
        #Select a bbox as a target
        target_bbox = random.choice(params["bboxes"])
        x1,y1,x2,y2,_ = target_bbox

        #Generate a random crop that contains that bbox
        crop_h_ratio = self.height
        crop_w_ratio = self.width

        x1_px = x1*img_w
        x2_px = x2*img_w
        y1_px = y1*img_h
        y2_px = y2*img_h

        min_start_w = max(0, x2_px-self.width)
        max_start_w = min(x1_px, img_w-self.width)
        min_start_h = max(0, y2_px-self.height)
        max_start_h = min(y1_px, img_h-self.height)

        y_min = int(round((random.random() * (max_start_h-min_start_h)) + min_start_h))
        x_min = int(round((random.random() * (max_start_w-min_start_w)) + min_start_w))

        return {"x_min": x_min, "y_min": y_min, "x_max": x_min+self.width, "y_max": y_min+self.height}

    def apply(self, img, x_min=0, y_min=0, x_max=0, y_max=0, **params):
        return F.crop(img, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
    
    def apply_to_bbox(self, bbox, x_min=0, y_min=0, x_max=0, y_max=0, rows=0, cols=0, **params):
        return F.bbox_crop(bbox, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, rows=rows, cols=cols)

def get_normalize_transform():
    return A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def get_tensor_transform():
    return ToTensorV2(p=1)

def get_train_transforms(target_img_size=512):
    return A.Compose(
        [
            A.RandomScale(scale_limit=(-0.7,0.1)),
            A.OneOf(
                [
                    BBoxOneSafeRandomCrop(target_img_size, target_img_size, p=1.0),
                    A.RandomCrop(target_img_size, target_img_size, p=1.0)
                ], p=1.0),
            A.Flip(p=0.5),
            A.RandomRotate90(p=1),
            A.Emboss(p=0.25),
            A.RandomSnow(p=0.15),
            A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.4, p=0.15),
            A.ToSepia(p=0.1),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.25, p=0.5),
                    A.ColorJitter(hue=0.05, p=0.5)
                ], p=1.0),
            A.GaussNoise(p=1.0)
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )

def get_valid_transforms(target_img_size=512):
    return get_train_transforms(target_img_size) #unless we decide to do something different, we validate on the same augmentations we train on

def get_inference_transforms(target_img_size=512, normalize=True):
    tfs = [A.Resize(target_img_size, target_img_size)]

    if(normalize):
        tfs.append(get_normalize_transform())

    tfs.append(get_tensor_transform())

    return A.Compose(tfs, p=1.0, bbox_params=A.BboxParams(format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]))