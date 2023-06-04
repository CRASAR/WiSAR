import torch
import numpy as np

from torch.utils.data import Dataset

from utils.data_augmentations import get_valid_transforms, get_normalize_transform, get_tensor_transform

class EfficientDetDataset(Dataset):
    def __init__(
        self, dataset_adaptor, transforms=get_valid_transforms(), normalize_transform=get_normalize_transform(), tensor_transform=get_tensor_transform()
    ):
        self.ds = dataset_adaptor
        self.transforms = transforms
        self.normalize_transform = normalize_transform
        self.tensor_transform = tensor_transform
        self.__normalize_images = True

    def set_normalization(self, normalize_images):
        self.__normalize_images = bool(normalize_images)

    def __getitem__(self, index):
        (
            image,
            pascal_bboxes,
            class_labels,
            image_id,
        ) = self.ds.get_image_and_labels_by_idx(index)

        sample = {
            "image": np.array(image, dtype=np.uint8),
            "bboxes": pascal_bboxes,
            "labels": class_labels,
        }

        sample = self.transforms(**sample)
        if(self.__normalize_images):
            sample = self.normalize_transform(**sample)
        sample = self.tensor_transform(**sample)
        sample["bboxes"] = np.array(sample["bboxes"])
        image = sample["image"]
        pascal_bboxes = sample["bboxes"]
        labels = sample["labels"]

        _, new_h, new_w = image.shape
        if(len(sample["bboxes"]) > 0):
            sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][
                :, [1, 0, 3, 2]
            ]  # convert to yxyx
        else:
            sample["bboxes"] = torch.tensor([[]]).reshape(0,4)

        target = {
            "bboxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
            "labels": torch.as_tensor(labels),
            "image_id": torch.tensor([image_id]),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
        }

        return image, target, image_id

    def __len__(self):
        return len(self.ds)