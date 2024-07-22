import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class OurDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # img_path = os.path.join(self.image_dir, self.images[index])       #Wei run for Windows
        # mask_path = os.path.join(self.mask_dir, self.images[index].replace('.tif','.png'))
        # mask_path = os.path.join(self.mask_dir, self.images[index])       #Wei run for Windows

        img_path = self.image_dir + "/" + self.images[index]         #Wei
        mask_path = self.mask_dir + "/" +  self.images[index].replace('.tif','.png')

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        mask[mask!=1.0]=0.0         #2023/3/24添加，针对cvc_clinicDB进行修改

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
