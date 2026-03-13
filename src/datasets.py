import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2


def crop_roi(pil_img, row, pad_ratio=0.08):
    w, h = pil_img.size
    x1, y1, x2, y2 = int(row["roi_x1"]), int(row["roi_y1"]), int(row["roi_x2"]), int(row["roi_y2"])

    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return pil_img.crop((x1, y1, x2, y2))


def equalize_rgb(img_pil):
    img = np.array(img_pil)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return Image.fromarray(out.astype(np.uint8))


def clahe_rgb(img_pil, clip_limit=2.0, tile_grid_size=(8, 8)):
    img = np.array(img_pil)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return Image.fromarray(out.astype(np.uint8))


class GTSRBCustomDataset(Dataset):
    def __init__(self, df, transform=None, use_roi_crop=False, preprocessing="none", pad_ratio=0.08):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform
        self.use_roi_crop = use_roi_crop
        self.preprocessing = preprocessing
        self.pad_ratio = pad_ratio

    def __len__(self):
        return len(self.df)

    def _preprocess(self, img, row):
        if self.use_roi_crop:
            img = crop_roi(img, row, pad_ratio=self.pad_ratio)

        if self.preprocessing == "equalize":
            img = equalize_rgb(img)
        elif self.preprocessing == "clahe":
            img = clahe_rgb(img)

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self._preprocess(img, row)

        if self.transform is not None:
            img = self.transform(img)

        label = int(row["label"])
        return img, label
