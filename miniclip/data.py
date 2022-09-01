import glob
import json
import os
import warnings

import torch
from PIL import Image
from torch.utils.data import Dataset


class MemeDataset(Dataset):
    def __init__(self, data, preprocess):
        self.preprocess = preprocess
        self.img_paths = []
        self.captions = []
        for img_path, captions in data.items():
            for cap in captions:
                self.img_paths.append(img_path)
                self.captions.append(cap)
        self.processed_cache = {}
        for img_path in data:
            self.processed_cache[img_path] = self.preprocess(Image.open(img_path))
        self.img_paths_set = list(data.keys())
        self.path2label = {path: self.img_paths_set.index(path) for path in self.img_paths_set}

        if len(self) == 0:
            warnings.warn("Dataset has zero size")

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = self.processed_cache[img_path]
        caption = self.captions[idx]
        label = self.path2label[img_path]
        return image, caption, label

    @classmethod
    def from_directories(cls, img_root, json_root, preprocess):
        img_paths = glob.glob(os.path.join(img_root, "*.jpg"))
        img_path_to_caption = dict()

        for img_path in img_paths:
            name = img_path.split("/")[-1].split(".")[0]

            with open(os.path.join(json_root, name + ".json"), "r") as f:
                captions = json.load(f)
                temp = []

                for cap in captions:
                    cap_cat = cap[0] + " " + cap[1]
                    if "http" not in (cap_cat) and 8 <= len(cap_cat) <= 72:
                        temp.append(cap_cat)

                img_path_to_caption[img_path] = temp

        return cls(img_path_to_caption, preprocess)


def train_val_split(dataset, train_proportion=0.8):
    train_size = int(len(dataset) * train_proportion)
    val_size = len(dataset) - train_size

    lengths = [train_size, val_size]

    return torch.utils.data.random_split(dataset, lengths)
