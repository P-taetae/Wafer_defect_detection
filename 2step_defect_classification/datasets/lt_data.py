import os
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
import random


class LT_Dataset(Dataset):
    train_txt = ""
    test_txt = ""

    def __init__(self, root, train=True, transform=None, target_transform=None, use_randaug=False):
        self.img_path = []
        self.labels = []
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.use_randaug = use_randaug

        if train:
            self.txt = self.train_txt
        else:
            self.txt = self.test_txt

        with open(self.txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
    
        self.cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        if self.train and self.use_randaug:
            r = random.random()
            if r < 0.5:
                image = self.transform[0](image)
            else:
                image = self.transform[1](image)
        else:
            if self.transform is not None:
                image = self.transform(image)

        # if self.target_transform is not None:
        #     label = self.target_transform(target)

        # if self.transform is not None:
        #     image = self.transform(image)

        return image, label
    
    def get_cls_num_list(self):
        counter = defaultdict(int)
        for label in self.labels:
            counter[label] += 1
        labels = list(counter.keys())
        labels.sort()
        cls_num_list = [counter[label] for label in labels]
        return cls_num_list
