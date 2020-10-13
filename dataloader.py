from abc import abstractmethod
import pandas as pd
import torch
import numpy as np
from torch.utils import data
import cv2
from collections import OrderedDict

class STDNDataset(torch.utils.data.Dataset):
    def __init__(self, data_config, phase="train", **kwargs):
        self.data_config = data_config
        self.phasal_config = data_config[phase]
        self.path = self.phasal_config.path
        self.data_columns = self.data_config.data_columns
        self.needed_columns = self.data_config.needed_columns
        self.target_columns = self.data_config.target_columns
        self.transforms = self.phasal_config.transforms()
        self.df = self._read_list()
        lis = np.arange(0, len(self.df))
        self.df['batch_idx'] = lis
        print('Number of samples ', len(self.df))
        print("Number of spoofs ", len(self.df[self.df.label == 1]))
        print("Number of reals ", len(self.df[self.df.label == 0]))
        super().__init__()

    def image_reader(self, im_path, color_mode='rgb'):
        img = cv2.imread(im_path)
        img = cv2.cvtColor(img, eval(f"cv2.COLOR_BGR2{color_mode.upper()}"))
        return img.astype(np.uint8)

    def __getitem__(self, idx):
        input_dict = self._get_single_instance(idx)
        return input_dict

    def __len__(self):
        return len(self.df)

    def _read_list(self):
        data_df = pd.read_csv(self.path)
        data_df = data_df[self.needed_columns]
        return data_df

    @abstractmethod
    def _get_single_instance(self, idx):
        item_dict = OrderedDict()
        subdf = self.df[self.df.batch_idx == idx]

        for column in self.needed_columns:
            data = []
            for img_path in subdf[column].values:
                data.append(img_path)
            item_dict[column] = data


        for column_name in self.data_columns:

            item_dict[column_name] = [self.image_reader(x, self.data_config.color_mode[column_name]) for x in
                                      item_dict[column_name]]



        if self.transforms is not None:
            item_dict = self.transforms(item_dict)
        for column_name in self.target_columns:
            # print(column_name, item_dict[column_name])
            item_dict[column_name] = torch.Tensor(item_dict[column_name]).long()
        return item_dict


