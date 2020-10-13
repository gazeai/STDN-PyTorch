import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for i, data in enumerate(tqdm(loader)):
            label = data["label"]
            # print(label)
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        print(f"Label set: {self.labels_set}")
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        self.array_of_labels = []

        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
            self.array_of_labels.append(self.label_to_indices[l])
        max_label = len(max(self.array_of_labels, key=len))
        print(f"max_labels: {max_label}")
        for l in range(len(self.array_of_labels)):
            if len(self.array_of_labels[l]) < max_label:
                #                 print(l)
                new = np.concatenate((self.array_of_labels[l], np.random.choice(self.array_of_labels[l],
                                                                                max_label - len(
                                                                                    self.array_of_labels[l]))), axis=0)
                self.array_of_labels[l] = new
        self.array_of_labels = np.array(self.array_of_labels)
        print(f"Array shape: {self.array_of_labels.shape}")
        lis = []

        for arr in self.array_of_labels:
            if len(arr) % n_samples != 0:
                slice_val = len(arr) - (len(arr) % n_samples)
                lis.append(np.reshape(arr[:slice_val], (-1, n_samples)))
            else:
                lis.append(np.reshape(arr, (-1, n_samples)))

        self.all_indices = np.concatenate(tuple(lis), axis=-1)
        print(f"After concatenation array shape: {self.array_of_labels.shape}")
        # self.n_classes = n_classes
        # self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = n_samples * n_classes
        # super(BalancedBatchSampler, self).__init__(balanced_batch_sampler, self.batch_size, False)

    def __iter__(self):
        for indices in self.all_indices:
            yield indices

    def __len__(self):
        return len(self.dataset) // self.batch_size
