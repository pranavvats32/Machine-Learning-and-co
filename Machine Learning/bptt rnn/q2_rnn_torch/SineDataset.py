import torch
from torch.utils.data import Dataset
from typing import Tuple

class SineDataset(Dataset):
    def __init__(self, dataset_size, samples_per_sequence, sample_step):
        """
        Create a dataset containing dataset_size sine curves, each with
        samples_per_sequence samples. For each sequence, the samples are
        computed as sin(x/T), where T is chosen uniformly at random in
        the range [1, 2] and x is a vector containing values of the form 
        x_0 + sample_step*K, 0 <= K < samples_per_sequence where x_0
        is chosen uniformly at random in the range [0, 4pi]
        """
        self.dataset_size = dataset_size
        self.samples_per_sequence = samples_per_sequence
        self.sample_step = sample_step

        min_T = 1
        max_T = 2

        sample_start = torch.rand(size=(dataset_size,))*4*torch.pi
        T = torch.rand(size=(dataset_size,))*(max_T - min_T) + min_T
        sequences = [s + torch.linspace(s, s + sample_step*(samples_per_sequence + 1), 
            samples_per_sequence) for s in sample_start]
        sin_arg_points = torch.stack(sequences, dim=0)
        self.sin_arg_points = sin_arg_points
        sequences = torch.sin(torch.divide(sin_arg_points, T[:, None]))

        self.sequences = sequences

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor]:
        """
        Return the index-th sequence of the dataset and corresponding ground
        truth. The sequence contains all but the last samples of the sequence
        created at initialization, and the ground truth all but the first,
        such that the ground truth is the sequence "delayed" by one sample.
        """
        return (self.sequences[index, :-1], self.sequences[index, 1:])