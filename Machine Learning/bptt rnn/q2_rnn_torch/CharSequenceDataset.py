import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from typing import Tuple

class CharSequenceDataset(Dataset):
    def __init__(self, dataset_size : int, to_remember_len : int, 
                 blank_separation_len : int):
        """
        Create a dataset containing character sequences of length
        to_remember_len random characters in {0, ..., 7}, then
        blank_separation_len blanks, a delimiter and another to_remember_len
        blanks. The ground truth for each sequence is a sequence of
        to_remember_len + blank_separation + 1 blanks followed by the
        to_remember_len characters of the sequence.

        Input sequences are one-hot encoded.
        """
        #define the integers corresponding to the blank and delimiter chars
        blank_char = 8
        delim_char = 9

        self.dataset_size = dataset_size
        #generate random sequences, blanks and delimiters
        to_remember_seq = torch.randint(0, blank_char, (dataset_size, to_remember_len))
        blanks = blank_char*torch.ones((dataset_size, blank_separation_len))
        delimiters = delim_char*torch.ones((dataset_size, 1))
        #generate the "space" for the net's output
        blanks_for_answer = blank_char*torch.ones((dataset_size, to_remember_len))
        #concatenate everything and store
        sequences = torch.concat((to_remember_seq, blanks, delimiters,
            blanks_for_answer), dim=1).long()
        self.sequences = one_hot(sequences)
        #the ground truth is simply a number of blanks and then the initial
        #sentence
        blanks = blank_char*torch.ones((dataset_size, to_remember_len + 
            blank_separation_len + 1))
        self.ground_truth = torch.concat((blanks, to_remember_seq), dim=1).long()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor]:
        """
        Returns the index-th character sequence and corresponding ground
        truth.
        """
        return (self.sequences[index, :], self.ground_truth[index, :])