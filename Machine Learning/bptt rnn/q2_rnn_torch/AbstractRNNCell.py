from torch.nn import Module
from torch import tensor
from abc import ABC, abstractmethod

class AbstractRNNCell(Module, ABC):
    """
    An abstract base class for our simple recurrent networks.
    """
    @abstractmethod
    def get_rnn_type(self) -> str:
        """
        Return the type of RNN being implemented as a string.
        """
        pass

    @abstractmethod
    def forward(self, x : tensor, reset_hidden_state : bool = True):
        """
        Performs a forward pass of the RNN on input x.
        Args:
            -x: a tensor of shape [B, L, I_F] where B is the batch size
            (number of sequences processed in parallel), L is the length
            of the sequences and I_F is the dimensionality (features) of
            a single sample of the sequence. If x is 0-d, B = L = I_F = 1.
            If x is 1-d, L = I_F = 1. If x is 2-d, I_F = 1.
            -reset_hidden_state: whether to start processing x from
            scratch or continue from the last value of the hidden state
            of the RNN.
        Returns:
            A tensor of shape [B, L, O_F] containing an output of
            dimensionality (features) O_F for every sample of every
            sequence of the batch x. The output is squeezed to match
            x.shape in case x had fewer than 3 dimensions.
        """
        pass