import torch
from torch.nn import functional
from AbstractRNNCell import AbstractRNNCell

class BasicRNNCell(AbstractRNNCell):

    def __init__(self, input_size, hidden_size, output_size, device):
        """
        Create a basic RNN operating on inputs of feature dimension 
        input_size, outputting outputs of feature dimension output_size 
        and maintaining a hidden state of size hidden_size.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        #initialize the weight matrix (with weights for both input and
        #hidden state) as a trainable parameter
        self.W_xh_hh = torch.nn.Parameter(torch.nn.init.kaiming_normal_(
            torch.empty((hidden_size, input_size + hidden_size))))
        #same for bias vector
        self.bias_hh = torch.nn.Parameter(torch.zeros(hidden_size))
        #same for the "linear layer" mapping hidden state to outputs
        self.W_hy = torch.nn.Parameter(torch.nn.init.kaiming_normal_(
            torch.empty((output_size, hidden_size))))
        self.bias_hy = torch.nn.Parameter(torch.zeros(output_size))
        #no hidden state initially.
        self.has_hidden_state = False

    def get_rnn_type(self):
        return "BasicRNN"

    def forward(self, x : torch.tensor, 
                reset_hidden_state : bool = True)-> torch.tensor:
        if(len(x.shape) < 1): #batch sz 1, seq len 1, 1-d features
            x = x[None][None, None]
        elif(len(x.shape) < 2): #seq len 1, 1-d features
            x = x[:, None, None]
        elif(len(x.shape) < 3): #1-d features
            x = x[:, :, None]
        batch_size = x.shape[0]
        sequence_len = x.shape[1]
        #initialize hidden state to zeros if it's the first time processing
        #a batch or if we want the RNN to start from scratch
        if(reset_hidden_state or not self.has_hidden_state):
            self.hidden_state = torch.zeros((self.hidden_size, batch_size), 
                device=self.device)
            self.has_hidden_state = True
        outputs = torch.zeros((x.shape[0], x.shape[1], self.output_size), 
            device=self.device)
        
        for t in range(sequence_len):
            curr_x = x[:, t, :] #select the t-th timestep of all sequences

            combined_input = torch.concat((curr_x.T, self.hidden_state), dim=0)
            #implement RNN update equation
            self.hidden_state = torch.matmul(self.W_xh_hh, combined_input) + \
                self.bias_hh[:, None]
            self.hidden_state = functional.relu(self.hidden_state)
            #apply the "linear layer" of the output
            curr_y = (torch.matmul(self.W_hy, self.hidden_state) + \
                self.bias_hy[:, None]).T
            outputs[:, t, :] = curr_y

        return torch.squeeze(outputs)
