import torch
from torch.nn import functional
from AbstractRNNCell import AbstractRNNCell

class MyLSTMCell(AbstractRNNCell):

    def __init__(self, input_size, hidden_size, output_size, device):
        """
        Create an LSTM operating on inputs of feature dimension input_size,
        outputting outputs of feature dimension output_size and maintaining a
        hidden state of size hidden_size.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        #####Start Subtask 2a#####
        self.W_i = torch.nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.empty((hidden_size, input_size + hidden_size))))
        self.bias_i = torch.nn.Parameter(torch.zeros(hidden_size))
        
        self.W_f = torch.nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.empty((hidden_size, input_size + hidden_size))))
        self.bias_f = torch.nn.Parameter(torch.zeros(hidden_size))

        self.W_o = torch.nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.empty((hidden_size, input_size + hidden_size))))
        self.bias_o = torch.nn.Parameter(torch.zeros(hidden_size))    

        self.W_g = torch.nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.empty((hidden_size, input_size + hidden_size))))
        self.bias_g = torch.nn.Parameter(torch.zeros(hidden_size))            
        
        self.W_hy = torch.nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.empty((output_size, hidden_size))))
        self.bias_hy = torch.nn.Parameter(torch.zeros(output_size))    
        #####End Subtask 2a#####
        self.has_hidden_state = False
        self.device = device

    def get_rnn_type(self):
        return "LSTM"

    def forward(self, x : torch.tensor, reset_hidden_state : bool = True):
        if(len(x.shape) < 1): #batch sz 1, seq len 1, 1-d features
            x = x[None][None, None]
        elif(len(x.shape) < 2): #seq len 1, 1-d features
            x = x[:, None, None]
        elif(len(x.shape) < 3): #1-d features
            x = x[:, :, None]
        batch_size = x.shape[0]
        sequence_len = x.shape[1]

        if(reset_hidden_state or not self.has_hidden_state):
            self.hidden_state = torch.zeros((self.hidden_size, batch_size), 
                device=self.device)
            self.cell_state = torch.zeros((self.hidden_size, batch_size), 
                device=self.device)            
            self.has_hidden_state = True
        outputs = torch.zeros((x.shape[0], x.shape[1], self.output_size), 
            device=self.device)
        
        for t in range(sequence_len):
            curr_x = x[:, t, :]
            #####Start Subtask 2b#####
            combined_input = torch.concat((curr_x.T, self.hidden_state), dim=0)
            #implement LSTM update equations
            current_i = torch.sigmoid(torch.matmul(self.W_i, 
                combined_input) + self.bias_i[:, None])
            current_f = torch.sigmoid(torch.matmul(self.W_f, 
                combined_input) + self.bias_f[:, None])
            current_o = torch.sigmoid(torch.matmul(self.W_o, 
                combined_input) + self.bias_o[:, None])
            cell_state_tmp = torch.tanh(torch.matmul(self.W_g, 
                combined_input) + self.bias_g[:, None])
            self.cell_state = torch.sigmoid(current_f*self.cell_state + \
                current_i*cell_state_tmp)
            self.hidden_state = torch.tanh(self.cell_state)*current_o
            #apply the "linear layer" of the output
            curr_y = (torch.matmul(self.W_hy, self.hidden_state) + \
                self.bias_hy[:, None]).T
            outputs[:, t, :] = curr_y
            #####End Subtask 2b#####

        return torch.squeeze(outputs)
