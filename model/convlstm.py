import torch.nn as nn
import torch
import numpy as np


class ConvLSTMCell3D(nn.Module):

    def __init__(self, inChannel, hChannel, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        inChannel: int
            Number of channels of input tensor.
        hChannel: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell3D, self).__init__()

        self.inChannel = inChannel
        self.hChannel = hChannel

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv3d(in_channels=self.inChannel+self.hChannel,
                              out_channels=4 * self.hChannel,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.h_cur = None
        self.c_cur = None

    def forward(self, input_tensor):
        states_size=list(input_tensor.shape)
        states_size[1]=self.hChannel
        if self.h_cur is None:
            self.h_cur = nn.Parameter(torch.zeros(states_size),requires_grad=False)
        elif self.h_cur.shape[-1]>states_size[-1]:
            self.h_cur=self.h_cur[:,:,:,:,:states_size[-1]]
        elif self.h_cur.shape[-1]<states_size[-1]:
            temp=self.h_cur
            self.h_cur=nn.Parameter(torch.zeros(states_size),requires_grad=False)
            self.h_cur[:,:,:,:,:temp.shape[-1]]=temp
        if self.c_cur is None:
            self.c_cur = nn.Parameter(torch.zeros(states_size),requires_grad=False)
        elif self.c_cur.shape[-1]>states_size[-1]:
            self.c_cur=self.c_cur[:,:,:,:,:states_size[-1]]
        elif self.c_cur.shape[-1]<states_size[-1]:
            temp=self.c_cur
            self.c_cur=nn.Parameter(torch.zeros(states_size),requires_grad=False)
            self.c_cur[:,:,:,:,:temp.shape[-1]]=temp
        self.h_cur.data = self.h_cur.data.to(dtype=input_tensor.dtype,device=input_tensor.device)
        self.c_cur.data = self.c_cur.data.to(dtype=input_tensor.dtype,device=input_tensor.device)
        input_tensor = torch.cat([input_tensor,self.h_cur.data],dim=1)
        combined_conv = self.conv(input_tensor)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hChannel, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * self.c_cur.data + i * g
        h_next = o * torch.tanh(c_next)
        self.c_cur.data.copy_(c_next)
        self.h_cur.data.copy_(h_next)
        return h_next

    def reset(self):
        self.h_cur=None
        self.c_cur=None
        return
    def reset_h(self):
        self.h_cur=None
    def init_lstm_states(self,input_size):
        self.h_cur=nn.Parameter(torch.zeros(input_size), requires_grad=False)
        self.c_cur=nn.Parameter(torch.zeros(input_size), requires_grad=False)
        return


class ConvLSTMCells3D(nn.Module):

    """
    Parameters:
        inChannel: Number of channels in input
        hChannel: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        bias: Bias or no bias in Convolution
        Note: Will do same padding.

    Input:
        A tensor of size N, inChannel, D, H, W
    Output:
        A tensor of size N, hChannel, D, H, W
    """

    def __init__(self, inChannel, hChannel, kernel_size, num_layers, bias=True):
        super(ConvLSTMCells3D, self).__init__()
        self.inChannel = inChannel
        self.hChannel = hChannel
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.inChannel if i == 0 else self.hChannel
            cell_list.append(ConvLSTMCell3D(inChannel=cur_input_dim,
                                            hChannel=self.hChannel,
                                            kernel_size=self.kernel_size,
                                            bias=self.bias))
        self.cell_list = nn.Sequential(*cell_list)

    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input_tensor: todo
            6-D Tensor either of shape (t, b, c, d, h, w) or (b, t, c, d, h, w)
        reset: reset all states to 0

        Returns
        -------
        layer_output
        """
        output = input_tensor
        output = self.cell_list(output)
        return output

    def reset(self):
        for layer_idx in range(self.num_layers):
            self.cell_list[layer_idx].reset()
        return
    def reset_h(self):
        for layer_idx in range(self.num_layers):
            self.cell_list[layer_idx].reset_h()
        return
    def init_lstm_states(self,input_size):
        for layer_idx in range(self.num_layers):
            self.cell_list[layer_idx].init_lstm_states(input_size)
        return


if __name__ == '__main__':
    input_tensor = torch.randn(size=(1,2,64,64,64))
    model = ConvLSTMCells3D(inChannel=2,hChannel=2,kernel_size=3,num_layers=2)
    output_tensor1 = model(input_tensor)
    output_tensor2 = model(input_tensor)
    pass
