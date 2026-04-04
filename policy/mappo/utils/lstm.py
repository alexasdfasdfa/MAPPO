import torch
import torch.nn as nn
import time

class LSTMLayer(nn.Module):
    def __init__(self, input_shape, lstm_hidden_dim, recurrent_N):
        super().__init__()
        self.input_dim = input_shape
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(self.input_dim, lstm_hidden_dim, num_layers=recurrent_N, batch_first=True)

        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation
        :param state: tensor of shape (batch_size, human n, length of a joint state)
        :return:
        """
        size = state.shape
        # ========== 修复: 使用输入张量的设备，支持多卡 DataParallel ==========
        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim, device=state.device)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim, device=state.device)
        
        output, (hn, cn) = self.lstm(state, (h0, c0))
        hn = hn.squeeze(0)
        return hn
