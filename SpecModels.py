
#from . import util
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import models
from IPython.display import Audio
from scipy.io import wavfile


def encode_mu_law(x, mu=256):
    mu = mu-1
    fx = np.sign(x)*np.log(1+mu*np.abs(x))/np.log(1+mu)
    return np.floor((fx+1)/2*mu+0.5).astype(np.long)

def decode_mu_law(y, mu=256):
    mu = mu-1
    fx = (y-0.5)/mu*2-1
    x = np.sign(fx)/mu*((1+mu)**np.abs(fx)-1)
    return x

class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot,self).__init__()
        self.depth = depth
        self.ones = torch.eye(depth)
    def forward(self, X_in):
        X_in = X_in.long()
        return Variable(self.ones.index_select(0,X_in.data))
    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)



class WaveNet(nn.Module):
    def __init__(self, mu=256, n_residue=32, n_skip=512, dilation_depth=10, n_repeat=5):
        # mu: audio quantization size
        # n_residue: residue channels
        # n_skip: skip channels
        # dilation_depth &amp;amp;amp; n_repeat: dilation layer setup
        super(WaveNet, self).__init__()
        self.dilation_depth = dilation_depth
        dilations = self.dilations = [2 ** i for i in range(dilation_depth)] * n_repeat
        self.one_hot = One_Hot(mu)
        self.from_input = nn.Conv1d(in_channels=mu, out_channels=n_residue, kernel_size=1)
        self.conv_sigmoid = nn.ModuleList(
            [nn.Conv1d(in_channels=n_residue, out_channels=n_residue, kernel_size=2, dilation=d)
             for d in dilations])
        self.conv_tanh = nn.ModuleList(
            [nn.Conv1d(in_channels=n_residue, out_channels=n_residue, kernel_size=2, dilation=d)
             for d in dilations])
        self.skip_scale = nn.ModuleList([nn.Conv1d(in_channels=n_residue, out_channels=n_skip, kernel_size=1)
                                         for d in dilations])
        self.residue_scale = nn.ModuleList([nn.Conv1d(in_channels=n_residue, out_channels=n_residue, kernel_size=1)
                                            for d in dilations])
        self.conv_post_1 = nn.Conv1d(in_channels=n_skip, out_channels=n_skip, kernel_size=1)
        self.conv_post_2 = nn.Conv1d(in_channels=n_skip, out_channels=mu, kernel_size=1)

    def forward(self, input):
        output = self.preprocess(input)
        skip_connections = []  # save for generation purposes
        for s, t, skip_scale, residue_scale in zip(self.conv_sigmoid, self.conv_tanh, self.skip_scale,
                                                   self.residue_scale):
            output, skip = self.residue_forward(output, s, t, skip_scale, residue_scale)
            skip_connections.append(skip)
        # sum up skip connections
        output = sum([s[:, :, -output.size(2):] for s in skip_connections])
        output = self.postprocess(output)
        return output

    def preprocess(self, input):
        output = self.one_hot(input).unsqueeze(0).transpose(1, 2)
        output = self.from_input(output)
        return output

    def postprocess(self, input):
        output = nn.functional.elu(input)
        output = self.conv_post_1(output)
        output = nn.functional.elu(output)
        output = self.conv_post_2(output).squeeze(0).transpose(0, 1)
        return output

    def residue_forward(self, input, conv_sigmoid, conv_tanh, skip_scale, residue_scale):
        output = input
        output_sigmoid, output_tanh = conv_sigmoid(output), conv_tanh(output)
        output = nn.functional.sigmoid(output_sigmoid) * nn.functional.tanh(output_tanh)
        skip = skip_scale(output)
        output = residue_scale(output)
        output = output + input[:, :, -output.size(2):]
        return output, skip


def sine_generator(seq_size=6000, mu=256):
    framerate = 44100
    t = np.linspace(0, 5, framerate * 5)
    data = np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 224 * t)
    data = data / 2
    while True:
        start = np.random.randint(0, data.shape[0] - seq_size)
        ys = data[start:start + seq_size]
        ys = encode_mu_law(ys, mu)
        yield Variable(torch.from_numpy(ys[:seq_size]))


g = sine_generator(mu=64, seq_size=20000)


net = WaveNet(mu=64,n_residue=24,n_skip=128,dilation_depth=10,n_repeat=2)
print(net)
optimizer = optim.Adam(net.parameters(),lr=0.01)
batch_size = 64
loss_save = []
max_epoch = 2000
for epoch in range(max_epoch):
    optimizer.zero_grad()
    loss = 0
    for _ in range(batch_size):
        batch = next(g)
        x = batch[:-1]
        logits = net(x)
        sz = logits.size(0)
        loss = loss +nn.functional.cross_entropy(logits, batch[-sz:])
    loss = loss/batch_size
    loss.backward()
    optimizer.step()
    loss_save.append(loss.item())
    # monitor progress
    if epoch%100==0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))
        batch = next(g)
        logits = net(batch[:-1])
        _, i = logits.max(dim=1)
        plt.figure(figsize=[16,4])
        plt.plot(i.data.tolist())
        plt.plot(batch.data.tolist()[sum(net.dilations)+1:],'.',ms=1)
        plt.title('epoch {}'.format(epoch))
        plt.tight_layout()
        plt.show()