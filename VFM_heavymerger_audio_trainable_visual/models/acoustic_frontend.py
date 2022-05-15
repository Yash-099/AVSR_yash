import torch.nn as nn
import torch.nn.functional as F



class ResNetLayer1D(nn.Module):

    """
    A ResNet layer used to build the ResNet network.
    Architecture:
    --> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
     |                        |   |                                    |
     -----> downsample ------>    ------------------------------------->
    """

    def __init__(self, inplanes, outplanes, stride):
        super(ResNetLayer1D, self).__init__()
        self.conv1a = nn.Conv1d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1)#, bias=False)
        self.bn1a = nn.BatchNorm1d(outplanes)#, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv1d(outplanes, outplanes, kernel_size=3, stride=1, padding=1)#, bias=False)
        self.stride = stride
        self.downsample = nn.Conv1d(inplanes, outplanes, kernel_size=1, stride=stride)#, bias=False)
        self.outbna = nn.BatchNorm1d(outplanes)#, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv1d(outplanes, outplanes, kernel_size=3, stride=1, padding=1)#, bias=False)
        self.bn1b = nn.BatchNorm1d(outplanes)#, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv1d(outplanes, outplanes, kernel_size=3, stride=1, padding=1)#, bias=False)
        self.outbnb = nn.BatchNorm1d(outplanes)#, momentum=0.01, eps=0.001)
        return



    def forward(self, inputBatch):
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch




class ResNet1D(nn.Module):

    """
    An 18-layer ResNet architecture.
    """

    def __init__(self):
        super(ResNet1D, self).__init__()
        self.layer1 = ResNetLayer1D(64, 64, stride=2)
        self.layer2 = ResNetLayer1D(64, 128, stride=2)
        self.layer3 = ResNetLayer1D(128, 256, stride=2)
        self.layer4 = ResNetLayer1D(256, 512, stride=2)
        # self.avgpool = nn.AvgPool2d(kernel_size=(4,4), stride=(1,1))
        return


    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        # outputBatch = self.avgpool(batch)
        return batch


class AcousticFrontend(nn.Module):

    """
    A visual feature extraction module. Generates a 512-dim feature vector per video frame.
    Architecture: A 3D convolution block followed by an 18-layer ResNet.
    """

    def __init__(self):
        super(AcousticFrontend, self).__init__()
        self.frontend1D = nn.Sequential(
                            nn.Conv1d(1, 64, kernel_size=80, stride=4, bias=True),
                            nn.BatchNorm1d(64),
                            nn.ReLU()
                        )
        self.resnet = ResNet1D()
        self.downsample = nn.AvgPool1d(kernel_size=10, stride=10, padding=1)
        return


    def forward(self, inputBatch):
        # print(inputBatch.shape)
        inputBatch = inputBatch.unsqueeze(-1)
        inputBatch = inputBatch.transpose(1, 2)
        batchsize = inputBatch.shape[0]
        batch = self.frontend1D(inputBatch)

        # batch = batch.transpose(1, 2)
        # batch = batch.reshape(batch.shape[0]*batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])
        outputBatch = self.resnet(batch)
        outputBatch = self.downsample(outputBatch)
        outputBatch = outputBatch.reshape(batchsize, -1, 512)
        # outputBatch = outputBatch.transpose(1 ,2)
        # outputBatch = outputBatch.transpose(1, 2)
        return outputBatch





# import torch
# model = AcousticFrontend()

# kachra = model(torch.randn(2,640*62))

# print(kachra.shape)