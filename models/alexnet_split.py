import torch
from torch import nn

class AlexNet_split(nn.Module):
    def __init__(self, first_cut=-1, last_cut=-1, num_classes=10):
        super().__init__()
        
        self.first_cut = first_cut
        self.last_cut = last_cut

        start = False
        end = False
        itter = 0

        if self.last_cut == -1:
            end = True

        if self.first_cut == -1:
            start = True

        if first_cut == -1:
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2))
        itter += 1

        if start or ((not start) and (self.first_cut == itter)): 
            if end or ((not end) and (self.last_cut > itter)):
                self.layer2 = nn.Sequential(
                    nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2))
                start = True
            else:
                # reached end
                return 
        itter += 1

        if start or ((not start) and (self.first_cut == itter)): 
            if end or ((not end) and (self.last_cut > itter)):
                self.layer3 = nn.Sequential(
                    nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True))
                start = True
            else:
                # reached end
                return 
        itter += 1

        if start or ((not start) and (self.first_cut == itter)): 
            if end or ((not end) and (self.last_cut > itter)):        
                self.layer4 = nn.Sequential(
                    nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True))
                start = True
            else:
                # reached end
                return 
        itter += 1

        if start or ((not start) and (self.first_cut == itter)): 
            if end or ((not end) and (self.last_cut > itter)):        
                self.layer5 = nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2))
                start = True
            else:
                # reached end
                return 
        itter += 1

        if start or ((not start) and (self.first_cut == itter)): 
            if end or ((not end) and (self.last_cut > itter)): 
                #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
                self.fc = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(256 * 8 * 8, 4096),
                    nn.ReLU(inplace=True))
                start = True
            else:
                # reached end
                return 
        itter += 1


        if start or ((not start) and (self.first_cut == itter)): 
            if end or ((not end) and (self.last_cut > itter)): 
                self.fc1 = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True))
                start = True
            else:
                # reached end
                return 
        itter += 1

        if start or ((not start) and (self.first_cut == itter)): 
            if end or ((not end) and (self.last_cut > itter)):
                self.fc2= nn.Sequential(
                    nn.Linear(4096, num_classes))
                start = True
            else:
                # reached end
                return 
        itter += 1
        
    def forward(self, x):
        layer = 0
        start = False
        end = False

        if self.last_cut == -1:
            end = True

        if self.first_cut == -1:
            start = True
        #layer 1
        if start or ((not start) and (self.first_cut == layer)): 
            if end or ((not end) and (self.last_cut > layer)):
                x = self.layer1(x)
        
                start = True
            else:
                # reached end
                return x
        layer += 1
        #layer 2
        if start or ((not start) and (self.first_cut == layer)): 
            if end or ((not end) and (self.last_cut > layer)):
                x = self.layer2(x)
        
                start = True
            else:
                # reached end
                return x
        layer += 1
        #layer 3
        if start or ((not start) and (self.first_cut == layer)): 
            if end or ((not end) and (self.last_cut > layer)):
                x = self.layer3(x)
        
                start = True
            else:
                # reached end
                return x
        layer += 1
        #layer 4
        if start or ((not start) and (self.first_cut == layer)): 
            if end or ((not end) and (self.last_cut > layer)):
                x = self.layer4(x)
        
                start = True
            else:
                # reached end
                return x
        layer += 1
        #layer 5
        if start or ((not start) and (self.first_cut == layer)): 
            if end or ((not end) and (self.last_cut > layer)):
                x = self.layer5(x)
        
                start = True
            else:
                # reached end
                return x
        layer += 1

        #layer 6
        if start or ((not start) and (self.first_cut == layer)): 
            if end or ((not end) and (self.last_cut > layer)):
                #x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
        
                start = True
            else:
                # reached end
                return x
        layer += 1

        #layer 7
        if start or ((not start) and (self.first_cut == layer)): 
            if end or ((not end) and (self.last_cut > layer)):
                x = self.fc1(x)
        
                start = True
            else:
                # reached end
                return x
        layer += 1
        
        #layer 8
        if start or ((not start) and (self.first_cut == layer)): 
            if end or ((not end) and (self.last_cut > layer)):
                x = self.fc2(x)

                start = True
            else:
                # reached end
                return x
        return x
    

def get_AlexNet_split(first_cut=-1, last_cut=-1, num_classes=10):

    model_part_a = AlexNet_split(-1, first_cut, num_classes)
    model_part_b = AlexNet_split(first_cut, last_cut, num_classes)
    model_part_c = AlexNet_split(last_cut, -1, num_classes)

    return (model_part_a, model_part_b, model_part_c)
