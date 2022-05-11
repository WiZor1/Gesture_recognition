import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet18


class SignDataset(Dataset):
    def __init__(self,
                 df,
                 transforms=None,
                 path_name='Img',
                 class_name='Class_int',
                 ):
        self.df = df.reset_index(drop=True)
        self._transforms = transforms
        self._path_name = path_name
        self._class_name = class_name

    def __getitem__(self, idx):
        img = self.df[self._path_name][idx]
        label = torch.tensor(self.df[self._class_name][idx])
        if self._transforms is not None:
            img = self._transforms(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label
    
    def __len__(self):
        return len(self.df)


class MyResNet(nn.Module):
    def __init__(self, out_features, *argw, **kwargs):
        super().__init__(*argw, **kwargs)

        self.model = resnet18()
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                     bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=out_features, bias=True)
    
    def forward(self, x):
        return self.model(x)

df_transforms = transforms.Compose([
                                    transforms.Resize((240, 320)),
                                    transforms.ToTensor()
                                    ]
                                   )

if __name__ == '__main__':
    sign_detection = MyResNet(out_features=8).to(device)