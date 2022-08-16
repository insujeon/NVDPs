import numpy as np
#from scipy.ndimage import rotate
import torch
#from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

import glob
from PIL import Image

class OmniDataset(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self, train=True):
        data = np.load("./data/data.npy")
        data = torch.from_numpy(np.reshape(data, newshape=(1622, 20, 28, 28, 1)))

        if train:
            data = data[:1200,:,:,:,:]
        else:
            data = data[1200:,:,:,:,:]
        
        self.data = (data.permute(0,1,4,2,3)-1).abs().view(-1,1,28,28) 

        self.len = self.data.size(0)

    def __getitem__(self, index):

        return self.data[index], index

    def __len__(self):
        return self.len

# train_dataset = OmniDataset(train=True)
# train_loader = DataLoader(dataset=train_dataset,batch_size=32,shuffle=True,num_workers=1)
# test_dataset = OmniDataset(train=False)
# test_loader = DataLoader(dataset=test_dataset,batch_size=32,shuffle=True,num_workers=1)
# img1 = (x[0][0]-1).abs() #torch.Size([3,28,28]
# # img1 = img1.numpy() # TypeError: tensor or list of tensors expected, got <class 'numpy.ndarray'>
# save_image(img1, 'img1.png')

class CelebADataset(Dataset):
    """CelebA dataset."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        path_to_data : string
            Path to CelebA data files.

        subsample : int
            Only load every |subsample| number of images.

        transform : torchvision.transforms
            Torchvision transforms to be applied to each image.
        """
        self.img_paths = glob.glob(path_to_data + '/*.jpg')[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = Image.open(sample_path)

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0
