import numpy as np
import torch
from math import pi
from torch.utils.data import Dataset
    
class TrigonometryData(Dataset):
    """
    Dataset of functions f(x) = a * f(x - b) where f is one of the the sin, cosine, or thanh function.
    a and b are randomly sampled. f(x) is The function is evaluated from -pi to pi.

    Parameters
    ----------
    amplitude_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the function is sampled.
    shift_range : tuple of float
        Defines the range from which the shift (i.e. b) of the sine function is sampled.
    num_samples : int
        Number of samples of the function contained in dataset.
    num_points : int
        Number of points at which to evaluate f(x) for x in [-pi, pi].
    """

    def __init__(self, amplitude_range=(1, 2), shift_range=(-.1,.1), num_samples=2000, num_points=400):
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []
        a_min, a_max = amplitude_range
        b_min, b_max = shift_range
        for i in range(num_samples):
            a = (a_max - a_min) * np.random.rand() + a_min
            b = (b_max - b_min) * np.random.rand() + b_min
            x = torch.linspace(-pi, pi, num_points).unsqueeze(1)
            y = a * torch.sin(2*x-b*pi)
            self.data.append((x,y))          
            y2 = a * torch.cos(2*x-b*pi)
            self.data.append((x,y2))
            y3= a * torch.tanh(2*x-b*pi)
            self.data.append((x,y3))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples
    
# dataset Visualization
# dataset = TrigonometryData( amplitude_range=(1.5, 2), shift_range=(-.1, .1), num_samples=1000, num_points=200)
# # Visualize data samples
# for i in range(50):
#     x, y = dataset[i] 
#     plt.plot(x.numpy(), y.numpy(), c='b', alpha=0.3)
#     plt.xlim(-pi, pi)
#     plt.ylim(-2, 2)
#
# batch_size = 16 # 32
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# x,y = iter(data_loader).next()
# jth = 7
# plt.plot(x[jth].squeeze().numpy(), y[jth].squeeze().numpy(), c='b', alpha=0.3)
# plt.xlim(-pi, pi)
# plt.ylim(-2, 2)
# x.size(), y.size()

# Training Pipeline    
#     for batch_idx, (x, y) in enumerate(data_loader):
#         model.train()        
#         num_context = randint(4, args.max_num_context) # m ~ (3,97)
#         num_total_points = randint(num_context+1, 100) # n ~ [m+1, 100) #99
#         num_target = num_total_points - num_context 

#         loc = np.random.choice(200, size=num_context+num_target, replace=False)
#         x_c = x[:, loc[:num_context], :].cuda()
#         y_c = y[:, loc[:num_context], :].cuda()
#         x_t = x[:, loc, :].cuda()
#         y_t = y[:, loc, :].cuda() 
        
#         model.apply(renorm_weights)
#         optimizer.zero_grad()
            
#         mu, std = model(x_c, y_c, x_t, y_t)        