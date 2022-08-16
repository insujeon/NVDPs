import random
import torch

# 2D data preprocessing
def get_context_idx(N, device, num_point=784):
    # generate the indeces of the N context points in a flattened image
    idx = random.sample(range(0, num_point), N)
    idx = torch.tensor(idx, device=device)
    return idx

def generate_grid(h, w, device):
    rows = torch.linspace(0, 1, h, device=device)
    cols = torch.linspace(0, 1, w, device=device)
    grid = torch.stack([cols.repeat(h, 1).t().contiguous().view(-1), rows.repeat(w)], dim=1)
    grid = grid.unsqueeze(0)
    return grid

def generate_grid2(h, w, device):
    rows = torch.linspace(-1, 1, h, device=device)
    cols = torch.linspace(-1, 1, w, device=device)
    grid = torch.stack([cols.repeat(h, 1).t().contiguous().view(-1), rows.repeat(w)], dim=1)
    grid = grid.unsqueeze(0)
    return grid

def idx_to_y(idx, data):
    # get the [0;1] pixel intensity at each index
    y = torch.index_select(data, dim=1, index=idx)
    return y

def idx_to_x(idx, batch_size, x_grid):
    # From flat idx to 2d coordinates of the 28x28 grid. E.g. 35 -> (1, 7)
    # Equivalent to np.unravel_index()
    x = torch.index_select(x_grid, dim=1, index=idx)
    x = x.expand(batch_size, -1, -1)
    return x
