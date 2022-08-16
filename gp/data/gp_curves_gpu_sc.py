import pandas as pd
import numpy as np
import collections
import torch
from random import randint

# The (A)NP takes as input a `NPRegressionDescription` namedtuple with fields:
#   `query`: a tuple containing ((context_x, context_y), target_x)
#   `target_y`: a tensor containing the ground truth for the targets to be
#     predicted
#   `num_total_points`: A vector containing a scalar that describes the total
#     number of datapoints used (context + target)
#   `num_context_points`: A vector containing a scalar that describes the number
#     of datapoints used as context
# The GPCurvesReader returns the newly sampled data in this format at each
# iteration

NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points"),
)


class GPCurvesReader(object):
    """Generates curves using a Gaussian Process (GP).

  Supports vector inputs (x) and vector outputs (y). Kernel is
  mean-squared exponential, using the x-value l2 coordinate distance scaled by
  some factor chosen randomly in a range. Outputs are independent gaussian
  processes.
      """

    def __init__(
        self,
        batch_size,
        max_num_context,
        x_size=1,
        y_size=1,
        l1_scale=0.6,
        sigma_scale=1.0,
        random_kernel_parameters=True,
        testing=False,
    ):
        """Creates a regression dataset of functions sampled from a GP.

    Args:
      batch_size: An integer.
      max_num_context: The max number of observations in the context.
      x_size: Integer >= 1 for length of "x values" vector.
      y_size: Integer >= 1 for length of "y values" vector.
      l1_scale: Float; typical scale for kernel distance function.
      sigma_scale: Float; typical scale for variance.
      random_kernel_parameters: If `True`, the kernel parameters (l1 and sigma) 
          will be sampled uniformly within [0.1, l1_scale] and [0.1, sigma_scale].
      testing: Boolean that indicates whether we are testing. If so there are
          more targets for visualization.
        """
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._random_kernel_parameters = random_kernel_parameters
        self._testing = testing

    def _gaussian_kernel(self, xdata1, xdata2, l1, sigma_f, sigma_noise=2e-2):
        """Applies the Gaussian kernel to generate curve data.

    Args:
      xdata: Tensor of shape [B, num_total_points, x_size] with
          the values of the x-axis data.
      l1: Tensor of shape [B, y_size, x_size], the scale
          parameter of the Gaussian kernel.
      sigma_f: Tensor of shape [B, y_size], the magnitude
          of the std.
      sigma_noise: Float, std of the noise that we add for stability.

    Returns:
      The kernel, a float tensor of shape
      [B, y_size, num_total_points, num_total_points].
        """
        num_total_points = xdata1.shape[1]
        num_test_points = xdata2.shape[1]

        # Expand and take the difference
        xdata1 = xdata1.unsqueeze(1)  # [B, 1, num_total_points, x_size]
        xdata2 = xdata2.unsqueeze(2)  # [B, num_total_points, 1, x_size]
        diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]

        # [B, y_size, num_total_points, num_total_points, x_size]
        norm = (diff[:, None, :, :, :] / l1[:, :, None, None, :]) ** 2
        norm = torch.sum(norm, -1)  # [B, data_size, num_total_points, num_total_points]

        # [B, y_size, num_total_points, num_total_points]
        kernel = ((sigma_f) ** 2)[:, :, None, None] * torch.exp(-0.5 * norm)

        # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise ** 2) * torch.eye(num_test_points, num_total_points).cuda()

        return kernel

    def generate_curves(self):
        """Builds the op delivering the data.

    Generated functions are `float32` with x values between -2 and 2.
    
    Returns:
      A `CNPRegressionDescription` namedtuple.
        """
        num_context = randint(4, self._max_num_context) # m ~ [3,97)
        
        # If we are testing we want to have more targets and have them evenly
        # distributed in order to plot the function.
        if self._testing:
            num_total_points = 400
            num_target = num_total_points - num_context
            x_values = (torch.linspace(-2, 2, num_total_points).unsqueeze(0).repeat(self._batch_size, 1)).cuda()
            x_values = x_values.unsqueeze(-1)

        # During training the number of target points and their x-positions are
        # selected at random
        else:
            num_total_points = randint(num_context+1, 99) # n ~ [m+1, 100)
            num_target = num_total_points - num_context 
            x_values = (torch.rand((self._batch_size, num_total_points, self._x_size)) * 4 - 2).cuda()

        # Set kernel parameters
        # Either choose a set of random parameters for the mini-batch
        if self._random_kernel_parameters:
            l1 = (torch.rand((self._batch_size, self._y_size, self._x_size)).cuda()
                * (self._l1_scale - 0.1) + 0.1)
            sigma_f = (torch.rand((self._batch_size, self._y_size)).cuda()
                * (self._sigma_scale - 0.1) + 0.1)

        # Or use the same fixed parameters for all mini-batches
        else:
            l1 = torch.ones((self._batch_size, self._y_size, self._x_size)).cuda() * self._l1_scale
            sigma_f = torch.ones((self._batch_size, self._y_size)).cuda() * self._sigma_scale

        # Pass the x_values through the Gaussian kernel
        # [batch_size, y_size, num_total_points, num_total_points]
        kernel = self._gaussian_kernel(x_values, x_values, l1, sigma_f)

        # Calculate Cholesky, using double precision for better stability:
        cholesky = torch.cholesky(kernel)

        # Sample a curve
        # [batch_size, y_size, num_total_points, 1]
        y_values = torch.matmul(
            cholesky, torch.randn((self._batch_size, self._y_size, num_total_points, 1)).cuda()
        )
        # [batch_size, num_total_points, y_size]
        y_values = y_values.squeeze(3)
        y_values = y_values.permute(0, 2, 1)

        if self._testing:
            # In TESTING, 
            # Target set DO NOT contains Context set for calculating (normalized) predictive LL
            idx = torch.randperm(num_total_points)

            # Select the observations
            context_x = x_values[:, idx[:num_context]]
            context_y = y_values[:, idx[:num_context]]

            # Select the targets
            target_x = x_values[:, idx[num_context:]]
            target_y = y_values[:, idx[num_context:]]
                        
            kernel2 = self._gaussian_kernel(context_x, context_x, l1, sigma_f)
            cholesky2 = torch.linalg.cholesky(kernel2)
            
            
            kernel_s = self._gaussian_kernel(x_values, context_x, l1, sigma_f)
            Lk = torch.linalg.solve(cholesky2, kernel_s).squeeze()
            
            s2 = torch.max(torch.diag(kernel.squeeze()) - torch.sum(Lk**2, axis=0), torch.zeros_like(kernel.squeeze().diag()))
            stdv = torch.sqrt(s2)
            

        else:
            # In TRAINING, 
            # Target set contains context set in practice.

            # Select the targets which will consist of the context points as well as
            # some new target points
            target_x = x_values[:, : num_target + num_context, :]
            target_y = y_values[:, : num_target + num_context, :]

            # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        query = ((context_x, context_y), target_x, stdv)

        return NPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=target_x.shape[1],
            num_context_points=num_context)
