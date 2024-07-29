import torch
import torch.nn.functional as functional
import numpy as np
from torch.utils.data import Dataset, IterableDataset


class ReProjectorDataset(IterableDataset):
    def __init__(self,
                 d=64,
                 G=512,
                 num_active_features=5,
                 device='cpu',
                 dtype=torch.float32,
                 proj=None,
                 target_proj=None,
                ):
        """
        Follows https://arxiv.org/pdf/2211.09169.pdf, specifically the re-projector task.

        args:
            d: number of dimensions
            G: number of ground truth features
            proj: np projection matrix. If None, randomly initialized.
            target_proj: np projection matrix. If None, randomly initialized.
        """
        self.d = d
        self.G = G

        # project the ground truth features into a lower dimensional space
        if proj is None:
            self.proj = torch.randn(G, d).to(device).to(dtype)
        else:
            self.proj = torch.tensor(proj).to(device).to(dtype)

        # if target_proj is None:
        #     self.target_proj = torch.randn(G, d).to(device).to(dtype)
        # else:
        #     self.target_proj = torch.tensor(target_proj).to(device).to(dtype)

        # probability of a feature being active by zipf's law
        self.probs = torch.tensor([(i+1)**(-1.1) for i in range(G)]).to(dtype)
        self.probs = (self.probs * num_active_features) / self.probs.sum()
        self.probs = self.probs.to(device)

    def __iter__(self):
        while True:
            # Sample G-dimensional binary random variable using the precomputed probabilities
            binary_rv = torch.distributions.Bernoulli(self.probs).sample()

            # project to lower dimension
            sample = binary_rv @ self.proj

            # # project to target
            # target = binary_rv @ self.target_proj

            yield sample # , target
    
    def get_batch(self, batch_size):
        binary_rv = torch.distributions.Bernoulli(self.probs).sample((batch_size,))
        # binary_rv.shape is (batch_size, G)

        # project to lower dimension
        sample = binary_rv @ self.proj

        # # project to target
        # target = binary_rv @ self.target_proj

        return sample #, target


if __name__ == '__main__':
    dataset = ReProjectorDataset()
    b = dataset.get_batch(32)
    print(b.shape)
    print(b)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
