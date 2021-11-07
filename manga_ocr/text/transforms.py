import torch


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.15):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        tensor = tensor + torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor, min=0, max=1.0)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)