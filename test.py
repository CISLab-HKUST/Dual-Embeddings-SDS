import torch
from argparse import ArgumentParser, Namespace
# a = 'a {} person in a hat'
# b = ['tk1', 'tk2']

# print(a.format(" ".join(b)))
# a = torch.nn.Parameter(torch.zeros((1, 77, 1024)).clone()[None])

# b = torch.randn((2, 77, 1024))

# a.data = b[None, 0]

# print(a, b[0])

from argument import TrainEmbeddingParams
parser = ArgumentParser(description="Training script parameters")
a = TrainEmbeddingParams(parser)
print(a)