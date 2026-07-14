import torch
import pandas as pd

v=torch.tensor([3.0,-4.0])
print(torch.norm(v))
print(torch.norm(v,p=1))
print(torch.norm(v,float('inf')))
