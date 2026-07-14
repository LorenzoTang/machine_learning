import  torch
X=torch.arange(12).reshape(3,4)
A = X.numpy()
B = torch.tensor(A)
print(type(A), type(B))

a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))