import torch
import numpy as np

# Tensor initialization
# from data

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# from numpy arrays

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# from other tensors

x_ones = torch.ones_like(x_data)
print(f"Ones: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random: \n {x_rand} \n")

# from probability distributions

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random: \n {rand_tensor} \n")
print(f"Ones: \n {ones_tensor} \n")
print(f"Zeros: \n {zeros_tensor}")

# Attributes of a Tensor

tensor = torch.rand(3,4)

print(f"Shape: {tensor.shape}")
print(f"Datatype: {tensor.dtype}")
print(f"Device: {tensor.device}")

# Operations on Tensors

# move tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# Indexing and Slicing

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# Combining Tensors (concatenation)
t1 = torch.cat([tensor, tensor, tensor], dim=0)
print(t1)
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Arithmetic operations

# addition
sum_tensor = tensor + tensor
neg_sum_tensor = tensor - tensor

# matrix multiplication
y1 = tensor @ tensor.T # .T is the transpose operation
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)


# the element-wise product
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# Single-element tensors

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In-place operations

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# Pytorch-NumPy: FULL COMPATIBILITY, SAME MEMORY LOCATION!

# Tensor -> NumPy array

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy -> Tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")