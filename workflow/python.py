import torch

# Create a tensor
a = torch.tensor([[[3,4,6],[1,2,3],[4,5,6]],
               [[1,2,3],[4,5,6],[7,8,9]],
               [[1,2,3],[4,5,6],[7,8,9]]], dtype = torch.int8)
print(a)
print(a.dtype)
print(a.size())
print(a.shape)
print(a.dim())
print(a.ndim)
print(a.numel())
print(a.stride())
print(a.device)

# Create a random tensor
b = torch.rand([5,3,3], dtype = torch.float32)
print(b)
# Create a scalar tensor
e = torch.tensor(5)
print(e)
print(e.item())


# Create a tensor with all zeros
c = torch.zeros([3,3,3], dtype = torch.float16)
print(c)
# Create a tensor with all ones
d = torch.ones(3,3,3, dtype = torch.int8)
print(d)
# Create a tensor with the identity matrix
e = torch.eye(3, dtype = torch.int8)
print(e)

# Create a tensor with a range
f = torch.arange(0,10)
print(f)
# Create a tensor with a range and a step
g = torch.arange(0,10,2)
print(g)

# Create a tensor like another tensor
h = torch.zeros_like(a)
print(h)
i = torch.ones_like(a)
print(i)
j = torch.rand_like(b)
print(j)

# Create a tensor with a linspace
k = torch.linspace(3,10,5, dtype = torch.int8)
print(k)

# 3 most  common errors
# 1. Tensor not right datatype.
# 2. Tensor not right shape.
# 3. Tensor not on right device.
l = torch.tensor([1,2,3],
                  dtype = None, # What datatype is your tensor.
                  device = None, # What devise is your tensor on.
                  requires_grad = False) # Whether to track gradients with this tensor operations or not.
print(l)
print(l.dtype)

# Changing tensor datatype
m = torch.tensor([1,2,3], dtype = torch.float32)
print(m)
n = m.int()
print(n)
o = m.float()
print(o)
p = m.double()
print(p)
q = m.to(torch.float64)
print(q)
r = m.type(torch.int16)
print(r)
print(o*p)

# Tensor operations
s = torch.tensor([1,2,3])
t = torch.tensor([4,5,6])

# Elementwise multiplication (mul, div, add, sub) / Matrix multiplication (mm, matmul, @)
aa = s * t
print(aa)
bb = torch.mul(s,t)
print(bb)
cc = torch.mul(s,t) #(s, t, alpha = 2) for add or sub = s + t*2
print(cc)

# Inplace elementwise multiplication (mul, div, add, sub) / Inplace matrix multiplication (mm, matmul, @)
s.mul_(t)
print(s)
s *= t
print(s)

# Transpose
u = torch.tensor([[1,2,3],[4,5,6]])
print(u)
v = u.t()
y = u.T
print(y)
print(v)
w = u.transpose(0,1)
print(w)
torch.transpose(u, 0, 1)
print(u)
x = u.permute(1,0)
print(x)

# Reshape
z = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=torch.float32)
print(z)
z = z.view(1,3,3)
print(z)
z = z.view(9)
print(z)
z= z.reshape(9,1)
print(z)

# Squeeze
z = z.squeeze()
print(z)
z = z.unsqueeze(0)
print(z)

# Concatenate
z = torch.tensor([1,2,1,3])
z = torch.cat((z,z), dim = 0)
print(z)
z = torch.cat((z,z), dim = 0)
print(z)
z = torch.cat((z,z), dim = 0)
print(z)

# Stack
z = torch.tensor([1,2,3])
z = torch.stack((z,z), dim = 0)
print(z)
z = torch.stack((z,z), dim = 0)
print(z)
z = torch.stack((z,z), dim = 0)
print(z)

# Split
z = torch.tensor([1,2,3,4,5,6])
z = torch.split(z, 2, dim = 0)
print(z)

# Chunk
z = torch.tensor([1,2,3,4,5,6])
z = torch.chunk(z, 3, dim = 0)
print(z)

# Indexing
z = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(z[0])
print(z[0,0])
print(z[0,:])
print(z[0:2,0:2])

# Masking
z = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
mask = z > 5
print(mask)
print(z[mask])
print(z[z>5])

# Sorting
z = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(z)
print(torch.sort(z, dim = 0))
print(torch.sort(z, dim = 1))

# Max and Min
z = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(z)
print(torch.max(z))
print(torch.min(z))
print(torch.max(z, dim = 0))
print(torch.min(z, dim = 1))

# Sum and Mean
z = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype = torch.float32)
print(z)
print(torch.sum(z))
print(torch.mean(z))
print(torch.sum(z, dim = 0))
print(torch.mean(z, dim = 1))

# Argmax and Argmin
z = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(z)
print(torch.argmax(z))
print(torch.argmin(z))
print(torch.argmax(z, dim = 0))
print(torch.argmin(z, dim = 1))

# Dot product
z = torch.tensor([1,2,3])
print(z)
print(z.dot(z))
print(z @ z)

# Cross product
z = torch.tensor([1,2,3])
print(z)
print(z.cross(z))

# Inverse
z = torch.tensor([[1,2,4],[4,5,6],[7,8,9]], dtype = torch.float32)
print(z)
print(torch.inverse(z))

# Determinant
z = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype = torch.float32)
print(z)
print(torch.det(z))

# Eigenvalues and Eigenvectors
z = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype = torch.float32)
print(z)

# Note: torch.eig() is deprecated and torch.linalg.eigvals() and torch.linalg.eig() should be used instead.

#L, _ = torch.eig(A) should be replaced with:
#L_complex = torch.linalg.eigvals(A)

#L, V = torch.eig(A, eigenvectors=True) should be replaced with:
#L_complex, V_complex = torch.linalg.eig(A)

print(torch.linalg.eig(z))

# Norm
z = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype = torch.float32)
print(z)
print(torch.norm(z))
print(torch.norm(z, dim = 0))
print(torch.norm(z, dim = 1))

# Trigonometric functions
z = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype = torch.float32)
print(z)
print(torch.sin(z))
print(torch.cosh(z))
print(torch.exp(z))
print(torch.log(z))
print(torch.log10(z))

# Rounding
z = torch.tensor([[1.1,2.2,3.3],[4.4,5.5,6.6],[7.7,8.8,9.9]], dtype = torch.float32)
print(z)
print(torch.round(z))
print(torch.floor(z))
print(torch.ceil(z))
print(torch.trunc(z))

# Absolute value
z = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype = torch.float32)
print(z)
print(torch.abs(z))

# Clamp
z = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype = torch.float32)
print(z)
print(torch.clamp(z, min = 3, max = 7))

# Comparison
z = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype = torch.float32)
print(z)
print(z == 5)
print(z != 5)
print(z > 5)
print(z < 5)
print(z >= 5)
print(z <= 5)

# Logical
z = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype = torch.float32)
print(z)
print(torch.logical_and(z > 3, z < 7))
print(torch.logical_or(z > 3, z < 7))
print(torch.logical_not(z > 3))

# Bitwise
z = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype = torch.int32)
print(z)
print(z & 5)
print(z | 5)
print(z ^ 5)
print(~z)

# Shift
z = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype = torch.int32)
print(z)
print(z << 1)
print(z >> 1)

# Set manual seed
torch.manual_seed(42)   
print(torch.rand(2))
torch.manual_seed(42)
print(torch.rand(2))
