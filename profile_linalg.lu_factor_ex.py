from functools import partial
from torch.testing._internal.common_utils import  make_fullrank_matrices_with_distinct_singular_values
import torch
import itertools
import timeit

make_fullrank = make_fullrank_matrices_with_distinct_singular_values
make_A_cpu = partial(make_fullrank, device='cpu', dtype=torch.float32)


sizes = [(1024,1024), (128,128), (64,64)]
batches = [(),(1, 3), (2,)]

As_cpu = []
As_mps = []

for size, batch in itertools.product(sizes, batches):
    shape = batch + size
    As_cpu.append(make_A_cpu(*shape))
    As_mps.append(As_cpu[-1].clone().to("mps"))

# profile mps vs cpu in torch.linalg.lu_factor_ex
def lu_factor(tensors):
    for A in tensors:
        torch.linalg.lu_factor_ex(A)
    

def lu_factor_mps():
    lu_factor(As_mps)

def lu_factor_cpu():
    lu_factor(As_cpu)

print("MPS")
print(timeit.timeit(lu_factor_mps, number=100))
print("CPU")
print(timeit.timeit(lu_factor_cpu, number=100))