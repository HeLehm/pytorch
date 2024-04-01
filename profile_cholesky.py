import timeit
import torch
import itertools



def get_positive_definite_matrix(shape):
    matrix_dims = (shape[-2], shape[-1])
    batch_dims = shape[:-2]
    A = torch.randn(*matrix_dims)
    A = torch.mm(A, A.T)
    return A.expand(*batch_dims, *matrix_dims)




batches = [(2,), (),  (3, 2)]
n = [10, 100, 1000]


for batch, n in itertools.product(batches, n):
    shape = batch + (n, n)
    A = get_positive_definite_matrix(shape)

    # time cpu
    cpu_time = timeit.timeit(lambda: torch.linalg.cholesky(A), number=100)

    A_mps = A.to("mps")
    # time mps
    mps_time = timeit.timeit(lambda: torch.linalg.cholesky(A_mps), number=100)

    print(f"CPU time for {shape}: {cpu_time}")
    print(f"MPS time for {shape}: {mps_time}")
    print(f"Speedup for {shape}: {cpu_time/mps_time}")
    print()