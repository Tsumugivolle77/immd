import numpy as np
import scipy.sparse as sp
from cf_algorithms_to_complete import centered_cosine_sim

if __name__ == '__main__':
    test = 1

    if test == 1:
        x = [float(i + 1) for i in range(100)]

    elif test == 2:
        nan_index = [i + c for i in range(0, 100, 10) for c in range(2, 7)]
        print(nan_index)
        x = [float(i + 1) for i in range(100) if i not in nan_index]

    y = sp.csr_matrix(np.nan_to_num(x[::-1])).transpose()
    x = sp.csr_matrix(np.nan_to_num(x)).transpose()
    print(f"x={x}\n y={y}")
    print(f"centered cosine similarity for x and y is: {centered_cosine_sim(x, y)}")