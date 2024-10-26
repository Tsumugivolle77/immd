import sys
import numpy as np
import scipy.sparse as sp
import cf_algorithms_to_complete as cfa
from sklearn.metrics.pairwise import cosine_similarity
import cf_data as cfd
from cf_config import config

if __name__ == '__main__':
    # test exercise 1, 3
    um_movielens    = cfd.read_movielens_file_and_convert_to_um(config.file_path, max_rows=config.max_rows)
    um_movielens_sp = sp.csr_matrix(np.nan_to_num(um_movielens))
    print(sys.getsizeof(um_movielens)/1024**2)
    print(sys.getsizeof(um_movielens_sp)/1024**2)
    # um_lecture    = get_um_by_name(config, 'lecture_1')
    # um_lecture_sp = sp.csr_matrix(np.nan_to_num(get_um_by_name(config, 'lecture_1')))

    # print(rate_all_items(um_lecture, 0, 2))
    # print(rate_all_items_for_sparse(um_lecture_sp, 0, 2))
    # print(cfa.rate_all_items(um_movielens, 0, 2))
    # print(cfa.rate_all_items_for_sparse(um_movielens_sp, 0, 2))

    # test exercise 2
    # test = 1
    #
    # if test == 1:
    #     x = [float(i + 1) for i in range(100)]
    #
    # elif test == 2:
    #     nan_index = [i + c for i in range(0, 100, 10) for c in range(2, 7)]
    #     print(nan_index)
    #     x = [float(i + 1) if i not in nan_index else np.nan for i in range(100)]
    #
    # y = sp.csr_matrix(np.nan_to_num(x[::-1])).transpose()
    # x = sp.csr_matrix(np.nan_to_num(x)).transpose()
    # _x = cfa.center_for_sparse(x).transpose()
    # _y = cfa.center_for_sparse(y).transpose()
    # print(f"x={x}\n y={y}")
    # print(f"centered cosine similarity for x and y is: {cfa.centered_cosine_sim(x, y)}")
    # print(f"centered cosine similarity for x and y by sklearn is: {cosine_similarity(_x, _y)}")