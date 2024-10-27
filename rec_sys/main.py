from sys import exception

import numpy as np
import scipy.sparse as sp
import cf_algorithms_to_complete as cfa
from sklearn.metrics.pairwise import cosine_similarity
import data_util as cfd
import config
from rec_sys.cf_algorithms_to_complete import get_rated_by

# test exercise 1, 3, 4
def test_rate_all_items(which="lecture", exercise=1):
    print(f">>> Testing codes for exercise {exercise} working on {which} data")

    if which == "lecture":
        um_lecture = cfd.get_um_by_name(config.ConfigCf, 'lecture_1')

        if exercise == 1:
            print(cfa.rate_all_items(um_lecture, 0, 2))
        elif exercise == 3:
            um_lecture_sp = sp.csr_matrix(np.nan_to_num(um_lecture))
            print(cfa.rate_all_items_for_sparse(um_lecture_sp, 0, 2))
        elif exercise == 4:
            um_lecture_sp = sp.csr_matrix(np.nan_to_num(um_lecture))
            rated_by = get_rated_by(um_lecture_sp)
            print(cfa.rate_all_items_plusplus(um_lecture_sp, rated_by, 0, 2))
        else:
            raise exception(f"This test is not designed for exercise {exercise}")

    elif which == "movielens":
        um_movielens    = cfd.read_movielens_file_and_convert_to_um(config.ConfigCf.file_path, max_rows=config.ConfigCf.max_rows)

        if exercise == 1:
            print(cfa.rate_all_items(um_movielens, 0, 2))
        elif exercise == 3:
            um_movielens_sp = sp.csr_matrix(np.nan_to_num(um_movielens))
            print(cfa.rate_all_items_for_sparse(um_movielens_sp, 0, 2))
        elif exercise == 4:
            um_movielens_sp = sp.csr_matrix(np.nan_to_num(um_movielens))
            rated_by = get_rated_by(um_movielens_sp)
            print(cfa.rate_all_items_plusplus(um_movielens_sp, rated_by, 0, 2))
        else:
            raise exception(f"This test is not designed for exercise {exercise}")


# test exercise 2
def test_centered_cosine_similarity(sample=1):
    print(f">>> Testing codes for exercise 2 w/ {sample}")

    if sample == 1:
        x = [float(i + 1) for i in range(100)]
    elif sample == 2:
        nan_index = [i + c for i in range(0, 100, 10) for c in range(2, 7)]
        print(nan_index)
        x = [float(i + 1) if i not in nan_index else np.nan for i in range(100)]

    y = sp.csr_matrix(np.nan_to_num(x[::-1])).transpose()
    x = sp.csr_matrix(np.nan_to_num(x)).transpose()
    _x = cfa.center_for_sparse(x).transpose()
    _y = cfa.center_for_sparse(y).transpose()

    print(f"x={x}\n y={y}")
    print(f"centered cosine similarity for x and y is: {cfa.centered_cosine_sim(x, y)}")
    # calculate a result for compare with sklearn
    print(f"centered cosine similarity for x and y by sklearn is: {cosine_similarity(_x, _y)}")


# test exercise 4 `get_rated_by`
def test_get_rated_by():
    print(">>> Testing `get_rated_by`")
    a = sp.csr_matrix([
        [0, 1., 0,],
        [1., 0, .4],
        [1., -1, .5],
        [0., -2, .5],
    ])

    rated_by = get_rated_by(a)
    print(f"rated_by is: {rated_by}")


if __name__ == '__main__':
    # test exercise 1, 3
    test_rate_all_items("movielens", 4)
    # test_rate_all_items("lecture", 3)
    # test_rate_all_items("lecture", 4)

    # test exercise 2
    # test_centered_cosine_similarity(2)

    # test exercise 4
    # test_get_rated_by()