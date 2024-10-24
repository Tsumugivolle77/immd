# Artur Andrzejak, October 2024
# Algorithms for collaborative filtering

import numpy as np
import scipy.sparse as sp
from numpy.ma.core import nonzero
from scipy.sparse import csr_matrix


def complete_code(message):
    raise Exception(f"Please complete the code: {message}")
    return None


def center_and_nan_to_zero(matrix, axis=0):
    """ Center the matrix and replace nan values with zeros"""
    # Compute along axis 'axis' the mean of non-nan values
    # E.g. axis=0: mean of each column, since op is along rows (axis=0)
    means = np.nanmean(matrix, axis=axis)
    # Subtract the mean from each axis
    matrix_centered = matrix - means
    return np.nan_to_num(matrix_centered)


def cosine_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def fast_cosine_sim(utility_matrix, vector, axis=0):
    """ Compute the cosine similarity between the matrix and the vector"""
    # Compute the norms of each column
    norms = np.linalg.norm(utility_matrix, axis=axis)
    um_normalized = utility_matrix / norms
    # Compute the dot product of transposed normalized matrix and the vector
    dot = um_normalized.transpose().dot(vector)
    # Scale by the vector norm
    scaled = dot / np.linalg.norm(vector)
    return scaled


def center_for_sparse(matrix: sp.csr_matrix):
    matrix_centered = matrix.copy()
    _, cols = matrix_centered.get_shape()
    for i in range(cols):
        this_col = matrix_centered.getcol(i)
        nonzeros = this_col.nonzero()
        mean_v = this_col[nonzeros].mean()
        this_col[nonzeros] -= mean_v
        matrix_centered[:, i] = this_col

    return matrix_centered


# The input vectors are sparse with all the nan replaced by 0
def centered_cosine_sim(u, v):
    _u = center_for_sparse(u)
    _v = center_for_sparse(v)
    return np.dot(_u, _v) / (np.linalg.norm(_u) * np.linalg.norm(_v))


# The input vector and matrix are sparse with all the nan replaced by 0
def fast_centered_cosine_sim(utility_matrix: sp.csr_matrix, vector, axis=0):
    # Process the utility matrix and vector (if necessary) with NaN entries
    _utility_matrix = center_for_sparse(utility_matrix)
    _vector         = center_for_sparse(vector)
    """ Compute the cosine similarity between the matrix and the vector"""
    # Compute the norms of each column
    norms = sp.linalg.norm(_utility_matrix, axis=axis)
    um_normalized = _utility_matrix / norms
    # Compute the dot product of transposed normalized matrix and the vector
    dot = um_normalized.transpose().dot(_vector)
    # Scale by the vector norm
    scaled = dot / sp.linalg.norm(_vector)
    return scaled


# Implement the CF from the lecture 2 for sparse utility matrix
def rate_all_items_for_sparse(orig_utility_matrix: sp.csr_matrix, user_index, neighborhood_size):
    print(f"\n>>> CF computation for Sparse UM w/ shape: "
          + f"{orig_utility_matrix.shape}, user_index: {user_index}, neighborhood_size: {neighborhood_size}\n")

    user_col = orig_utility_matrix[:, user_index]
    # Compute the cosine similarity between the user and all other users
    similarities = fast_centered_cosine_sim(orig_utility_matrix, user_col)

    def rate_one_item(item_index):
        # If the user has already rated the item, return the rating
        if orig_utility_matrix[item_index, user_index] != 0:
            return orig_utility_matrix[item_index, user_index]

        # Find the indices of users who rated the item
        users_who_rated = orig_utility_matrix[item_index, :].nonzero()[1]
        # From those, get indices of users with the highest similarity (watch out: result indices are rel. to users_who_rated)
        best_among_who_rated = similarities[users_who_rated].toarray()[:, 0].argsort()
        # Select top neighborhood_size of them
        best_among_who_rated = best_among_who_rated[-neighborhood_size:]
        # Convert the indices back to the original utility matrix indices
        best_among_who_rated = users_who_rated[best_among_who_rated]
        best_among_who_rated = best_among_who_rated[best_among_who_rated.nonzero()]
        # print(f"similarities among who rated for {similarities[item_index, :]}", best_among_who_rated)
        if best_among_who_rated.size > 0:
            # Compute the rating of the item
            nearest_sim = similarities[best_among_who_rated]
            rating_of_item = orig_utility_matrix[item_index, best_among_who_rated].dot(nearest_sim) / sum(abs(nearest_sim))
        else:
            rating_of_item = np.nan
        print(f"item_idx: {item_index}, neighbors: {best_among_who_rated}, rating: {rating_of_item}")
        return rating_of_item

    num_items = orig_utility_matrix.shape[0]

    # Get all ratings
    ratings = list(map(rate_one_item, range(num_items)))
    return ratings


# Implement the CF from the lecture 1
def rate_all_items(orig_utility_matrix, user_index, neighborhood_size):
    print(f"\n>>> CF computation for UM w/ shape: "
          + f"{orig_utility_matrix.shape}, user_index: {user_index}, neighborhood_size: {neighborhood_size}\n")

    clean_utility_matrix = center_and_nan_to_zero(orig_utility_matrix)
    """ Compute the rating of all items not yet rated by the user"""
    user_col = clean_utility_matrix[:, user_index]
    # Compute the cosine similarity between the user and all other users
    similarities = fast_cosine_sim(clean_utility_matrix, user_col)

    def rate_one_item(item_index):
        # If the user has already rated the item, return the rating
        if not np.isnan(orig_utility_matrix[item_index, user_index]):
            return orig_utility_matrix[item_index, user_index]

        # Find the indices of users who rated the item
        users_who_rated = np.where(np.isnan(orig_utility_matrix[item_index, :]) == False)[0]
        # From those, get indices of users with the highest similarity (watch out: result indices are rel. to users_who_rated)
        best_among_who_rated = similarities[users_who_rated].argsort()
        # Select top neighborhood_size of them
        best_among_who_rated = best_among_who_rated[-neighborhood_size:]
        # Convert the indices back to the original utility matrix indices
        best_among_who_rated = users_who_rated[best_among_who_rated]
        # Retain only those indices where the similarity is not nan
        best_among_who_rated = best_among_who_rated[np.isnan(similarities[best_among_who_rated]) == False]
        if best_among_who_rated.size > 0:
            # Compute the rating of the item
            rating_of_item = orig_utility_matrix[item_index, best_among_who_rated].dot(similarities[best_among_who_rated]) / sum(abs(similarities[best_among_who_rated]))
        else:
            rating_of_item = np.nan
        print(f"item_idx: {item_index}, neighbors: {best_among_who_rated}, rating: {rating_of_item}")
        return rating_of_item

    num_items = orig_utility_matrix.shape[0]

    # Get all ratings
    ratings = list(map(rate_one_item, range(num_items)))
    return ratings

