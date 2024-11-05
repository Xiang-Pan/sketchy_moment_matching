import torch
import numpy as np
from numpy.linalg import qr, norm, eigh

def cov_greedy(X, m, alpha=0.5):
    """
    Select m rows from X starting from an empty set, adding rows that both contribute new directions
    and have sufficient alignment with the previously selected rows' subspace.

    Args:
    X: Data matrix with instances as rows.
    m: Number of rows to select.
    alpha: Balance factor between new direction strength and alignment with existing subspace (0 <= alpha <= 1).

    Returns:
    selected_indices: Indices of the selected rows.
    """
    n_rows, n_cols = X.shape
    if m > n_rows:
        raise ValueError("m cannot be greater than the number of rows in X.")

    selected_indices = []

    # Start by selecting the row with the maximum norm
    norms = np.linalg.norm(X, axis=1)
    max_norm_index = np.argmax(norms)
    selected_indices.append(max_norm_index)

    # Initialize QR decomposition with the first selected row
    Q, _ = qr(X[max_norm_index].reshape(1, -1).T, mode='reduced')

    for _ in range(1, m):
        if Q.shape[1] == n_cols:
            break  # All dimensions are spanned

        # Compute the projection of all rows onto the subspace spanned by Q
        projection = X @ Q @ Q.T
        orthogonal_components = X - projection

        # Calculate norms for both orthogonal and projection components
        orthogonal_norms = np.linalg.norm(orthogonal_components, axis=1)
        projection_norms = np.linalg.norm(projection, axis=1)

        # Compute composite scores considering both components
        composite_scores = alpha * orthogonal_norms + (1 - alpha) * projection_norms
        composite_scores[selected_indices] = -np.inf

        # Select the row with the highest composite score
        best_index = np.argmax(composite_scores)
        selected_indices.append(best_index)

        # Update the basis Q with the new row
        row_to_add = X[best_index].reshape(-1, 1)
        Q, _ = qr(np.hstack([Q, row_to_add]), mode='reduced')
    selected_indices = torch.tensor(selected_indices)
    return selected_indices


def cov_greedy_maxgap(X, m):
    """
    Select m rows from X that maximize coverage of the principal directions of X.T @ X,
    focusing on the largest gaps in representation.

    Args:
    X: Data matrix with instances as rows.
    m: Number of rows to select.

    Returns:
    selected_indices: Indices of the selected rows.
    """
    n_rows, n_cols = X.shape
    if m > n_rows:
        raise ValueError("m cannot be greater than the number of rows in X.")

    # Compute the full covariance matrix and its eigendecomposition
    cov_matrix = X.T @ X
    eigenvalues, eigenvectors = eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    principal_directions = eigenvectors[:, idx]

    selected_indices = []
    S = np.zeros((0, n_cols))  # Initialize S as an empty matrix

    for _ in range(m):
        if S.shape[0] > 0:
            # Project S onto each principal direction and calculate the coverage
            projection_strength = np.sum((S @ principal_directions)**2, axis=0)
        else:
            projection_strength = np.zeros(principal_directions.shape[1])

        # Identify the least covered principal direction
        least_covered_index = np.argmin(projection_strength)
        direction_to_cover = principal_directions[:, least_covered_index].reshape(-1, 1)

        # Find the row that best contributes to this direction
        contributions = np.abs(X @ direction_to_cover).flatten()  # Projection magnitude onto the direction
        contributions[selected_indices] = -np.inf  # Exclude already selected rows
        best_index = np.argmax(contributions)

        # Update selected indices and S
        selected_indices.append(best_index)
        S = np.vstack([S, X[best_index]])
    selected_indices = torch.tensor(selected_indices)
    return selected_indices


def cov_greedy_maxproj(X, m):
    """
    Select m rows from X using a matrix-wise operation to maximize coverage improvement
    across all principal directions of X.T @ X.

    Args:
    X: Data matrix with instances as rows.
    m: Number of rows to select.

    Returns:
    selected_indices: Indices of the selected rows.
    """
    n_rows, n_cols = X.shape
    if m > n_rows:
        raise ValueError("m cannot be greater than the number of rows in X.")

    # Compute the full covariance matrix and its eigendecomposition
    cov_matrix = X.T @ X
    eigenvalues, eigenvectors = eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    principal_directions = eigenvectors[:, idx]

    selected_indices = []
    S = np.zeros((0, n_cols))  # Initialize S as an empty matrix

    for _ in range(m):
        if S.shape[0] > 0:
            # Compute the current coverage of each direction by S
            projection_strength_current = np.sum((S @ principal_directions)**2, axis=0)
        else:
            projection_strength_current = np.zeros(principal_directions.shape[1])

        # Compute the potential improvement for all rows if added to S
        potential_projections = X @ principal_directions  # Projection of all rows onto all principal directions
        potential_projection_strength = np.sum(potential_projections**2, axis=1)  # Sum of squares along directions

        if S.shape[0] > 0:
            # Correcting the broadcast issue:
            potential_projection_strength += projection_strength_current.sum()  # Sum all current strengths and add to each row's total

        # Exclude already selected rows by setting their scores to -inf
        potential_projection_strength[selected_indices] = -np.inf

        # Select the row with the maximum improvement
        best_index = np.argmax(potential_projection_strength)
        selected_indices.append(best_index)

        # Update S to include the newly selected row
        S = np.vstack([S, X[best_index]])
    selected_indices = torch.tensor(selected_indices)
    return selected_indices



def cov_greedy_weighted_by_gaps(X, m):
    """
    Select m rows from X by evaluating contributions weighted by the representation gaps
    in principal directions of X.T @ X, and return indices as a torch tensor. Use original principal directions
    for gap measurement consistently.

    Args:
    X: Data matrix with instances as rows.
    m: Number of rows to select.

    Returns:
    selected_indices: Indices of the selected rows as a torch.tensor.
    """
    n_rows, n_cols = X.shape
    if m > n_rows:
        raise ValueError("m cannot be greater than the number of rows in X.")

    # Compute the full covariance matrix and its eigendecomposition
    cov_matrix = X.T @ X
    eigenvalues, eigenvectors = eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    principal_directions = eigenvectors[:, idx]
    full_eigenvalues = eigenvalues[idx]

    selected_indices = []
    S = np.zeros((0, n_cols))  # Initialize S as an empty matrix

    for _ in range(m):
        if S.shape[0] > 0:
            # Compute the current covariance matrix for selected subset S
            cov_selected = S.T @ S
            # Use original eigenvectors to project the selected covariance matrix
            selected_eigenvalues = np.diag(principal_directions.T @ cov_selected @ principal_directions)
            gap = full_eigenvalues[:len(selected_eigenvalues)] - selected_eigenvalues
        else:
            gap = full_eigenvalues.copy()  # Initial gaps are just the full eigenvalues

        # Calculate potential contributions for all rows to all principal directions
        projections = X @ principal_directions
        contributions = (projections ** 2).dot(gap)  # Weight contributions by gap sizes

        # Exclude already selected rows by setting their scores to -inf
        contributions[selected_indices] = -np.inf

        # Select the row with the maximum weighted contribution
        best_index = np.argmax(contributions)
        selected_indices.append(best_index)

        # Update S to include the newly selected row
        S = np.vstack([S, X[best_index]])

    # Convert selected indices to torch.tensor
    selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.int64)

    return selected_indices_tensor