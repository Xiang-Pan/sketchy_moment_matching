import torch
import numpy as np
from torchmetrics import Accuracy
from scipy.optimize import Bounds, NonlinearConstraint, minimize

def step_function(s):
    if x > 0:
        return 1
    else:
        return 0
# optimization target
# P_simga_s
class CoverageSelction():
    def __init__(self):
        self.loss = 0
        self.y_acc_metric = Accuracy(task="multiclass", num_classes=10)
        self.ce_loss_metric = torch.nn.CrossEntropyLoss()

    def get_loss(self, x, y):

        logit = x
        y_loss = self.ce_loss_metric(x, y)


def objective_function(params, X, Y, lambda_proj):
    Phi_X = X
    # diag
    theta = params[0]
    s = params[1]
    lambda_proj = 0.1
    S = torch.diag(s)
    XS = X * S
    Phi_XS = XS
    logit = X @ S @ theta
    y_loss = torch.nn.CrossEntropyLoss()(logit, Y) + lambda_proj * torch.norm(Phi_X - Phi_XS, p=2)
    # constra
    return y_loss

def binary_constraint(s):
    # This function should return 0 when all elements of s are close to 0 or 1
    return np.sum((s * (1 - s))**2)

# Set initial values for theta and s
initial_theta = np.random.rand(...)  # Replace with appropriate dimension
initial_s = np.random.rand(...)  # Replace with appropriate dimension

# Set bounds for s to be between 0 and 1
bounds = Bounds([0] * len(initial_s), [1] * len(initial_s))

# Define the binary constraint
binary_cons = NonlinearConstraint(binary_constraint, 0, 0)


result = minimize(
    objective_function,
    np.concatenate([initial_theta, initial_s]),
    bounds=bounds,
    constraints=[binary_cons],
)