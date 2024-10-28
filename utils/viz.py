import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from logging import getLogger
import wandb
logger = getLogger(__name__)
# logger.setLevel(logging.DEBUG)

def plot_tsne(X, Y, selected_idxes, dataset_name, wandb_dir, seed, epoch):
    logger.debug(f"X.shape: {X.shape}")
    train_X = X
    train_Y = Y
    # make the numpy
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=seed, verbose=1)
    if os.path.exists(f"cache/{dataset_name}/tsne_train_X.pt"):
        X_2d = torch.load(f"cache/{dataset_name}/tsne_train_X.pt")
    else:
        X_2d = tsne.fit_transform(X)
        os.makedirs(f"cache/{dataset_name}", exist_ok=True)
        torch.save(X_2d, f"cache/{dataset_name}/tsne_train_X.pt")
    # plot the tsne
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    
    clf = LogisticRegression(random_state=0).fit(X_2d, train_Y)
    # plot the decision boundary
    h = .02
    x_min, x_max = X_2d[:, 0].min() - .5, X_2d[:, 0].max() + .5
    y_min, y_max = X_2d[:, 1].min() - .5, X_2d[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=train_Y, cmap='viridis')
    plt.scatter(X_2d[selected_idxes, 0], X_2d[selected_idxes, 1], c='red')
    plt.legend(["train", "selected"])
    plt.savefig(f"{wandb_dir}/tsne_decision_boundary.pdf")
    wandb.log({"tsne_decision_boundary": wandb.Image(plt), "epoch": epoch})
    plt.close()

