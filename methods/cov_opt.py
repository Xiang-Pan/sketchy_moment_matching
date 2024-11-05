import wandb
from omegaconf import OmegaConf
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import os
import wandb.plot
from linear_probe import test_eb
from utils.hash_utils import get_cfg_hash, get_cfg_hash_without_fraction
import logging
from logging import getLogger

from typing import Tuple, Any
logger = getLogger(__name__)
logger.setLevel(level="DEBUG")
formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')

import lightning.pytorch as pl
from linear_probe import linear_probe
from hydra_zen import instantiate
from hydra_zen import zen
from utils.data_utils import load_eb_dataset_cfg

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import Trainer

def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, 0) - z
    ind = torch.arange(n_features) + 1
    ind = ind.to(v.device)
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = torch.clamp(v - theta, min=0)
    return w


def projection_simplex_pivot(v, z=1, random_state=None):
    if random_state is not None:
        torch.manual_seed(random_state)

    v = torch.tensor(v, device='cuda', dtype=torch.float32)
    n_features = v.size(0)
    U = torch.arange(n_features, device='cuda')
    s = torch.tensor(0.0, device='cuda')
    rho = torch.tensor(0.0, device='cuda')

    while U.numel() > 0:
        G = []
        L = []
        k = U[torch.randint(0, U.numel(), (1,))].item()
        ds = v[k]
        for j in U:
            if v[j] >= v[k]:
                if j != k:
                    ds += v[j]
                    G.append(j)
            elif v[j] < v[k]:
                L.append(j)
        drho = len(G) + 1
        if s + ds - (rho + drho) * v[k] < z:
            s += ds
            rho += drho
            U = torch.tensor(L, device='cuda')
        else:
            U = torch.tensor(G, device='cuda')

    theta = (s - z) / float(rho)
    return torch.maximum(v - theta, torch.tensor(0.0, device='cuda', dtype=torch.double))




def sparse_reg(s, alpha=2):
    return 1 / (torch.norm(s, p=alpha))

def get_cov(X, center=True):
    if center:
        X = X - torch.mean(X, dim=0)
    return (X.T @ X) / X.shape[0]

def select_top_m(s: Any,
                 m: int,
                 y: torch.Tensor = None,
                 class_conditioned: bool = False
                 ) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(s, np.ndarray):
        s = torch.tensor(s)
    if isinstance(s, torch.Tensor):
        s = s.cpu().detach()
    if isinstance(y, torch.Tensor):
        y = y.cpu()
    if class_conditioned:
        assert y is not None
        # for each y, select top m
        unique_classes = torch.unique(y)
        top_m_index = []
        for c in unique_classes:
            c_m = m // len(unique_classes)
            idxes = torch.where(y == c)[0]
            s_c = s[idxes]
            _, top_m_c = torch.topk(s_c, c_m, largest=True)
            s_idxes = idxes.squeeze()[top_m_c]  # Fix: Convert top_m_c to a 1-dimensional tensor
            top_m_index.append(s_idxes)
        top_m_index = torch.cat(top_m_index)
        top_m_s = s[top_m_index]
    else:
        top_m_index = torch.argsort(s, descending=True)[:m]
        top_m_s = s[top_m_index]
    top_m_index = top_m_index.cpu().detach().numpy()
    top_m_s = top_m_s.cpu().detach().numpy()
    return top_m_index, top_m_s


# def select_top_m(
#     s: torch.Tensor,
#     m: int,
#     y: torch.Tensor = None,
#     class_conditioned: bool = False,
# ) -> (torch.Tensor, torch.Tensor):
#     if class_conditioned:
#         assert y is not None
#         # for each y, select top m
#         unique_classes = torch.unique(y)
#         top_m_index = []
#         for c in unique_classes:
#             c_m = m // len(unique_classes)
#             idxes = torch.where(y == c)[0]
#             s_c = s[idxes]
#             _, top_m_c = torch.topk(s_c, c_m, largest=True)
#             s_idxes = idxes.squeeze()[top_m_c]  # Fix: Convert top_m_c to a 1-dimensional tensor
#             top_m_index.append(s_idxes)
#         top_m_index = torch.cat(top_m_index)
#         top_m_s = s[top_m_index]
#     else:
#         top_m_index = torch.argsort(s, descending=True)[:m]
#         top_m_s = s[top_m_index]
#     top_m_index = top_m_index
#     top_m_s = top_m_s
#     return top_m_index, top_m_s


def sample(s_simplex, m):
    sampled_idx = np.random.choice(np.arange(len(s_simplex)), m, p=s_simplex)
    return sampled_idx

class ValidateOnImprovedTrainLoss(pl.Callback):
    def __init__(self, start_epoch):
        self.best_train_loss = float('inf')
        self.start_epoch = start_epoch
        self.last_val_epoch = 0

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch

        if current_epoch >= self.start_epoch:
            current_train_loss = trainer.callback_metrics["train/cov_loss_vec_norm"]

            if current_train_loss < self.best_train_loss:
                self.best_train_loss = current_train_loss
                # Trigger validation
                trainer.validate(pl_module)


class CovOptimizationModule(pl.LightningModule):
    def __init__(self,
                    n,
                    p,
                    c,
                    m,
                    eigen_cutoff,
                    sparse_scale,
                    simplex_method,
                    s_init_method,
                    gamma_init_method,
                    # CFG
                    cfg,
                    # DATA
                    X,
                    Y,
                    uniform_scale=0.0,  #! DEPRECATED
                    Cov=None,
                    ):
        super(CovOptimizationModule, self).__init__()
        """_summary_
        Args:
            n: the number of samples
            p: the number of features
            m: the intended number of selected samples, only used for test
            c: 1/c is the lower bound
            eigen_cutoff: the number or the method of eigenvectors to be used, we set the rest to -1
            uniform_scale[DEPRECATED]: the uniform regularization scale, if None, we do not use it
        """
        self.n = torch.tensor(n).long()
        self.p = p
        self.m = m
        self.c = c
        self.X = X
        self.X = X.double()
        self.Y = Y.long()

        if Cov is not None:
            logger.info("Cov is not None")
            self.Cov = Cov
        else:
            logger.info("Using X to calculate Cov")
            self.Cov = get_cov(X, center=False)
        self.Cov = self.Cov.double()
        self.U, self.Sigma, self.V = torch.svd(self.Cov)

        self.s = nn.Parameter(self._s_init(s_init_method), requires_grad=True)
        self.gamma = nn.Parameter(self._gamma_init(gamma_init_method), requires_grad=True)
        self.sparse_scale = sparse_scale
        self.uniform_scale = uniform_scale
        self.simplex_method = simplex_method

        self.cfg = cfg

        #! DEPRECATED: eigen cutoff is deprecated!
        self.eigen_cutoff = eigen_cutoff
        self._pgd_gamma()

    def _pgd_gamma(self):
        if self.eigen_cutoff == "r_n":
            self.k = self.calculate_r_n()
            logger.info(f"r_n: {self.k}")
        elif self.eigen_cutoff != -1:
            self.k = self.eigen_cutoff
            k = self.k
            top = torch.clamp(self.gamma[:k], 1/self.c, torch.inf)
            tail = torch.clamp(self.gamma[k:], 0, 0)
            self.gamma.data = torch.cat([top, tail])
            logger.info(f"eigen_cutoff: {self.k}")
        else:
            gamma_value = torch.clamp(self.gamma, 1/self.c, torch.inf)
            self.gamma.data = gamma_value

    def _s_init(self, s_init_method):
        if s_init_method == "uniform":
            s_init = torch.ones((self.n)).double().cuda()
            s_init = s_init / torch.sum(s_init)
            # assert torch.sum(s_init) == 1
            assert torch.abs(torch.sum(s_init) - 1) < 1e-3, f"sum(s): {torch.sum(s_init)}"
        elif s_init_method == "random_m":
            s_init = torch.zeros((self.n)).double().cuda()
            random_s_idx = torch.randperm(self.n)[:self.m]
            s_init[random_s_idx] = 1/self.m
            assert torch.abs(torch.sum(s_init) - 1) < 1e-3, f"sum(s): {torch.sum(s_init)}"
        elif s_init_method == "top_1_m":
            # calculate the projection of the top 1 eigenvector
            top_1_eigenvector = self.U[:, 0]
            # calculate each sample's projection
            s_init = self.X @ top_1_eigenvector
            # set top m to 1/m
            top_m = torch.topk(s_init, self.m, largest=True)
            s_init = torch.zeros_like(s_init)
            s_init[top_m.indices] = 1/self.m
            assert torch.abs(torch.sum(s_init) - 1) < 1e-3, f"sum(s): {torch.sum(s_init)}"
        else:
            raise ValueError(f"Unknown s_init_method: {s_init_method}")
        return s_init

    def _gamma_init(self, gamma_init_method):
        if gamma_init_method == "uniform":
            gamma_init = torch.ones((self.p)).double().cuda()
            gamma_init = 1/self.c * gamma_init
        elif gamma_init_method == "uniform_pertrubed":
            gamma_init = torch.ones((self.p)).double().cuda()
            gamma_init = 1/self.c * gamma_init + torch.randn_like(gamma_init) * 0.01
        else:
            raise ValueError(f"Unknown gamma_init_method: {gamma_init_method}")
        return gamma_init

    def calculate_r_n(self):
        X = self.X
        p = self.p
        n = self.n
        Cov = X.T @ X / n
        U, Sigma, V = torch.svd(Cov)
        for t in range(1, p):
            Cov_t = U[:, :t] @ torch.diag(Sigma[:t]) @ V[:, :t].T
            if torch.trace(Cov - Cov_t) <= torch.norm(Cov, p=2) / n:
                return t
        return self.p

    def get_cov_m_gap(self):
        X = self.X
        s = self.s
        U, Sigma, V = self.U, self.Sigma, self.V
        gamma = self.gamma
        Cov = self.Cov
        top_m = torch.topk(s, self.m, largest=True)
        mX = X[top_m.indices, :]
        Cov_m = get_cov(mX, center=False)
        cov_m_gap = torch.norm(Cov_m - Cov, p="fro") / torch.norm(Cov, p="fro")
        cov_m_loss_vec = (U.T @ Cov_m @ U) - gamma * Sigma
        return cov_m_gap, cov_m_loss_vec

    def get_cov_m_gap_weighted(self):
        X = self.X
        s = self.s
        U, Sigma, V = self.U, self.Sigma, self.V
        gamma = self.gamma
        Cov = self.Cov
        top_m = torch.topk(s, self.m, largest=True)
        mX = X[top_m.indices, :]
        mX = mX * s[top_m.indices].unsqueeze(1)
        Cov_m = get_cov(mX, center=False)
        cov_m_gap = torch.norm(Cov_m - Cov, p="fro") / torch.norm(Cov, p="fro")
        cov_m_loss_vec = (U.T @ Cov_m @ U) - gamma * Sigma
        return cov_m_gap, cov_m_loss_vec

    # @torch.compile
    def get_loss(self, X, s, gamma, prefix=""):
        U, Sigma, V = self.U, self.Sigma, self.V
        Cov = self.Cov
        n, p = X.shape
        assert torch.abs(torch.sum(s) - 1) < 1e-3, f"sum(s): {torch.sum(s)}"
        assert torch.all(s >= 0)
        assert torch.all(s <= 1)

        SX = s.unsqueeze(1) * X
        Cov_S = SX.T @ SX
        cov_s_gap = torch.norm(Cov_S - Cov, p="fro") / torch.norm(Cov, p="fro")
        cov_m_gap, cov_m_loss_vec = self.get_cov_m_gap()
        cov_m_gap_weighted, cov_m_loss_vec_weighted = self.get_cov_m_gap_weighted()

        cov_loss_vec = torch.diag(U.T @ Cov_S @ U) - gamma * Sigma
        cov_loss_vec_norm = torch.norm(cov_loss_vec, p=2)
        cov_m_loss = torch.norm(cov_m_loss_vec, p=2)

        sparse_loss = self.sparse_scale * sparse_reg(s, alpha=2)

        # uniform_loss = 0
        # if self.uniform_scale != 0:
        #     unique_labels = torch.unique(self.Y)
        #     for c in unique_labels:
        #         idxes = torch.nonzero(self.Y == c).squeeze()
        #         s_c = s[idxes]
        #         len_idxes = len(idxes)
        #         uniform_loss += (torch.norm(s_c, p=1).sum() - len_idxes)**2
        #     uniform_loss = self.uniform_scale * uniform_loss
        # self.log({f"{prefix}/uniform_loss": uniform_loss.item()})

        loss = cov_loss_vec_norm + sparse_loss
        self.log(f"{prefix}/epoch", self.current_epoch)
        self.log(f"{prefix}/cov_m_loss", cov_m_loss.item())
        self.log(f"{prefix}/sparse_loss", sparse_loss.item(), prog_bar=True)
        self.log(f"{prefix}/cov_loss_vec_norm", cov_loss_vec_norm.item(), prog_bar=True)
        self.log(f"{prefix}/loss", loss.item(), prog_bar=True)
        self.log(f"{prefix}/cov_s_gap", cov_s_gap.item())
        self.log(f"{prefix}/cov_m_gap", cov_m_gap.item())
        self.log(f"{prefix}/cov_m_gap_weighted", cov_m_gap_weighted.item())
        self.log(f"{prefix}/l0_ratio", torch.norm(s, p=0).item() / self.n)
        return loss

    def forward(self, X):
        if self.simplex_method == "softmax_reparam":
            s_simplex = torch.softmax(self.s, dim=0)
        else:
            s_simplex = self.s
        assert torch.abs(torch.sum(s_simplex) - 1) < 1e-3, f"sum(s): {torch.sum(s_simplex)}"
        loss = self.get_loss(X, s_simplex, self.gamma, prefix="train")
        return loss


    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():
            # keep gamma lower bound
            # self.gamma.clamp_(1/self.c, torch.inf)
            self._pgd_gamma()
            if self.simplex_method == "softmax_pgd":
                self.s.data = torch.softmax(self.s, dim=0)
            elif self.simplex_method == "sort_pgd":
                self.s.data = projection_simplex_sort(self.s.data, z=1)
            else:
                assert self.simplex_method in ["softmax_reparam"]
        if self.eigen_cutoff != -1:
            logger.info(f"gamma: {self.gamma[self.eigen_cutoff]}")
        else:
            logger.debug(f"gamma: {self.gamma}")


    def configure_optimizers(self):
        cfg = self.cfg
        logger.info(f"Adam optimizer with lr: {cfg.selection.optimizer.lr}")
        optimizer = torch.optim.Adam([self.s, self.gamma], lr=cfg.selection.optimizer.lr)
        return optimizer
        # return {"optimizer": optimizer, "lr_scheduler": None, "monitor": "train/loss"}
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=200, verbose=True)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train/loss"}
        # cosine
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        #                                                        T_max=cfg.selection.max_epochs, eta_min=1e-6)
        # optimizer = torch.optim.Adam([self.s, self.gamma], lr=cfg.selection.optimizer.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    # step_size=200,
                                                    # gamma=0.1)
        # scheduler = instantiate(cfg.selection.scheduler, optimizer=optimizer)
        # return [optimizer], [scheduler]

    def test_selection(self):
        if self.simplex_method == "softmax_reparam":
            if self.current_epoch == 0:
                s_simplex = self.s
            else:
                s_simplex = torch.softmax(self.s, dim=0)
        else:
            s_simplex = self.s
        dataset_name = self.cfg.dataset.name
        seed = self.cfg.seed
        class_unconditioned_idxes, class_unconditioned_weights = select_top_m(s_simplex, self.m, y=None, class_conditioned=False)

        test_eb(
            dataset_dict=load_eb_dataset_cfg(self.cfg),
            idxes=class_unconditioned_idxes,
            weights=None,
            seed=seed,
            use_weights=False,
            name="class_unconditioned",
            use_mlp=False,
            task="classification",
            dataset_name=dataset_name,
            test_as_val=True,
            tune=False,
            epoch=self.current_epoch
        )

        full_idx = torch.arange(self.n)
        sample_times = 3
        for i in range(sample_times):
            sampled_idx = s_simplex.multinomial(self.m, replacement=False).cpu().numpy()
            test_eb(
                dataset_dict=load_eb_dataset_cfg(self.cfg),
                idxes=sampled_idx,
                weights=None,
                seed=seed,
                use_weights=False,
                name=f"sampled_{i}",
                use_mlp=False,
                task="classification",
                dataset_name=dataset_name,
                test_as_val=True,
                tune=False,
                epoch=self.current_epoch
            )
        # class_conditioned_idxes, class_conditioned_weights = select_top_m(s_simplex, self.m, y=self.Y, class_conditioned=True)
        # test_eb(
        #     dataset_dict=load_eb_dataset_cfg(self.cfg),
        #     idxes=class_conditioned_idxes,
        #     weights=None,
        #     seed=seed,
        #     use_weights=False,
        #     name="class_conditioned",
        #     use_mlp=False,
        #     task="classification",
        #     dataset_name=dataset_name,
        #     test_as_val=True,
        #     tune=False,
        #     epoch=self.current_epoch
        # )


    def training_step(self, batch, batch_idx):
        # DO NOT USE BATCH
        X = self.X
        loss = self(X)
        return loss

    def validation_step(self, batch, batch_idx):
        #* for debugging
        if self.current_epoch == 0:
            return
        self.test_selection()

    def train_dataloader(self):
        #* dummy dataloader
        return torch.utils.data.TensorDataset(torch.zeros(1, 1))

    def val_dataloader(self):
        #* dummy dataloader
        return torch.utils.data.TensorDataset(torch.zeros(1, 1))

def prepare_train_X(cfg, dataset_dict=None):
    if dataset_dict is None:
        dataset_dict = load_eb_dataset_cfg(cfg, device="cuda")
    method = cfg.selection.method
    if method in ["cov", "cov_perclass"]:
        train_X = dataset_dict["train"]["X"]
    elif method in ["cov_ntk", "cov_ntk_perclass"]:
        from grad_utils import get_grads
        dataset_name = cfg.dataset.name
        backbone_name = cfg.backbone.name
        backbone_version = cfg.backbone.version
        use_target = cfg.selection.use_target
        layers = cfg.selection.layers
        cls_pretrain_size = cfg.selection.cls_pretrain_size if hasattr(cfg.selection, "cls_pretrain_size") else None
        feature_scale = cfg.selection.feature_scale if hasattr(cfg.selection, "feature_scale") else 1.0
        logger.critical(f"layers: {layers}")
        sketching_dim = cfg.selection.sketching_dim
        train_X = get_grads(dataset_name, backbone_name, backbone_version, sketching_dim=sketching_dim, layers=layers, use_target=use_target, feature_scale=feature_scale, cls_pretrain_size=cls_pretrain_size).to("cuda")
    else:
        raise ValueError(f"Unknown method: {method}")
    # sketching!
    if hasattr(cfg.selection, "sketching_dim"):
        sketching_dim = cfg.selection.sketching_dim
        if sketching_dim < train_X.shape[1]:
            logger.info(f"sketching_dim: {cfg.selection.sketching_dim}")
            k = cfg.selection.sketching_dim
            eb_dim = train_X.shape[1]
            S = torch.randn((eb_dim, k), device='cuda').normal_(mean=0, std=(1/k)**0.5)
            train_X = train_X @ S
        else:
            logger.info(f"sketching_dim >= train_X.shape[1], no sketching")
    return train_X


def cov_cvx_run(cfg, sparse_scale=1.0, c=1.0):
    import cvxpy as cp
    import numpy as np
    import torch
    from utils.data_utils import load_eb_dataset_cfg

    m = 1000

    eb_dataset = load_eb_dataset_cfg(cfg, device="cuda")
    X = eb_dataset["train"]["X"]
    Y = eb_dataset["train"]["Y"]
    # select top 2000
    # X = X[:2000, :]
    # Y = Y[:2000]
    Cov = X.T @ X / X.shape[0]
    U, Sigma, V = torch.svd(X.T @ X / X.shape[0])

    X_np = X.cpu().numpy()
    U_np = U.cpu().numpy()
    Sigma_np = Sigma.cpu().numpy()
    Cov_np = Cov.cpu().numpy()

    # Dimensions
    n, p = X_np.shape

    # Variables
    s = cp.Variable(n)
    gamma = cp.Variable(p)

    # Covariance matrix with selection vector s
    S_mat = cp.diag(s)
    Cov_S = X_np.T @ S_mat @ X_np

    # Objective components
    cov_loss_vec = cp.diag(U_np.T @ Cov_S @ U_np) - cp.multiply(gamma, Sigma_np)
    cov_loss_vec_norm = cp.norm(cov_loss_vec, 2)

    # Sparse regularization term
    sparse_loss = sparse_scale * cp.norm(s, 1)

    # Total loss
    loss = cov_loss_vec_norm + sparse_loss

    # Constraints
    constraints = [
        cp.sum(s) == 1,
        s >= 0,
        s <= 1,
        gamma >= 1/c,
    ]

    # Problem
    problem = cp.Problem(cp.Minimize(loss), constraints)

    # Solve the problem
    problem.solve()

    # Results
    s_opt = s.value
    gamma_opt = gamma.value

    print("Optimal s:", s_opt)
    print("Optimal gamma:", gamma_opt)

    # Select top m elements from s
    top_m_indices = np.argsort(s_opt)[-m:]
    s_top_m = np.zeros_like(s_opt)
    s_top_m[top_m_indices] = s_opt[top_m_indices]
    s_top_m /= np.sum(s_top_m)  # Normalize to sum to 1

    print("Top m s:", s_top_m)


def cov_run(cfg):
    c = cfg.selection.c
    seed = cfg.selection.seed
    max_epochs = cfg.selection.max_epochs
    dataset_dict = load_eb_dataset_cfg(cfg, device="cuda")
    train_dataset = dataset_dict["train"]
    train_X = prepare_train_X(cfg, dataset_dict)
    train_Y = dataset_dict["train"]["Y"]
    n, p = train_X.shape
    logger.info(f"n: {n}, p: {p}")

    fraction = cfg.selection.fraction
    if type(fraction) == float:
        m = int(fraction * n)
    elif type(fraction) == int:
        m = fraction
    logger.info(f"m: {m}")

    # load the deepcore
    from methods.deepcore_methods import deepcore_load
    if hasattr(cfg.selection, "preselection"):
        preselection = cfg.selection.preselection
        _, method, m = preselection.split("_")
        dataset = cfg.dataset.name
        backbone = cfg.backbone.name
        m = cfg.selection.fraction
        test_as_val = True
        layers = -1
        preselection_idxes, preselection_weights = deepcore_load(dataset=dataset, backbone=backbone, method=method, m=m, seed=cfg.seed, test_as_val=test_as_val, layers=layers)
        n = len(preselection_idxes)
        train_X = train_X[preselection_idxes, :]
        train_Y = train_Y[preselection_idxes]

    # perclass
    if cfg.selection.method in ["cov_perclass", "cov_ntk_perclass"]:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_hash = get_cfg_hash(cfg_dict)

        output_dir = f"outputs/selection/{cfg_hash}"
        os.makedirs(output_dir, exist_ok=True)
        num_classes = len(torch.unique(train_Y))
        def optimize_class(i):
            print(i)
            c_m = m // num_classes
            c_n = len(torch.where(train_Y == i)[0])
            c_train_X = train_X[torch.where(train_Y == i)[0], :]
            c_train_Y = train_Y[torch.where(train_Y == i)[0]]
            model = CovOptimizationModule(
                                            n=c_n,
                                            p=p,
                                            m=c_m,
                                            c=c, # 1/c is the lower bound
                                            uniform_scale=cfg.selection.uniform_scale if hasattr(cfg.selection, "uniform_scale") else 0.0,
                                            eigen_cutoff=cfg.selection.eigen_cutoff,
                                            sparse_scale=cfg.selection.sparse_scale,
                                            simplex_method=cfg.selection.simplex_method,
                                            s_init_method=cfg.selection.s_init_method,
                                            gamma_init_method=cfg.selection.gamma_init_method,
                                            cfg=cfg,
                                            X=c_train_X,
                                            Y=c_train_Y,
                                            Cov=None,
                                        )
            # wandb_logger = WandbLogger(name=f"{cfg_hash}-class_{i}",
            #                         project="data_pruning-selection",
            #                         entity="WANDB_ENTITY",
            #                         config=cfg_dict, tags=["0.08-debug-yijun"],
            #                         dir=f"outputs/selection/{cfg_hash}-class_{i}")
            check_val_every_n_epoch = 100000
            trainer = Trainer(max_epochs=max_epochs,
                                # logger=wandb_logger,
                                log_every_n_steps=1,
                                check_val_every_n_epoch=check_val_every_n_epoch,
                            )
            trainer.fit(model)
            s = model.s.cpu().detach()
            c_unconditioned_idxes, c_unconditioned_weights = select_top_m(s, model.m, y=None, class_conditioned=False)
            c_sampled_idxes = sample(s.cpu().detach().numpy(), m)
            c_sampled_weights = s[c_sampled_idxes].cpu().detach().numpy()

            # recover the original idxes since we are using class index here
            c_unconditioned_idxes = torch.where(train_Y == i)[0][c_unconditioned_idxes]
            c_sampled_idxes = torch.where(train_Y == i)[0][c_sampled_idxes]

            # make sure torch
            if isinstance(c_unconditioned_idxes, np.ndarray):
                c_unconditioned_idxes = torch.tensor(c_unconditioned_idxes)
            if isinstance(c_unconditioned_weights, np.ndarray):
                c_unconditioned_weights = torch.tensor(c_unconditioned_weights)
            if isinstance(c_sampled_idxes, np.ndarray):
                c_sampled_idxes = torch.tensor(c_sampled_idxes)
            if isinstance(c_sampled_weights, np.ndarray):
                c_sampled_weights = torch.tensor(c_sampled_weights)

            torch.save(model.s, f"{output_dir}/s_{i}.pt")
            torch.save(c_unconditioned_idxes, f"{output_dir}/c_unconditioned_idxes_{i}.pt")
            torch.save(c_unconditioned_weights, f"{output_dir}/c_unconditioned_weights_{i}.pt")
            torch.save(c_sampled_idxes, f"{output_dir}/c_sampled_idxes_{i}.pt")
            torch.save(c_sampled_weights, f"{output_dir}/c_sampled_weights_{i}.pt")
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1)(delayed(optimize_class)(i) for i in range(num_classes))
        # merge the results
        c_unconditioned_idxes = torch.cat([torch.load(f"{output_dir}/c_unconditioned_idxes_{i}.pt") for i in range(num_classes)])
        c_unconditioned_weights = torch.cat([torch.load(f"{output_dir}/c_unconditioned_weights_{i}.pt") for i in range(num_classes)])
        c_sampled_idxes = torch.cat([torch.load(f"{output_dir}/c_sampled_idxes_{i}.pt") for i in range(num_classes)])
        c_sampled_weights = torch.cat([torch.load(f"{output_dir}/c_sampled_weights_{i}.pt") for i in range(num_classes)]
        )
        c_conditioned_idxes = None
    else:
        m = cfg.selection.fraction

        model = CovOptimizationModule(
                                        n=n,
                                        p=p,
                                        m=m,
                                        c=c, # 1/c is the lower bound
                                        uniform_scale=cfg.selection.uniform_scale if hasattr(cfg.selection, "uniform_scale") else 0.0,
                                        eigen_cutoff=cfg.selection.eigen_cutoff,
                                        sparse_scale=cfg.selection.sparse_scale,
                                        simplex_method=cfg.selection.simplex_method,
                                        s_init_method=cfg.selection.s_init_method,
                                        gamma_init_method=cfg.selection.gamma_init_method,
                                        cfg=cfg,
                                        X=train_X,
                                        Y=train_Y,
                                        Cov=None,
                                    )
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_hash = get_cfg_hash(cfg_dict)
        output_dir = f"outputs/selection/{cfg_hash}"
        wandb_logger = WandbLogger(name=cfg_hash,
                                project="data_pruning-selection",
                                entity="WANDB_ENTITY",
                                config=cfg_dict, tags=["0.08-debug-yijun"],
                                dir=f"outputs/selection/{cfg_hash}")
        check_val_every_n_epoch = 1000
        trainer = Trainer(max_epochs=max_epochs,
                            logger=wandb_logger,
                            log_every_n_steps=1,
                            check_val_every_n_epoch=check_val_every_n_epoch,
                        )
        trainer.fit(model)

        s = model.s.cpu().detach()
        c_unconditioned_idxes, c_unconditioned_weights = select_top_m(s, model.m, y=None, class_conditioned=False)
        c_sampled_idxes = sample(s.cpu().detach().numpy(), m)
        c_sampled_weights = s[c_sampled_idxes].cpu().detach().numpy()

        #TEST: class conditioend is problematic here, it will fail if there is not enough samples in a class
        # check the number of samples in each class
        from collections import Counter
        class_count = Counter(train_Y)
        # make sure each class have m // len(unique_classes) samples
        def can_do_class_conditioned():
            if "perclass" in cfg.selection.method:
                return False
            for c in class_count:
                if class_count[c] < m // len(class_count):
                    return False
            return True

        if can_do_class_conditioned():
            c_conditioned_idxes, c_conditioned_weights = select_top_m(s, model.m, y=train_Y, class_conditioned=True)
        else:
            c_conditioned_idxes = None
            c_conditioned_weights = None

        # recover the original idxes if preselection is used
        if hasattr(cfg.selection, "preselection"):
            c_unconditioned_idxes = preselection_idxes[c_unconditioned_idxes]
            c_sampled_idxes = preselection_idxes[c_sampled_idxes]
            if c_conditioned_idxes is not None:
                c_conditioned_idxes = preselection_idxes[c_conditioned_idxes]


        cfg_hash_without_fraction = get_cfg_hash_without_fraction(cfg)
        output_dir = f"outputs/selection/{cfg_hash}"
        s_output_dir = f"outputs/selection/{cfg_hash_without_fraction}"

        os.makedirs(s_output_dir, exist_ok=True)
        torch.save(model.s, f"{s_output_dir}/s.pt")
        logger.critical(f"saved to {s_output_dir}: {s_output_dir}/s.pt")

    os.makedirs(output_dir, exist_ok=True)
    torch.save(c_unconditioned_idxes, f"{output_dir}/c_unconditioned_idxes.pt")
    torch.save(c_unconditioned_weights, f"{output_dir}/c_unconditioned_weights.pt")
    torch.save(c_sampled_idxes, f"{output_dir}/c_sampled_idxes.pt")
    torch.save(c_sampled_weights, f"{output_dir}/c_sampled_weights.pt")
    if c_conditioned_idxes is not None:
        torch.save(c_conditioned_idxes, f"{output_dir}/c_conditioned_idxes.pt")
        torch.save(c_conditioned_weights, f"{output_dir}/c_conditioned_weights.pt")
    wandb.finish()