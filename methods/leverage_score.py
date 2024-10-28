import numpy as np
from data_utils import load_eb_dataset
from sampling_utils import select_by_prob
from sklearn.decomposition import randomized_svd
import logging
import copy
from tqdm import tqdm
import wandb
from opt import test_selection
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def compute_leverage(matrixA, low_rank=False, n_components=20,
                     n_iter=5, use_residual=False):
    '''
    Computes leverage scores of the input matrix A with two possible options
    1) Exact Leverage Scores
    2) Low-rank leverage scores

    In each case the leverage scores are computed using the SVD. Here we
    compute the row leverage scores for the imput matrix A and a leverage score
    vector corresponding to each row is returned.

    In either case, the SVD is computed as: U, S, V = svd(matrixA)

    Parameters
    ----------
    matrixA: 2D array

    low-rank: bool (optional)
    default: False

    n_components: int (optional)
    default: 20

    n_iter: int (optional)
    default: 5

    Returns
    -------
    lev_vec: 1D array
    '''
    # Transpose is taken for computing row leverage scores
    matrixA = matrixA.numpy() if hasattr(matrixA, 'numpy') else matrixA
    _ , _, v_mat = np.linalg.svd(matrixA.T, full_matrices=False)

    # faster approximation of the SVD using randomized SVD from sklearn.
    if low_rank:
        _, _, v_mat = randomized_svd(matrixA.T,
                                     n_components=n_components,
                                     n_iter=n_iter,
                                     random_state=None)

    # gets the row-norms
    if use_residual:
        lev_vec = np.sum(v_mat, axis=0)
    else:
        lev_vec = np.sum(v_mat ** 2, axis=0)
    return lev_vec


import torch
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

class LeverageScoreFeature(object):
    def __init__(self,
                 use_residual,
                 use_random,
                 dataset_name,
                 num_classes,
                 use_raw_G,
                 soft_orth,
                 wandb_dir,
                 region_selection=None,
                 ):
        self.use_residual = use_residual
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.wandb_dir = wandb_dir
        self.use_raw_G = use_raw_G
        self.soft_orth = soft_orth
        self.use_random = use_random
        self.region_selection = region_selection
        os.makedirs(self.wandb_dir, exist_ok=True)
        
    def setup(self, dataset_dict, residual):
        self.load_K(dataset_dict)
        self.residual = residual
        logger.debug(f"residual: {self.residual}")
        if self.residual is not None:
            self.residual = torch.tensor(self.residual).cuda()
        self.leverage_score = self.calculate_leverage_score()

    def load_K(self, dataset_dict):
        self.X = dataset_dict["train"]["X"]
        self.Y = dataset_dict["train"]["Y"]
        self.K = self.X @ self.X.T
        self.G = self.X

    def calculate_leverage_score(self, 
                                 G=None, 
                                 use_residual=None,
                                 residual=None, 
                                 use_raw_G=None) -> torch.Tensor:
        #NOTE: it returns the combined leverage score
        if use_residual is None:
            use_residual = self.use_residual
        if G is None:
            G = self.G
        if use_raw_G is None:
            use_raw_G = self.use_raw_G
        if residual is None:
            residual = self.residual

        logger.debug(f"use_residual: {use_residual}, use_raw_G: {use_raw_G}")

        if use_raw_G:
            ls = G
        else:
            Q, R =  torch.linalg.qr(G)
            ls = Q
        if use_residual == "multiply":
            assert residual is not None
            leverage_scores = torch.norm(ls, dim=1, p=2)
            leverage_scores = leverage_scores.cuda()
            logger.debug(f"leverage_scores: {leverage_scores.shape}")
            logger.debug(residual)
            logger.debug(f"residual: {residual.shape}")
            leverage_scores = residual * leverage_scores
        elif use_residual == "gradient_dot":
            # make sure the residual is a vector
            pass
        elif use_residual == "add":
            assert residual is not None
            leverage_scores = torch.norm(ls, dim=1, p=2) ** 2
            leverage_scores = leverage_scores.cuda()
            # make sure they are on the same scale
            leverage_scores = residual + leverage_scores
        elif use_residual == True:
            raise NotImplementedError
        else:
            assert use_residual == False
            leverage_scores = torch.norm(ls, dim=1, p=2)**2
        return leverage_scores

    def plot(self):
        leverage_score = self.calculate_leverage_score(use_raw_G=False).cpu().numpy()
        Y = self.Y.cpu().numpy()
        base_size = 5
        if self.num_classes == 2:
            fig, axes = plt.subplots(1, 2, figsize=(base_size * 2, base_size))
            for i in range(self.num_classes):
                ax = axes[i]
                ax.hist(leverage_score[Y == i])
                ax.set_title(f"class {i}")
            plt.title(f"{self.dataset_name}")
            plt.savefig(f"{self.wandb_dir}/leverage_score_class.pdf")
        else:
            fig, axes = plt.subplots(self.num_classes // 2, 2, figsize=(base_size * 2, base_size * (self.num_classes // 2)))
            for i in range(self.num_classes):
                ax = axes[i // 2, i % 2]
                ax.hist(leverage_score[Y == i])
                ax.set_title(f"class {i}")
            # set figure title
            plt.suptitle(f"{self.dataset_name}")
            plt.savefig(f"{self.wandb_dir}/leverage_score_class.pdf")
        logger.critical(f"{self.wandb_dir}/leverage_score_class.pdf")

        avg_leverage_score = defaultdict(list)
        for i in range(self.num_classes):
            avg_leverage_score[i] = np.mean(leverage_score[Y == i])
        plt.figure()
        plt.bar(avg_leverage_score.keys(), avg_leverage_score.values(), label="class-wise average leverage score")
        plt.suptitle(f"{self.dataset_name}")
        plt.savefig(f"{self.wandb_dir}/leverage_score_avg.pdf")
        logger.critical(f"{self.wandb_dir}/leverage_score_avg.pdf")

    def get_top_m_idxes(self, G, m, targets=None, class_conditioned=False, residual=None, selected_idxes=[], use_residual=None, use_raw_G=None):
        if use_residual is None:
            use_residual = self.use_residual
        if use_raw_G is None:
            use_raw_G = self.use_raw_G
        G = G.cpu()
        if targets is not None:
            targets = targets.cpu()
        raw_scores = self.calculate_leverage_score(G, residual=residual, use_residual=use_residual)
        prob = raw_scores / torch.sum(raw_scores)
        top_m_index = select_by_prob(prob, m, targets=targets, class_conditioned=class_conditioned, selected_idxes=selected_idxes)
        return top_m_index

    def adaptive_update(self, G, g_i, soft_orth=None):
        if soft_orth is None:
            soft_orth = self.soft_orth
        second_term = ((G @ g_i.T) @ g_i) / torch.norm(g_i, p=2)**2
        second_term = (1 - soft_orth) * second_term
        new_G = G - second_term
        return new_G

    def adaptive_distance_penalty(self, sampled_idx, G):
        if sampled_idx == 0:
            penalty = 0
        return G
    
from pytorch_lightning import LightningModule
class LeverageScoreNTK(LightningModule):
    def __init__(self, cfg, task):
        super().__init__()
        self.cfg = cfg
        self.classifier_name = cfg.backbone.classifier
        if cfg.dataset.name == "utk":
            task_type = "regression"
            self.net = ResNet(backbone_name=cfg.backbone.name,
                            version=cfg.backbone.version,
                            num_classes=1,
                            classifier_name=cfg.backbone.classifier)
        else:
            task_type = "classification"
            self.net = ResNet(backbone_name=cfg.backbone.name,
                            version=cfg.backbone.version,
                            num_classes=cfg.dataset.num_classes,
                            classifier_name=cfg.backbone.classifier)
        # check fc
        params = {k: v for k, v in self.net.named_parameters() if "fc" in k}
        buffers = {k: v for k, v in self.net.named_buffers() if "fc" in k}
        logger.info(params.keys())
        logger.info(buffers.keys())

        self.output_dir = get_grad_dir(cfg, phase="train")
        self.task = task
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_finetuned(self):
        cfg = self.cfg
        last_ckpt_path = get_last_ckpt_path(cfg)
        state_dict = torch.load(last_ckpt_path)["state_dict"]
        state_dict = migrate_state_dict(state_dict, "net.net", "net")
        state_dict = migrate_state_dict(state_dict, "net.feature_encoder", None)
        state_dict = migrate_state_dict(state_dict, "net.fc.0", "net.fc.fc0")
        state_dict = migrate_state_dict(state_dict, "net.fc.1", "net.fc.fc1")
        self.net.load_state_dict(state_dict)
        self.net = self.net.cuda()
    
    def get_vectorized_params(self):
        return torch.cat([p.view(-1) for p in self.parameters()])
    
    def get_vectorized_grads(self):
        return torch.cat([p.grad.view(-1) for p in self.parameters()])
    
    def get_classifier_vectorized_params(self):
        return torch.cat([p.view(-1) for p in self.net.classifier.parameters()])
    
    def get_classifier_vectorized_grads(self):
        return torch.cat([p.grad.view(-1) for p in self.net.classifier.parameters()])
        
    def set_classifier_from_sklearn(self, model):
        if type(model) == Pipeline:
            model = model.named_steps["clf"]
        assert type(model) in [LogisticRegression, MLPClassifier, Ridge]
        if type(model) == LogisticRegression:
            weight = model.coef_
            bias = model.intercept_
            linear_model = nn.Linear(weight.shape[1], weight.shape[0]).cuda()
            linear_model.weight.data = torch.from_numpy(weight).float().cuda()
            linear_model.bias.data = torch.from_numpy(bias).float().cuda()
            self.net.net.fc = linear_model
            
            logger.debug(f"net.fc: {self.net.net.fc}")
        elif type(model) == MLPClassifier:
            weight = model.coefs_.T
            bias = model.intercepts_
            linear_model = nn.Linear(weight[0].shape[1], weight[0].shape[0]).cuda()
            linear_model.weight.data = torch.from_numpy(weight[0]).float().cuda()
            linear_model.bias.data = torch.from_numpy(bias[0]).float().cuda()
            self.net.net.fc = linear_model
            logger.debug(f"net.fc: {self.net.net.fc}")
    
    def forward(self, x):
        return self.net(x)
    
    #NOTE: everything need to be the nn.Module
    def calculate_per_sample_grads(self, batch, classifier_only=False, batch_idx=0, task="classification"):
        data, targets = batch
        import torch.func as F
        model = self.net
        model.zero_grad()
        # get current device
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        if classifier_only:
            if self.classifier_name == "linear":
                params = {k: v for k, v in model.named_parameters() if "fc" in k}
                buffers = {k: v for k, v in model.named_buffers() if "fc" in k}
            elif self.classifier_name == "mlp_50":
                params = {k: v for k, v in model.named_parameters() if "fc1" in k}
                buffers = {k: v for k, v in model.named_buffers() if "fc1" in k}
            else:
                raise NotImplementedError
        else:
            params = {k: v for k, v in model.named_parameters()}
            buffers = {k: v for k, v in model.named_buffers()}
        logger.info(params.keys())
        logger.info(buffers.keys())
        if task == "classification":
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.MSELoss()
        def compute_loss(params, buffers, sample, target):
            batch = sample.unsqueeze(0)
            targets = target.unsqueeze(0)
            predictions = F.functional_call(model, (params, buffers), (batch,))
            loss = loss_fn(predictions, targets)
            return loss
        ft_compute_grad = F.grad(compute_loss)
        ft_compute_sample_grad = F.vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
        ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, targets)
        per_sample_grads = [v.view((data.shape[0], -1)) for k, v in ft_per_sample_grads.items() if v is not None]
        per_sample_grads = torch.cat(per_sample_grads, dim=1)
        device_id = torch.cuda.current_device()
        # logger.debug(f"current device: {device_id}")
        # logger.debug(torch.cuda.list_gpu_processes(device_id))
        torch.save(per_sample_grads, f"{self.output_dir}/batch-idx={batch_idx}_device-id={device_id}.pt")
        return per_sample_grads
    
    def validation_step(self, batch, batch_idx):
        self.calculate_per_sample_grads(batch, classifier_only=False, batch_idx=batch_idx)
    
    def construct_G(self, dataset, k, classifier_only=False):
        net = self.net.net
        fc = self.net.net.fc
        if type(fc) == nn.Linear:
            cls_eb_dim = fc.weight.shape[1]
            output_dim = fc.weight.shape[0]
        else:
            cls_eb_dim = fc.fc1.weight.shape[1]
            output_dim = fc.fc1.weight.shape[0]
        logger.info("fc: %s", fc)
        logger.info("self.net", net)
        logger.info("classifier embedding dimension: %d", cls_eb_dim)
        logger.info("classifier output dimension: %d", output_dim)
        if k >= cls_eb_dim and classifier_only:
            logger.info("k is larger than the classifier embedding dimension, no sketching is needed")
        G_list = []
        if self.cfg.backbone.name == "resnet18":
            if classifier_only:
                batch_size = 200
            else:
                batch_size = 100
        else:
            batch_size = 32
        batch_size = int(batch_size)
        logger.info("batch_size: %d", batch_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        for batch_idx, (data, targets) in enumerate(tqdm(dataloader, desc="construct_G")):
            if batch_idx == 0:
                #newline
                logger.debug("\n")
                logger.debug("data: %s, targets: %s", data.shape, targets.shape)
            data, targets = data.cuda(), targets.cuda()
            batch = (data, targets)
            per_sample_grads = self.calculate_per_sample_grads(batch, classifier_only=classifier_only, task=self.task)
            per_sample_grads = per_sample_grads.cpu().detach()
            torch.cuda.empty_cache()
            G_list.append(per_sample_grads)
            if batch_idx == 0:
                logger.debug("per_sample_grads: %s", per_sample_grads.shape)
        
        
        # for batch_idx, (data, targets) in enumerate(tqdm(dataloader, desc="construct_G")):
            # calculate_per_sample_grads(data, targets)
            
        # from gpuparallel import GPUParallel, delayed
        # model = self.net
        # def init(device_id=None, **kwargs):
        #     global model
        #     model = copy.deepcopy(self.net).to(device_id)
        
        # gp = GPUParallel(n_gpu=2, init_fn=init)
        # gp(delayed(calculate_per_sample_grads)(data, targets, batch_idx=batch_idx) for batch_idx, (data, targets) in enumerate(dataloader))
        
            # if k >= cls_eb_dim and classifier_only:
            #     # no sketching
            #     per_sample_grads = self.get_per_sample_gradients(data, targets, classifier_only=classifier_only)
            #     per_sample_grads = per_sample_grads.cpu().detach()
            #     torch.cuda.empty_cache()
            #     del per_sample_grads
            #     torch.save(per_sample_grads, f"{self.output_dir}/per_sample_grads_batch_idx={batch_idx}.pt")
            #     G_list.append(per_sample_grads)
            # else:
            #     logger.debug("sketching")
            #     per_sample_grads = self.get_per_sample_gradients(data, targets, classifier_only=classifier_only)
            #     # per_sample_grads = per_sample_grads
            #     torch.save(per_sample_grads, f"{self.output_dir}/per_sample_grads_batch_idx={batch_idx}.pt")
            #     # S = torch.randn((per_sample_grads.shape[1], k)).normal_(mean=0, std=(1/k)**0.5).to("cuda:0")
            #     # sketched_per_sample_grads = (per_sample_grads @ S).cpu().detach()
            #     # torch.cuda.empty_cache()
            #     # torch.save(sketched_per_sample_grads, f"{self.output_dir}/sketched_per_sample_grads_batch_idx={batch_idx}.pt")
            #     # G_list.append(sketched_per_sample_grads)
        # # clear the memory
        # G_list = [torch.load(f"{self.output_dir}/sketched_per_sample_grads_batch_idx={i}.pt") for i in range(len(G_list))]
        G = torch.cat(G_list, dim=0)
        # save_obj(G, f"{self.output_dir}/G.pt")
        torch.save(G, f"{self.output_dir}/G.pt")
        return G

    def check_G(self, G, dataset):
        # if type(self.net.fc) == nn.Linear:
        #     cls_eb_dim = self.net.net.fc.weight.shape[1]
        # else:
        #     cls_eb_dim = self.net.net.fc.fc1.weight.shape[1]
        G = torch.load(f"{self.output_dir}/G.pt")
        if G.shape[0] != len(dataset):
            logger.info("G shape: %s, dataset length: %s", G.shape[0], len(dataset))
            return False
        return True
    
    def get_G(self, dataset, classifier_only=True, k=128):
        n = len(dataset)
        logger.info(f"{self.output_dir}/G.pt")
        if os.path.exists(f"{self.output_dir}/G.pt"):
            G = torch.load(f"{self.output_dir}/G.pt")
            if self.check_G(G, dataset):
                return G
        tilde_G = self.construct_G(dataset, classifier_only=classifier_only, k=k)
        return tilde_G


    def get_top_m_idxes(self, G,
                        m,
                        targets=None,
                        use_residual=False,
                        class_conditioned=False,
                        residual=None,
                        use_raw_G=False,
                        anchor_idxes=[],
                        selected_idxes=[]):
        G = G.cpu()
        if targets is not None:
            targets = targets.cpu()
        if use_raw_G:
            Q = G
        else:
            Q, R = torch.linalg.qr(G)
        if use_residual:
            leverage_scores = torch.norm(G, dim=1, p=2)
            unnorm_prob = residual * leverage_scores 
            prob = unnorm_prob / torch.sum(unnorm_prob)
            logger.debug("prob: %s", prob.shape)
        else:
            leverage_scores = torch.norm(G, dim=1, p=2)**2
            prob = leverage_scores / torch.sum(leverage_scores)
            logger.debug("prob: %s", prob.shape)

        top_m_index = select_by_prob(prob, m, targets=targets, class_conditioned=class_conditioned, selected_idxes=selected_idxes, anchor_idxes=anchor_idxes)
        return top_m_index
        
    def adaptive_update(self, G, g_i):
        """_summary_:
            use previously sampled g_i to update G
        Args:
            G: leverage score matrix
            g: B x p, the sampled leverage score matrix
        """        
        # \begin{align}
        # q \in G-\frac{\left(G_g g_i\right) g_i^{\top}}{\left\|g_i\right\|_\nu^2}
        # \end{align}
        # NxP @ PxB @ BxP
        logger.debug("G shape: %s, g_i shape: %s", G.shape, g_i.shape)
        new_G = G - ((G @ g_i.T) @ g_i) / torch.norm(g_i, p=2)**2
        return new_G


def class_unconditioned_sampling(leverage_score_model, 
                                 sample_wise_loss, 
                                 eb_dataset_dict,
                                 output_dir, 
                                 cfg,
                                 task, 
                                 unpretrained_eb_dataset_dict,
                                 preselected_idxes,
                                 region_cfg=None):
    G = copy.deepcopy(leverage_score_model.G)
    
    m = cfg.selection.fraction * len(eb_dataset_dict["train"]["X"])
    logger.info("m: %d", m)
    logger.info("cfg.selection.B: %s", cfg.selection.B)
    if type(cfg.selection.B) == float:
        B = int(m * cfg.selection.B)
        B = max(B, 1)
    else:
        B = cfg.selection.B
    
    iter_num = int(np.ceil(m / B))
    logger.info("iter_num: %d", iter_num)

    adaptive_G = copy.deepcopy(G)

    class_unconditioned_idxes = []
    for i in tqdm(range(iter_num), desc="class_unconditioned"):
        batch_size = min(m, B)
        batch_size = int(batch_size)
        idx_i = leverage_score_model.get_top_m_idxes(G=adaptive_G, m=batch_size, residual=sample_wise_loss, class_conditioned=False, selected_idxes=class_unconditioned_idxes)
        g_i = G[idx_i]
        if i == 0:
            logger.debug(f"G.shape: {G.shape}, g_i.shape: {g_i.shape}")
        adaptive_G = leverage_score_model.adaptive_update(adaptive_G, g_i)
        class_unconditioned_idxes.append(idx_i)
    class_unconditioned_idxes = np.concatenate(class_unconditioned_idxes)
    # preselected_idxes
    class_unconditioned_idxes = preselected_idxes[class_unconditioned_idxes]
    sel_y = torch.tensor([eb_dataset_dict["train"]["Y"][i] for i in class_unconditioned_idxes])
    sel_classes, sel_classes_count = torch.unique(sel_y, return_counts=True)
    wandb.log({"class_unconditioned/sel_classes_distribution": {k.item(): v.item() for k, v in zip(sel_classes, sel_classes_count)}})
    if len(sel_classes) > 1:
        test_selection(cfg, class_unconditioned_idxes, eb_dataset_dict, unpretrained_eb_dataset_dict, name="class_unconditioned", task=task)
    else:
        logger.info("class_unconditioned: only one class")
    torch.save(class_unconditioned_idxes, f"{output_dir}/class_unconditioned_idx.pth")


def oversample_class_conditioned(leverage_score_model, sample_wise_loss, eb_dataset_dict, output_dir, cfg, unpretrained_eb_dataset_dict):
    G = copy.deepcopy(leverage_score_model.G)
    m = cfg.selection.fraction * len(eb_dataset_dict["train"]["X"])
    if type(cfg.selection.B) == float:
        B = int(m * cfg.selection.B)
        B = max(B, 1)
    else:
        B = cfg.selection.B
    if cfg.selection.fraction > 0.1:
        return
    scaled_m = int(m * 4)
    logger.info("scaled_m: %d", scaled_m)
    adaptive_G = copy.deepcopy(G)
    class_unconditioned_idxes = []
    iter_num = int(np.ceil(scaled_m / B))
    for i in tqdm(range(iter_num), desc="[oversample_class_conditioned]:[oversampling]"):
        use_raw_G = cfg.selection.use_raw_G
        batch_size = min(scaled_m, B)
        idx_i = leverage_score_model.get_top_m_idxes(G=adaptive_G, m=batch_size, residual=sample_wise_loss, class_conditioned=False, selected_idxes=class_unconditioned_idxes)
        g_i = G[idx_i]
        adaptive_G = leverage_score_model.adaptive_update(adaptive_G, g_i)
        class_unconditioned_idxes.append(idx_i)
    # index of full dataset
    class_unconditioned_idxes = np.concatenate(class_unconditioned_idxes)
    sel_y = torch.tensor([eb_dataset_dict["train"]["Y"][i] for i in class_unconditioned_idxes])
    # log sel_y distribution
    sel_classes, sel_classes_count = torch.unique(sel_y, return_counts=True)
    logger.debug("sel_classes: %s", sel_classes)
    logger.debug("sel_classes_count: %s", sel_classes_count)
    num_classes = cfg.dataset.num_classes
    # subsample to m to make sure the distribution is balanced
    each_class_m = m // num_classes
    each_class_m = int(each_class_m)
    subsampled_idxes = []
    for c in tqdm(range(num_classes), desc="[oversample_class_conditioned]:[subsample]"):
        idxes_c = torch.where(sel_y == c)[0]
        idxes_c = idxes_c[torch.randperm(len(idxes_c))[:each_class_m]]
        subsampled_idxes.append(class_unconditioned_idxes[idxes_c])
        logger.info(f"\n class: {c}, count: {len(idxes_c)}")

    subsampled_idxes = np.concatenate(subsampled_idxes)
    subsampled_idxes = torch.from_numpy(subsampled_idxes).long()
    sel_y = torch.tensor([eb_dataset_dict["train"]["Y"][i] for i in subsampled_idxes])
    sel_classes, sel_classes_count = torch.unique(sel_y, return_counts=True)
    wandb.log({"class_unconditioned/sel_classes_distribution": {k.item(): v.item() for k, v in zip(sel_classes, sel_classes_count)}})
    if len(sel_classes) > 1:
        test_selection(cfg, subsampled_idxes, eb_dataset_dict, unpretrained_eb_dataset_dict, name="oversample_class_conditioned", task="classification")
    else:
        logger.info("oversample_class_conditioned: only one class")
    # save the selection
    os.makedirs(output_dir, exist_ok=True)
    torch.save(class_unconditioned_idxes, f"{output_dir}/oversample_class_conditioned_idx.pth")

def class_conditioned_sampling(leverage_score_model, sample_wise_loss, eb_dataset_dict, output_dir, cfg, unpretrained_eb_dataset_dict):
    G = copy.deepcopy(leverage_score_model.G)
    m = cfg.selection.fraction * len(eb_dataset_dict["train"]["X"])
    if type(cfg.selection.B) == float:
        B = int(m * cfg.selection.B)
        B = max(B, 1)
    else:
        B = cfg.selection.B
    iter_num = int(np.ceil(m / B))
    targets = eb_dataset_dict["train"]["Y"]
    class_conditioned_idxes = []
    adaptive_G = G
    for i in tqdm(range(iter_num), desc="class_condition"):
        use_raw_G = cfg.selection.use_raw_G if i > 1 else False
        idx_i = leverage_score_model.get_top_m_idxes(G=adaptive_G,
                                                    m=B,
                                                    residual=sample_wise_loss,
                                                    use_residual=cfg.selection.use_residual, 
                                                    class_conditioned=True, 
                                                    targets=targets,
                                                    use_raw_G=use_raw_G)
        g_i = G[idx_i]
        adaptive_G = leverage_score_model.adaptive_update(adaptive_G, g_i)
        class_conditioned_idxes.append(idx_i)
    class_conditioned_idxes = np.concatenate(class_conditioned_idxes)
    torch.save(class_conditioned_idxes, f"{output_dir}/class_conditioned_idx.pth")
    test_selection(cfg, class_conditioned_idxes, eb_dataset_dict, unpretrained_eb_dataset_dict, name="class_conditioned", task="classification")





# from omegaconf import MISSING
# from dataclasses import dataclass

# @dataclass
# class LeverageScoreConfig:
#     use_residual: bool = MISSING
#     dataset_name: str = MISSING
#     num_classes: int = MISSING
#     wandb_dir: str = MISSING

# from hydra.core.config_store import ConfigStore
# cs = ConfigStore.instance()
# cs.store(name="debug", node=LeverageScoreConfig)



# # make config
# if __name__ == '__main__':
#     from hydra_zen import make_config
#     config = OmegaConf.create({
#         'use_residual': False,
#         'dataset_name': 'mnist',
#         'num_classes': 10,
#         'wandb_dir': 'wandb_dir'
#     })
#     ls = builds(LeverageScoreFeature, populate_full_signature=True)
#     print_yaml(ls)
#     # save to 
    
    
    
    
    # from hydra_zen import to_yaml
    # def print_yaml(x): 
    #     print(to_yaml(x))
    # from hydra_zen import just, store, builds, kwargs_of
    # import hydra_zen
    # from pprint import pprint as pp
    # from hydra_zen import make_config
    # ZenBuilds_DNN = builds(LeverageScoreFeature, populate_full_signature=True)
    # print_yaml(ZenBuilds_DNN)