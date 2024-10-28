import torch
def select_by_prob(prob, m, targets=None, class_conditioned=False, selected_idxes=[]):
    if class_conditioned:
        c_m = m // len(torch.unique(targets))
        assert targets is not None
        y = targets
        unique_classes = torch.unique(y)
        top_m_index = []
        for c in unique_classes:
            c_idxes = torch.where(y == c)[0]
            # prune selected_idxes
            left_idxes = torch.tensor(list(set(c_idxes) - set(selected_idxes)))
            leverage_scores_c = prob[left_idxes]
            _, top_m_index_c = torch.topk(leverage_scores_c, c_m, largest=True)
            top_m_index_c = left_idxes[top_m_index_c]
            top_m_index.append(top_m_index_c)
        top_m_index = torch.cat(top_m_index)
        return top_m_index
    else:
        left_idxes = torch.tensor(list(set(range(len(prob))) - set(selected_idxes)))
        leverage_scores = prob[left_idxes]
        _, top_m_index = torch.topk(leverage_scores, m, largest=True)
        top_m_index = top_m_index.cpu()
        top_m_index = left_idxes[top_m_index]
    return top_m_index

