import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusterLookup(nn.Module):
    def __init__(self, dim: int, n_classes: int):
        super(ClusterLookup, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        # the clusters will be the same endmembers!
        

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))
            
    def forward(self, x, alpha, log_probs=False, clusters=None):
        # x shape: [P, C] where P is number of pixels
        clusters = self.clusters if clusters is None else clusters
        normed_clusters = F.normalize(clusters, dim=1)  # [N, C]
        normed_features = F.normalize(x, dim=1)  # [P, C]
        
        # Computing inner products with matrix multiplication
        # [P, C] × [C, N] -> [P, N]
        inner_products = torch.matmul(normed_features, normed_clusters.t())
        
        if alpha is None:
            cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), clusters.shape[0]) \
                .to(torch.float32)  # [P, N]
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)  # [P, N]
            
        #cluster_loss = -(cluster_probs * inner_products).sum(1).mean()
        
        if log_probs:
            return nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return inner_products, cluster_probs