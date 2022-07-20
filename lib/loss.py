"""
Loss functions

Author: Shengyu Huang
Last modified: 30.11.2020
"""
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from lib.utils import square_distance
from sklearn.metrics import precision_recall_fscore_support
# from datasets.dataloader import batch_neighbors_kpconv
# from itertools import chain
import abc
class Loss(abc.ABC):
    def __init__(self,keys: list or tuple):
        self.keys=list(keys)

    @abc.abstractmethod
    def __call__(self, data_pr, data_gt, **kwargs):
        pass
class Batch_hard_Rindex_loss(Loss):
    def __init__(self,cfg):
        super().__init__(['triplet_ranking_Rindex_loss'])
        self.R_perm=torch.from_numpy(np.load(f'{cfg.SO3_related_files}/60_60.npy').astype(np.int).reshape([-1])).cuda()
        self.class_loss=torch.nn.CrossEntropyLoss()

    def eqvloss(self,eqvfeat0,eqvfeat1):
        B,F,G=eqvfeat0.shape
        eqvfeat0=eqvfeat0[:,:,self.R_perm].reshape([B,F,G,G])
        score=torch.einsum('bfgk,bfk->bg',eqvfeat0,eqvfeat1)
        return score

class L2_loss(Loss):
    def __init__(self,cfg):
        super().__init__(['L2_Loss'])
        self.loss=torch.nn.MSELoss(reduction='sum')

    def __call__(self,patch_op,patch_gt):
        return self.loss(patch_op,patch_gt)

class MetricLoss(nn.Module):
    """
    We evaluate both contrastive loss and circle loss
    """
    def __init__(self,configs,log_scale=16, pos_optimal=0.1, neg_optimal=1.4):
        super(MetricLoss,self).__init__()
        self.log_scale = log_scale
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal

        self.pose_loss=torch.nn.MSELoss(reduction='sum')

        self.image_feature=configs.image_feature
        self.node_overlap=configs.node_overlap
        self.quaternion=configs.quaternion

        self.pos_margin = configs.pos_margin
        self.neg_margin = configs.neg_margin
        self.max_points = configs.max_points
        # self.loss=Batch_hard_Rindex_loss(cfg=)

        self.safe_radius = configs.safe_radius
        self.matchability_radius = configs.matchability_radius
        self.pos_radius = configs.pos_radius # just to take care of the numeric precision

    def get_circle_loss(self, coords_dist, feats_dist):
        """
        Modified from: https://github.com/XuyangBai/D3Feat.pytorch
        """
        pos_mask = coords_dist < self.pos_radius
        neg_mask = coords_dist > self.safe_radius

        ## get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1)>0) * (neg_mask.sum(-1)>0)).detach()
        col_sel = ((pos_mask.sum(-2)>0) * (neg_mask.sum(-2)>0)).detach()

        # get alpha for both positive and negative pairs
        pos_weight = feats_dist - 1e5 * (~pos_mask).float() # mask the non-positive
        pos_weight = (pos_weight - self.pos_optimal) # mask the uninformative positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight).detach()

        neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative
        neg_weight = (self.neg_optimal - neg_weight) # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight),neg_weight).detach()

        lse_pos_row = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-1)
        lse_pos_col = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-2)

        lse_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-1)
        lse_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-2)

        loss_row = F.softplus(lse_pos_row + lse_neg_row)/self.log_scale
        loss_col = F.softplus(lse_pos_col + lse_neg_col)/self.log_scale

        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2

        return circle_loss

    def get_recall(self,coords_dist,feats_dist):
        """
        Get feature match recall, divided by number of true inliers
        """
        pos_mask = coords_dist < self.pos_radius
        n_gt_pos = (pos_mask.sum(-1)>0).float().sum()+1e-12
        # print("feats_dist:",torch.min(feats_dist, -1))
        _, sel_idx = torch.min(feats_dist, -1)
        sel_dist = torch.gather(coords_dist,dim=-1,index=sel_idx[:,None])[pos_mask.sum(-1)>0]
        n_pred_pos = (sel_dist < self.pos_radius).float().sum()
        recall = n_pred_pos / n_gt_pos
        return recall

    def get_weighted_bce_loss(self, prediction, gt):
        loss = nn.BCELoss(reduction='none')
        class_loss = loss(prediction, gt)
        weights = torch.ones_like(gt)
        w_negative = gt.sum()/gt.size(0)
        w_positive = 1 - w_negative

        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative


        w_class_loss = torch.mean(weights * class_loss)
        #######################################
        # get classification precision and recall
        predicted_labels = prediction.detach().cpu().round().numpy()

        cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().round().numpy(),predicted_labels, average='binary')

        return w_class_loss, cls_precision, cls_recall



    def forward(self, inputs):
        """
        Circle loss for metric learning, here we feed the positive pairs only
        Input:
            src_pcd:        [N, 3]
            tgt_pcd:        [M, 3]
            rot:            [3, 3]
            trans:          [3, 1]
            src_feats:      [N, C]
            tgt_feats:      [M, C]
        """

        rot, trans = inputs['rot'], inputs['trans']
        src_feats,tgt_feats=inputs['src_feats'],inputs['tgt_feats']
        src_pcd, tgt_pcd = inputs['src_pcd_raw'], inputs['tgt_pcd_raw']
        correspondence = inputs['correspondences']
        scores_overlap,scores_saliency=inputs['scores_overlap'],inputs['scores_saliency']

        src_pcd = (torch.matmul(rot,src_pcd.transpose(0,1))+trans).transpose(0,1)
        stats=dict()
        src_idx = list(set(correspondence[:,0].int().tolist()))
        tgt_idx = list(set(correspondence[:,1].int().tolist()))

        #######################
        if self.image_feature:
            # calculate depth_loss
            pass
        if self.node_overlap:
            node_overlap_gt = inputs['node_overlap_gt']
            #
            node_overlap_score=inputs['node_overlap_score_pred']
            node_overlap_gt=inputs['node_overlap_gt']
            node_overlap_loss, node_overlap_recall, node_overlap_precision = self.get_weighted_bce_loss(node_overlap_score, node_overlap_gt)
            stats['node_overlap_loss'] = node_overlap_loss
            stats['node_overlap_recall'] = node_overlap_recall
            stats['node_overlap_precision'] = node_overlap_precision
            # stats['node_overlap_loss'] = 0
            # stats['node_overlap_recall'] = 0
            # stats['node_overlap_precision'] = 0

        if self.quaternion:
            # l2 loss
            quaternion_pred,quaternion_gt=inputs['quaternion_pred'],inputs['quaternion_gt']
            trans_pred,trans_gt=inputs['trans_pred'],inputs['trans_gt']
            # quaternion_loss=self.quaternion_loss(quaternion_pred,quaternion_gt)
            quaternion_loss=self.pose_loss(quaternion_pred,quaternion_gt)
            trans_loss=self.pose_loss(trans_pred,trans_gt)
            pose_loss=quaternion_loss+trans_loss
            stats['pose_loss']=pose_loss


            pass

        #######################
        # get BCE loss for overlap, here the ground truth label is obtained from correspondence information
        src_gt = torch.zeros(src_pcd.size(0))
        src_gt[src_idx]=1.
        tgt_gt = torch.zeros(tgt_pcd.size(0))
        tgt_gt[tgt_idx]=1.
        gt_labels = torch.cat((src_gt, tgt_gt)).to(torch.device('cuda'))

        class_loss, cls_precision, cls_recall = self.get_weighted_bce_loss(scores_overlap, gt_labels)
        stats['overlap_loss'] = class_loss
        stats['overlap_recall'] = cls_recall
        stats['overlap_precision'] = cls_precision

        #######################
        # get BCE loss for saliency part, here we only supervise points in the overlap region
        src_feats_sel, src_pcd_sel = src_feats[src_idx], src_pcd[src_idx]
        tgt_feats_sel, tgt_pcd_sel = tgt_feats[tgt_idx], tgt_pcd[tgt_idx]
        scores = torch.matmul(src_feats_sel, tgt_feats_sel.transpose(0,1))
        _, idx = scores.max(1)
        distance_1 = torch.norm(src_pcd_sel - tgt_pcd_sel[idx], p=2, dim=1)
        _, idx = scores.max(0)
        distance_2 = torch.norm(tgt_pcd_sel - src_pcd_sel[idx], p=2, dim=1)


        gt_labels = torch.cat(((distance_1<self.matchability_radius).float(), (distance_2<self.matchability_radius).float()))
        src_saliency_scores = scores_saliency[:src_pcd.size(0)][src_idx]
        tgt_saliency_scores = scores_saliency[src_pcd.size(0):][tgt_idx]
        scores_saliency = torch.cat((src_saliency_scores, tgt_saliency_scores))

        class_loss, cls_precision, cls_recall = self.get_weighted_bce_loss(scores_saliency, gt_labels)
        stats['saliency_loss'] = class_loss
        stats['saliency_recall'] = cls_recall
        stats['saliency_precision'] = cls_precision

        #######################################
        # filter some of correspondence as we are using different radius for "overlap" and "correspondence"
        c_dist = torch.norm(src_pcd[correspondence[:,0]] - tgt_pcd[correspondence[:,1]], dim = 1)
        c_select = c_dist < self.pos_radius - 0.001
        correspondence = correspondence[c_select]
        if(correspondence.size(0) > self.max_points):
            choice = np.random.permutation(correspondence.size(0))[:self.max_points]
            correspondence = correspondence[choice]
        src_idx = correspondence[:,0]
        tgt_idx = correspondence[:,1]
        src_pcd, tgt_pcd = src_pcd[src_idx], tgt_pcd[tgt_idx]
        src_feats, tgt_feats = src_feats[src_idx], tgt_feats[tgt_idx]
        #######################
        # get L2 distance between source / target point cloud
        coords_dist = torch.sqrt(square_distance(src_pcd[None,:,:], tgt_pcd[None,:,:]).squeeze(0))
        feats_dist = torch.sqrt(square_distance(src_feats[None,:,:], tgt_feats[None,:,:],normalised=True)).squeeze(0)

        ##############################
        # get FMR and circle loss
        ##############################
        recall = self.get_recall(coords_dist, feats_dist)
        circle_loss = self.get_circle_loss(coords_dist, feats_dist)

        stats['circle_loss']= circle_loss
        stats['recall']=recall

        return stats



