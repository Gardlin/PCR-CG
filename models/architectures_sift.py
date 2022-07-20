import torch.nn.functional as F
import numpy as np
from models.gcn import GCN
from lib.utils import square_distance
from models.r_eval import quaternion_from_matrix, matrix_from_quaternion, compute_R_diff
from lib.utils import square_distance
from models import build_backbone

from models.blocks import *


class KPFCNN(nn.Module):

    def __init__(self, config):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############
        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_feats_dim
        out_dim = config.first_feats_dim
        self.image_feature = config.image_feature
        self.img_num= config.img_num
        self.init_mode=config.init_mode
        self.node_overlap = config.node_overlap
        self.quaternion = config.quaternion

        self.K = config.num_kernel_points
        self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))
        self.final_feats_dim = config.final_feats_dim

        #####################
        # List Encoder blocks
        #####################
        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # bottleneck layer and GNN part
        #####################
        gnn_feats_dim = config.gnn_feats_dim
        self.bottle = nn.Conv1d(in_dim, gnn_feats_dim, kernel_size=1, bias=True)
        k = config.dgcnn_k
        num_head = config.num_head
        self.gnn = GCN(num_head, gnn_feats_dim, k, config.nets)
        self.proj_gnn = nn.Conv1d(gnn_feats_dim, gnn_feats_dim, kernel_size=1, bias=True)
        self.proj_score = nn.Conv1d(gnn_feats_dim, 1, kernel_size=1, bias=True)

        #####################
        # List Decoder blocks
        #####################
        out_dim = gnn_feats_dim + 2

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2
            # pass

        if self.node_overlap:
            self.node_overlap_predict = nn.Conv1d(gnn_feats_dim, 1, kernel_size=1, bias=True)

        if self.quaternion:
            self.folding1 = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU())
            self.linear1 = nn.Linear(1024, 4)
            self.linear2 = nn.Linear(1024, 3)
        return

    def regular_score(self, score):
        score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
        score = torch.where(torch.isinf(score), torch.zeros_like(score), score)
        return score

    def forward(self, batch,backbone2d=None):
        # Get input features
        x = batch['features'].clone().detach()
        src_pcd_raw = batch['src_pcd_raw']
        tgt_pcd_raw = batch['tgt_pcd_raw']

        len_src_c = batch['stack_lengths'][-1][0]
        len_src_f = batch['stack_lengths'][0][0]
        pcd_c = batch['points'][-1]
        pcd_f = batch['points'][0]
        src_pcd_c, tgt_pcd_c = pcd_c[:len_src_c], pcd_c[len_src_c:]

        ### image
        res = {}
        if self.image_feature:
            if self.img_num==3:
                src_color1 = batch['src_color1']
                src_color2 = batch['src_color2']
                src_color3 = batch['src_color3']
                tgt_color1 = batch['tgt_color1']
                tgt_color2 = batch['tgt_color2']
                tgt_color3 = batch['tgt_color3']
                id_name = batch['id_name']
                src1_inds2d = batch['src1_inds2d']
                src2_inds2d = batch['src2_inds2d']
                src3_inds2d = batch['src3_inds2d']
                tgt1_inds2d = batch['tgt1_inds2d']
                tgt2_inds2d = batch['tgt2_inds2d']
                tgt3_inds2d = batch['tgt3_inds2d']
                src1_inds3d = batch['src1_inds3d']
                src2_inds3d = batch['src2_inds3d']
                src3_inds3d = batch['src3_inds3d']
                tgt1_inds3d = batch['tgt1_inds3d']
                tgt2_inds3d = batch['tgt2_inds3d']
                tgt3_inds3d = batch['tgt3_inds3d']

                src1_feature2d = backbone2d(src_color1.unsqueeze(0).cuda()).squeeze(0)  # 1 128 120 160 3d->2d
                src2_feature2d = backbone2d(src_color2.unsqueeze(0).cuda()).squeeze(0)  # 1 128 120 160 3d->2d
                src3_feature2d = backbone2d(src_color3.unsqueeze(0).cuda()).squeeze(0)  # 1 128 120 160 3d->2d
                tgt1_feature2d = backbone2d(tgt_color1.unsqueeze(0).cuda()).squeeze(0)  # 1 128 120 160 3d->2d
                tgt2_feature2d = backbone2d(tgt_color2.unsqueeze(0).cuda()).squeeze(0)  # 1 128 120 160 3d->2d
                tgt3_feature2d = backbone2d(tgt_color3.unsqueeze(0).cuda()).squeeze(0)  # 1 128 120 160 3d->2d

                src1_feats = src1_feature2d[:, src1_inds2d[:, 1], src1_inds2d[:, 0]]
                src2_feats = src2_feature2d[:, src2_inds2d[:, 1], src2_inds2d[:, 0]]
                src3_feats = src3_feature2d[:, src3_inds2d[:, 1], src3_inds2d[:, 0]]
                tgt1_feats = tgt1_feature2d[:, tgt1_inds2d[:, 1], tgt1_inds2d[:, 0]]
                tgt2_feats = tgt2_feature2d[:, tgt2_inds2d[:, 1], tgt2_inds2d[:, 0]]
                tgt3_feats = tgt3_feature2d[:, tgt3_inds2d[:, 1], tgt3_inds2d[:, 0]]
                src1_point_with_2dfeature=torch.cat((src1_feats.transpose(1,0),torch.ones(src1_feats.shape[1],1).cuda()),dim=-1).detach()
                src2_point_with_2dfeature=torch.cat((src2_feats.transpose(1,0),torch.ones(src2_feats.shape[1],1).cuda()),dim=-1).detach()
                src3_point_with_2dfeature=torch.cat((src3_feats.transpose(1,0),torch.ones(src3_feats.shape[1],1).cuda()),dim=-1).detach()
                tgt1_point_with_2dfeature=torch.cat((tgt1_feats.transpose(1,0),torch.ones(tgt1_feats.shape[1],1).cuda()),dim=-1).detach()
                tgt2_point_with_2dfeature=torch.cat((tgt2_feats.transpose(1,0),torch.ones(tgt2_feats.shape[1],1).cuda()),dim=-1).detach()
                tgt3_point_with_2dfeature=torch.cat((tgt3_feats.transpose(1,0),torch.ones(tgt3_feats.shape[1],1).cuda()),dim=-1).detach()
                x=x.repeat(1,129)
                "=================================================================="
                tgt1_inds3d=tgt1_inds3d+src_pcd_raw.shape[0]
                tgt2_inds3d=tgt2_inds3d+src_pcd_raw.shape[0]
                tgt3_inds3d=tgt3_inds3d+src_pcd_raw.shape[0]
                x[src3_inds3d,:]=src3_point_with_2dfeature
                x[src2_inds3d,:]=src2_point_with_2dfeature
                x[src1_inds3d,:]=src1_point_with_2dfeature
                x[tgt3_inds3d,:]=tgt3_point_with_2dfeature
                x[tgt2_inds3d,:]=tgt2_point_with_2dfeature
                x[tgt1_inds3d,:]=tgt1_point_with_2dfeature
            elif self.img_num==2:
                src_color1 = batch['src_color1']
                src_color2 = batch['src_color2']
                tgt_color1 = batch['tgt_color1']
                tgt_color2 = batch['tgt_color2']
                id_name = batch['id_name']
                src1_inds2d = batch['src1_inds2d']
                src2_inds2d = batch['src2_inds2d']
                tgt1_inds2d = batch['tgt1_inds2d']
                tgt2_inds2d = batch['tgt2_inds2d']
                src1_inds3d = batch['src1_inds3d']
                src2_inds3d = batch['src2_inds3d']
                tgt1_inds3d = batch['tgt1_inds3d']
                tgt2_inds3d = batch['tgt2_inds3d']
                ## concat 2d feature


                # src1_feature2d = backbone2d(src_color1.unsqueeze(0).cuda()).squeeze(0)  # 1 128 120 160 3d->2d
                # src2_feature2d = backbone2d(src_color2.unsqueeze(0).cuda()).squeeze(0)  # 1 128 120 160 3d->2d
                # tgt1_feature2d = backbone2d(tgt_color1.unsqueeze(0).cuda()).squeeze(0)  # 1 128 120 160 3d->2d
                # tgt2_feature2d = backbone2d(tgt_color2.unsqueeze(0).cuda()).squeeze(0)  # 1 128 120 160 3d->2d
                # src1_feats = src1_feature2d[:, src1_inds2d[:, 1], src1_inds2d[:, 0]]
                # src2_feats = src2_feature2d[:, src2_inds2d[:, 1], src2_inds2d[:, 0]]
                # tgt1_feats = tgt1_feature2d[:, tgt1_inds2d[:, 1], tgt1_inds2d[:, 0]]
                # tgt2_feats = tgt2_feature2d[:, tgt2_inds2d[:, 1], tgt2_inds2d[:, 0]]
                # src1_feature2d = backbone2d(src_color1.unsqueeze(0).cuda()).squeeze(0)  # 1 128 120 160 3d->2d
                # src2_feature2d = backbone2d(src_color2.unsqueeze(0).cuda()).squeeze(0)  # 1 128 120 160 3d->2d
                # tgt1_feature2d = backbone2d(tgt_color1.unsqueeze(0).cuda()).squeeze(0)  # 1 128 120 160 3d->2d
                # tgt2_feature2d = backbone2d(tgt_color2.unsqueeze(0).cuda()).squeeze(0)  # 1 128 120 160 3d->2d
                # src1_feats = src1_feature2d[:, src1_inds2d[:, 1], src1_inds2d[:, 0]]
                # src2_feats = src2_feature2d[:, src2_inds2d[:, 1], src2_inds2d[:, 0]]
                # tgt1_feats = tgt1_feature2d[:, tgt1_inds2d[:, 1], tgt1_inds2d[:, 0]]
                # tgt2_feats = tgt2_feature2d[:, tgt2_inds2d[:, 1], tgt2_inds2d[:, 0]]

                # src1_point_with_2dfeature=torch.cat((src1_feats.transpose(1,0),torch.ones(src1_feats.shape[1],1).cuda()),dim=-1).detach()
                # src2_point_with_2dfeature=torch.cat((src2_feats.transpose(1,0),torch.ones(src2_feats.shape[1],1).cuda()),dim=-1).detach()
                # tgt1_point_with_2dfeature=torch.cat((tgt1_feats.transpose(1,0),torch.ones(tgt1_feats.shape[1],1).cuda()),dim=-1).detach()
                # tgt2_point_with_2dfeature=torch.cat((tgt2_feats.transpose(1,0),torch.ones(tgt2_feats.shape[1],1).cuda()),dim=-1).detach()


                src1_feat=torch.ones((120,160,128)).cuda()
                src2_feat=torch.ones((120,160,128)).cuda()
                tgt1_feat=torch.ones((120,160,128)).cuda()
                tgt2_feat=torch.ones((120,160,128)).cuda()
                src1_feat[batch['detect_1'][:,1],batch['detect_1'][:,0],:]=batch['des1'][:,:]
                src2_feat[batch['detect_2'][:,1],batch['detect_2'][:,0],:]=batch['des2'][:,:]
                tgt1_feat[batch['detect_3'][:,1],batch['detect_3'][:,0],:]=batch['des3'][:,:]
                tgt2_feat[batch['detect_4'][:,1],batch['detect_4'][:,0],:]=batch['des4'][:,:]

                src1_feat,tgt1_feat,src2_feat,tgt2_feat=src1_feat.permute(2,0,1),tgt1_feat.permute(2,0,1),src2_feat.permute(2,0,1),tgt2_feat.permute(2,0,1)
                src1_feats = src1_feat[:, src1_inds2d[:, 1], src1_inds2d[:, 0]]
                src2_feats = src2_feat[:, src2_inds2d[:, 1], src2_inds2d[:, 0]]
                tgt1_feats = tgt1_feat[:, tgt1_inds2d[:, 1], tgt1_inds2d[:, 0]]
                tgt2_feats = tgt2_feat[:, tgt2_inds2d[:, 1], tgt2_inds2d[:, 0]]
                src1_point_with_2dfeature=torch.cat((src1_feats.transpose(1,0),torch.ones(src1_feats.shape[1],1).cuda()),dim=-1).detach()
                src2_point_with_2dfeature=torch.cat((src2_feats.transpose(1,0),torch.ones(src2_feats.shape[1],1).cuda()),dim=-1).detach()
                tgt1_point_with_2dfeature=torch.cat((tgt1_feats.transpose(1,0),torch.ones(tgt1_feats.shape[1],1).cuda()),dim=-1).detach()
                tgt2_point_with_2dfeature=torch.cat((tgt2_feats.transpose(1,0),torch.ones(tgt2_feats.shape[1],1).cuda()),dim=-1).detach()

                # src1_point_with_2dfeature=torch.cat((src1_feat.transpose(1,0),torch.ones(src1_feat.shape[1],1).cuda()),dim=-1).detach()
                #     src2_point_with_2dfeature=torch.cat((src2_feat.transpose(1,0),torch.ones(src2_feat.shape[1],1).cuda()),dim=-1).detach()
                #     tgt1_point_with_2dfeature=torch.cat((tgt1_feat.transpose(1,0),torch.ones(tgt1_feat.shape[1],1).cuda()),dim=-1).detach()
                #     tgt2_point_with_2dfeature=torch.cat((tgt2_feat.transpose(1,0),torch.ones(tgt2_feat.shape[1],1).cuda()),dim=-1).detach()
                x=x.repeat(1,129)
                "=================================================================="
                tgt1_inds3d=tgt1_inds3d+src_pcd_raw.shape[0]
                tgt2_inds3d=tgt2_inds3d+src_pcd_raw.shape[0]
                x[src2_inds3d,:]=src2_point_with_2dfeature
                x[src1_inds3d,:]=src1_point_with_2dfeature
                x[tgt2_inds3d,:]=tgt2_point_with_2dfeature
                x[tgt1_inds3d,:]=tgt1_point_with_2dfeature
            else:
                src_color1 = batch['src_color1']
                tgt_color1 = batch['tgt_color1']
                id_name = batch['id_name']
                src1_inds2d = batch['src1_inds2d']
                tgt1_inds2d = batch['tgt1_inds2d']
                src1_inds3d = batch['src1_inds3d']
                tgt1_inds3d = batch['tgt1_inds3d']
                ## concat 2d feature
                src1_feature2d = backbone2d(src_color1.unsqueeze(0).cuda()).squeeze(0)  # 1 128 120 160 3d->2d
                tgt1_feature2d = backbone2d(tgt_color1.unsqueeze(0).cuda()).squeeze(0)  # 1 128 120 160 3d->2d
                src1_feats = src1_feature2d[:, src1_inds2d[:, 1], src1_inds2d[:, 0]]
                tgt1_feats = tgt1_feature2d[:, tgt1_inds2d[:, 1], tgt1_inds2d[:, 0]]

                src1_point_with_2dfeature=torch.cat((src1_feats.transpose(1,0),torch.ones(src1_feats.shape[1],1).cuda()),dim=-1).detach()
                tgt1_point_with_2dfeature=torch.cat((tgt1_feats.transpose(1,0),torch.ones(tgt1_feats.shape[1],1).cuda()),dim=-1).detach()
                x=x.repeat(1,129)
                "=================================================================="
                tgt1_inds3d=tgt1_inds3d+src_pcd_raw.shape[0]
                x[src1_inds3d,:]=src1_point_with_2dfeature
                x[tgt1_inds3d,:]=tgt1_point_with_2dfeature


        sigmoid = nn.Sigmoid()
        # print("x_shape",x.shape)
        #################################
        # 1. joint encoder part
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        #################################
        # 2. project the bottleneck features
        feats_c = x.transpose(0, 1).unsqueeze(0)  # [1, C, N]
        feats_c = self.bottle(feats_c)  # [1, C, N]

        #################################
        # 3. apply GNN to communicate the features and get overlap score
        src_feats_c, tgt_feats_c = feats_c[:, :, :len_src_c], feats_c[:, :, len_src_c:]
        src_feats_c, tgt_feats_c = self.gnn(src_pcd_c.unsqueeze(0).transpose(1, 2),
                                            tgt_pcd_c.unsqueeze(0).transpose(1, 2), src_feats_c, tgt_feats_c)
        feats_c = torch.cat([src_feats_c, tgt_feats_c], dim=-1)

        feats_c = self.proj_gnn(feats_c)
        scores_c = self.proj_score(feats_c)

        feats_gnn_norm = F.normalize(feats_c, p=2, dim=1).squeeze(0).transpose(0, 1)  # [N, C]
        feats_gnn_raw = feats_c.squeeze(0).transpose(0, 1)
        scores_c_raw = scores_c.squeeze(0).transpose(0, 1)  # [N, 1]


        # super node_overlap supervison
        if self.node_overlap:
            # input feat_gnn_norm  first_[1 256 668] --output[1 1 668] output==> n supernode_predict_overlap_score
            # output node_overlap_score
            node_overlap_score_pred = self.node_overlap_predict(feats_c)
            node_overlap_score_pred = torch.clamp(sigmoid(node_overlap_score_pred.view(-1)),min=0,max=1) # torch.clamp(sigmoid(node_overlap_score_pred.view(-1)),min=0,max=1)
            res['node_overlap_score_pred'] = node_overlap_score_pred

        ####################################
        # 4. decoder part
        src_feats_gnn, tgt_feats_gnn = feats_gnn_norm[:len_src_c], feats_gnn_norm[len_src_c:]
        inner_products = torch.matmul(src_feats_gnn, tgt_feats_gnn.transpose(0, 1))

        src_scores_c, tgt_scores_c = scores_c_raw[:len_src_c], scores_c_raw[len_src_c:]

        temperature = torch.exp(self.epsilon) + 0.03
        s1 = torch.matmul(F.softmax(inner_products / temperature, dim=1), tgt_scores_c)
        s2 = torch.matmul(F.softmax(inner_products.transpose(0, 1) / temperature, dim=1), src_scores_c)
        scores_saliency = torch.cat((s1, s2), dim=0)
        x = torch.cat([scores_c_raw, scores_saliency, feats_gnn_raw], dim=1)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)
        feats_f = x[:, :self.final_feats_dim]
        scores_overlap = x[:, self.final_feats_dim]
        scores_saliency = x[:, self.final_feats_dim + 1]

        # safe guard our score
        scores_overlap = torch.clamp(sigmoid(scores_overlap.view(-1)), min=0, max=1)
        scores_saliency = torch.clamp(sigmoid(scores_saliency.view(-1)), min=0, max=1)
        scores_overlap = self.regular_score(scores_overlap)
        scores_saliency = self.regular_score(scores_saliency)

        # normalise point-wise features
        feats_f = F.normalize(feats_f, p=2, dim=1)

        if self.quaternion:
            # feats_inv=torch.cat([avg_src_feats,avg_tgt_feats],0)
            # feats_inv=self.PartII_To_R_FC(feats_f)#bn 4 1 1

            temp = self.folding1(feats_f)
            feats_inv = self.linear1(temp)
            t = self.linear2(temp)
            # t=np.mean(t.detach().cpu().numpy(),axis=0)
            quaternion_pre = feats_inv[:, :]
            quaternion_pre = quaternion_pre / torch.norm(quaternion_pre, dim=1)[:, None]
            quaternion_pred = torch.mean(quaternion_pre, dim=0)
            res['trans_pred'] = torch.mean(t, dim=0)
            res['quaternion_pred'] = quaternion_pred
            # calculate quaternion_loss
            # rot=matrix_from_quaternion(quaternion_pre)
            # trans=np.vstack((np.hstack((rot,t[:,None])),np.zeros((1,4))))
            # # trans=torch.stack((torch.vstack(rot,t[:,None]),torch.zeros((1,4))))
            # trans[3,3]=1

            # Rt = self.linear1(torch.cat([avg_src_feats,avg_tgt_feats],0))
            # Rt = self.linear2(Rt)

        res['feats_f'] = feats_f
        res['scores_overlap'] = scores_overlap
        res['scores_saliency'] = scores_saliency

        return res
