import time, os, torch,copy
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from lib.timer import Timer, AverageMeter
from lib.utils import Logger,validate_gradient
from datasets.visualize import viz_supernode ,save_ply
from tqdm import tqdm
import torch.nn.functional as F
import gc
import abc
from models.r_eval import quaternion_from_matrix,matrix_from_quaternion,compute_R_diff
from models import build_backbone
def load_state_with_same_shape(model, weights):
    print("Loading weights:" + ', '.join(weights.keys()))
    model_state = model.state_dict()
    filtered_weights = {
        k[9:]: v for k, v in weights.items() if k[9:] in model_state and v.size() == model_state[k[9:]].size()
    }
    print("Loaded weights:" + ', '.join(filtered_weights.keys()))
    return filtered_weights
# def load_state_with_same_shape(model, weights):
#     print("Loading weights:" + ', '.join(weights.keys()))
#     model_state = model.state_dict()
#     filtered_weights = {
#         k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()
#     }
#     print("Loaded weights:" + ', '.join(filtered_weights.keys()))
#     return filtered_weights

class Trainer(object):
    def __init__(self, args):
        self.config = args
        config=args
        # parameters
        self.start_epoch = 1
        self.max_epoch = args.max_epoch
        self.device = args.device
        self.verbose = args.verbose
        self.max_points = args.max_points
        self.init_mode=config.init_mode
        self.node_overlap = config.node_overlap
        self.quaternion = config.quaternion
        self.image_feature=args.image_feature
        self.node_overlap=args.node_overlap
        self.quaternion=args.quaternion

        self.overlap_threshold=args.overlap_threshold
        if self.image_feature:
            if self.init_mode=='image_net':
                self.backbone2d = build_backbone('Res50UNet',128, pretrained=True)
                print('image net init !!!!!!')

            elif self.init_mode=='3dmatch':
                self.backbone2d = build_backbone('Res50UNet',128, pretrained=False)
                path='/home/lab507/yjl/OverlapPredator-my/image_overlap_fixed/testimage_True_node_overlap_False/indoor/model__epoch_27.pth'
                self.backbone2d=self.resume_checkpoint(self.config.tdmatch_pth_path)#.to(self.device)
            elif self.init_mode=='pri3d':
                self.backbone2d = build_backbone('Res50UNet',128, pretrained=True)
                # path='/data1/3dmatch/pretrained_model/checkpoint5.pth'
                self.backbone2d=self.resume_checkpoint(self.config.pri3d_pth_path)#.to(self.device)
                # self.backbone2d = build_backbone('Res50UNet',
                #                                  128, pretrained=True)
                # path='/home/lab507/文档/image1_predator/checkpoint5.pth'
                # self.backbone2d=self.resume_checkpoint(path).to(self.device)
                print('pri3d init !!!!!!!!')
            else:
                self.backbone2d = build_backbone('Res50UNet',128, pretrained=False)
                print('random init!!!!!!!')
            self.backbone2d=self.backbone2d.to(self.device)
        self.model = args.model.to(self.device)
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_freq = args.scheduler_freq
        self.snapshot_freq = args.snapshot_freq
        self.snapshot_dir = args.snapshot_dir
        self.benchmark = args.benchmark
        self.iter_size = args.iter_size
        self.verbose_freq= args.verbose_freq

        self.w_circle_loss = args.w_circle_loss
        self.w_overlap_loss = args.w_overlap_loss
        self.w_saliency_loss = args.w_saliency_loss
        # self.w_node_overlap_loss = args.w_node_overlap_loss
        self.desc_loss = args.desc_loss

        self.best_loss = 1e5
        self.best_recall = -1e5
        # self.temp=torch.zeros(50001,60003)

        # self.snapshot_dir=
        self.save_dir = self.snapshot_dir
        self.tensorboard_dir = os.path.join(self.snapshot_dir,'runs')
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir,exist_ok=True)
        if not os.path.isdir(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir,exist_ok=True)
        self.writer = SummaryWriter(self.tensorboard_dir)
        self.logger = Logger(self.snapshot_dir)
        self.logger.write(f'#parameters {sum([x.nelement() for x in self.model.parameters()])/1000000.} M\n')


        if (args.pretrain !=''):
            self._load_pretrain(args.pretrain)

        self.loader =dict()
        self.loader['train']=args.train_loader
        self.loader['val']=args.val_loader
        self.loader['test'] = args.test_loader

        with open(f'{args.snapshot_dir}/model','w') as f:
            f.write(str(self.model))
        f.close()
    def resume_checkpoint(self, checkpoint_filename='models/checkpoint.pth'):
        import os
        from torch.serialization import default_restore_location
        if os.path.isfile(checkpoint_filename):
            print('===> Loading existing checkpoint')
            state = torch.load(checkpoint_filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
            # print(self.backbone2d)
            # load weights
            model = self.backbone2d
            matched_weights = load_state_with_same_shape(model, state['model'])
            # print("matched weight: ",matched_weights)
            model.load_state_dict(matched_weights, strict=False)
            del state
            return model
 
    def _snapshot(self, epoch, name=None):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_recall': self.best_recall
        }
        if name is None:
            fname=self.save_dir.split('/')[-2]
            benchmark=self.config.benchmark
            filename = os.path.join(self.save_dir, f'{benchmark}_{fname}_{epoch}.pth')
        else:
            filename = os.path.join(self.save_dir, f'model_{name}.pth')
        self.logger.write(f"Save model to {filename}\n")
        torch.save(state, filename)

    def load_state_with_same_shape(self,model, weights):
        self.logger.write("Loading weights:" + ', '.join(weights.keys()))
        model_state = model.state_dict()
        filtered_weights={}
        for k, v in model_state.items():
            if k in weights and v.size() == model_state[k].size():
                filtered_weights[k]=v
            else:
                filtered_weights[k]=model_state[k]

        # filtered_weights = {
        #     k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()
        # }
        self.logger.write("Loaded weights:" + ', '.join(filtered_weights.keys()))
        return filtered_weights

    def _load_pretrain(self, resume):
        if os.path.isfile(resume):
            state = torch.load(resume)
            self.model.load_state_dict(state['state_dict'])
            print("resume dict",state['state_dict'].keys())
            print("model dict",state['state_dict'].keys())

            # matched_weights = self.load_state_with_same_shape(self.model, state['state_dict'])
            # self.model.load_state_dict(matched_weights)
            self.start_epoch = state['epoch']
            self.scheduler.load_state_dict(state['scheduler'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.best_loss = state['best_loss']
            self.best_recall = state['best_recall']
            print((f'Successfully load pretrained model from {resume}!\n'))
            print((f'Current best recall {self.best_recall}\n'))
            print((f'Current best loss {self.best_loss}\n'))
            self.logger.write(f'Successfully load pretrained model from {resume}!\n')
            self.logger.write(f'Current best loss {self.best_loss}\n')
            self.logger.write(f'Current best recall {self.best_recall}\n')
        else:
            raise ValueError(f"=> no checkpoint found at '{resume}'")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']

    def stats_dict(self):
        stats=dict()
        stats['circle_loss']=0.
        stats['recall']=0.  # feature match recall, divided by number of ground truth pairs
        stats['saliency_loss'] = 0.
        stats['saliency_recall'] = 0.
        stats['saliency_precision'] = 0.
        stats['node_overlap_loss'] = 0.
        stats['node_overlap_recall'] = 0.
        stats['node_overlap_precision'] = 0.
        stats['overlap_loss'] = 0.
        stats['overlap_recall']=0.
        stats['overlap_precision']=0.
        stats['pose_loss'] = 0.
        stats['pose_recall']=0.
        stats['pose_precision']=0.
        stats['total_loss']=0.
        return stats

    def stats_meter(self):
        meters=dict()
        stats=self.stats_dict()
        for key,_ in stats.items():
            meters[key]=AverageMeter()
        return meters


    def inference_one_batch(self, inputs, phase):
        assert phase in ['train','val','test']
        ##################################
        # training
        if (phase == 'train'):
            self.model.train()
            ###############################################
            # forward pass #[N1, C1], [N2, C2]
            len_src = inputs['stack_lengths'][0][0]
            if self.image_feature:
                output = self.model(inputs,self.backbone2d)#[N1, C1], [N2, C2]
            else:
                output=self.model(inputs) #[N1, C1], [N2, C2]
            feats, scores_overlap, scores_saliency=output['feats_f'],output['scores_overlap'],output['scores_saliency']
            src_feats, tgt_feats = feats[:len_src], feats[len_src:]

            # get loss
            loss_input={}
            loss_input['src_feats'],loss_input['tgt_feats'],=src_feats,tgt_feats,
            loss_input['trans'],loss_input['rot']=inputs['trans'],inputs['rot']
            loss_input['scores_overlap'], loss_input['scores_saliency']=scores_overlap,scores_saliency

            loss_input['src_pcd_raw'], loss_input['tgt_pcd_raw'] =inputs['src_pcd_raw'],inputs['tgt_pcd_raw']
            #loss_input['node_overlap_list']=inputs['node_overlap_list']
            loss_input['correspondences']=inputs['correspondences']
            if self.node_overlap:

                node_overlap_gt=inputs['node_overlap_gt'].float()

                node_overlap_score=output['node_overlap_score_pred']
                loss_input['node_overlap_gt'],loss_input['node_overlap_score_pred']=node_overlap_gt,node_overlap_score
            if self.quaternion:
                quaternion_pred,trans_pred=output['quaternion_pred'],output['trans_pred']
                # trans_gt = inputs['trans']
                quaternion_gt=quaternion_from_matrix(loss_input['rot'].detach().cpu().numpy())
                quaternion_gt=torch.from_numpy(quaternion_gt).float().cuda()
                loss_input['quaternion_pred'],loss_input['quaternion_gt']=quaternion_pred,quaternion_gt
                loss_input['trans_pred'],loss_input['trans_gt']=trans_pred,inputs['trans'].squeeze()

            losses_name=['circle_loss','overlap_loss','saliency_loss','node_overlap_loss','pose_loss']
            res= self.desc_loss(loss_input)
            c_loss=torch.tensor(0.0).cuda()
            for k  in res.keys():
                if k in losses_name:
                    c_loss+=res[k]

    # c_loss = stats['circle_loss'] * self.w_circle_loss + stats['overlap_loss'] * self.w_overlap_loss + stats['saliency_loss'] * self.w_saliency_loss + stats['node_overlap_loss'] * self.w_node_overlap_loss

            c_loss.backward()

        else:
            self.model.eval()
            with torch.no_grad():
                ###############################################
                    # forward pass
                len_src = inputs['stack_lengths'][0][0]
                if self.image_feature:
                    output = self.model(inputs,self.backbone2d)#[N1, C1], [N2, C2]
                else:
                    output=self.model(inputs)
                feats, scores_overlap, scores_saliency=output['feats_f'],output['scores_overlap'],output['scores_saliency']
                src_feats, tgt_feats = feats[:len_src], feats[len_src:]

                # get loss
                loss_input={}
                loss_input['src_feats'],loss_input['tgt_feats'],=src_feats,tgt_feats,
                loss_input['trans'],loss_input['rot']=inputs['trans'],inputs['rot']
                loss_input['scores_overlap'], loss_input['scores_saliency']=scores_overlap,scores_saliency

                loss_input['src_pcd_raw'], loss_input['tgt_pcd_raw'] =inputs['src_pcd_raw'],inputs['tgt_pcd_raw']
                #loss_input['node_overlap_list']=inputs['node_overlap_list']
                loss_input['correspondences']=inputs['correspondences']
                if self.node_overlap:

                    node_overlap_gt=inputs['node_overlap_gt'].float()

                    node_overlap_score=output['node_overlap_score_pred']
                    loss_input['node_overlap_gt'],loss_input['node_overlap_score_pred']=node_overlap_gt,node_overlap_score
                if self.quaternion:
                    quaternion_pred,trans_pred=output['quaternion_pred'],output['trans_pred']
                    # trans_gt = inputs['trans']
                    quaternion_gt=quaternion_from_matrix(loss_input['rot'].detach().cpu().numpy())
                    quaternion_gt=torch.from_numpy(quaternion_gt).float().cuda()
                    loss_input['quaternion_pred'],loss_input['quaternion_gt']=quaternion_pred,quaternion_gt
                    loss_input['trans_pred'],loss_input['trans_gt']=trans_pred,inputs['trans'].squeeze()

                losses_name=['circle_loss','overlap_loss','saliency_loss','node_overlap_loss','pose_loss']

                res= self.desc_loss(loss_input)
        ##################################        
        # detach the gradients for loss terms
        # stats={}
        # stats['circle_loss'] = float(res['circle_loss'].detach())
        # stats['overlap_loss'] = float(res['overlap_loss'].detach())
        # stats['saliency_loss'] = float(res['saliency_loss'].detach())
        # if self.node_overlap:
        #     stats['node_overlap_loss'] = float(res['node_overlap_loss'].detach())
        # if self.quaternion:
        #     stats['pose_loss'] = float(res['pose_loss'].detach())

        res['circle_loss'] = float(res['circle_loss'].detach())
        res['overlap_loss'] = float(res['overlap_loss'].detach())
        res['saliency_loss'] = float(res['saliency_loss'].detach())
        if self.node_overlap:
            res['node_overlap_loss'] = float(res['node_overlap_loss'].detach())
        if self.quaternion:
            res['pose_loss'] = float(res['pose_loss'].detach())


        return res



    def inference_one_epoch(self,epoch, phase):
        gc.collect()
        assert phase in ['train','val','test']

        # init stats meter
        stats_meter = self.stats_meter()

        num_iter = int(len(self.loader[phase].dataset) // self.loader[phase].batch_size)
        c_loader_iter = self.loader[phase].__iter__()
        
        self.optimizer.zero_grad()
        # for c_iter in tqdm(range(num_iter)): # loop through this epoch
        #     ##################################
        #     # load inputs to device.
        #     inputs = c_loader_iter.next()
        #     print('iii')
        for c_iter in tqdm(range(num_iter)): # loop through this epoch   
            ##################################
            # load inputs to device.
            inputs = c_loader_iter.next()
            for k, v in inputs.items():  
                if type(v) == list:
                    inputs[k] = [item.to(self.device) for item in v]
                elif type(v)==type('d'):
                    inputs[k]=v
                else:
                    inputs[k] = v.to(self.device)
        # try:
            stats = self.inference_one_batch(inputs, phase)

            ###################################################
            # run optimisation
            if((c_iter+1) % self.iter_size == 0 and phase == 'train'):
                gradient_valid = validate_gradient(self.model)
                if(gradient_valid):
                    self.optimizer.step()
                else:
                    self.logger.write('gradient not valid\n')
                self.optimizer.zero_grad()

            ################################
            # update to stats_meter
            for key,value in stats.items():
                stats_meter[key].update(value)
        # except Exception as inst:
        #     print(inst)
            
            torch.cuda.empty_cache()
            
            if (c_iter + 1) % self.verbose_freq == 0 and self.verbose:
                curr_iter = num_iter * (epoch - 1) + c_iter
                for key, value in stats_meter.items():
                    self.writer.add_scalar(f'{phase}/{key}', value.avg, curr_iter)
                    self.writer.add_scalar('lr', self._get_lr(), curr_iter)

                message = f'{phase} Epoch: {epoch} [{c_iter+1:4d}/{num_iter}]  lr: {self._get_lr() }  '
                for key,value in stats_meter.items():
                    message += f'{key}: {value.avg:.2f}\t'

                self.logger.write(message + '\n')
                print('\n'+message + '\n'+self.save_dir+'\n')

        message = f'{phase} Epoch: {epoch}'
        for key,value in stats_meter.items():
            message += f'{key}: {value.avg:.2f}\t'
        self.logger.write(message+'\n')
        print(message+'\n')

        return stats_meter


    def train(self):
        print('start training...')
        for epoch in range(self.start_epoch, self.max_epoch):
            self.inference_one_epoch(epoch,'train')
            self.scheduler.step()
            
            stats_meter = self.inference_one_epoch(epoch,'val')
            
            if stats_meter['circle_loss'].avg < self.best_loss:
                self.best_loss = stats_meter['circle_loss'].avg
                self._snapshot(epoch,'best_loss')
            if stats_meter['recall'].avg > self.best_recall:
                self.best_recall = stats_meter['recall'].avg
                self._snapshot(epoch,'best_recall')
            if (epoch+1) % 1==0:
                self._snapshot(epoch)
            # we only add saliency loss when we get descent point-wise features
            if(stats_meter['recall'].avg>0.3):
                self.w_saliency_loss = 1.
            else:
                self.w_saliency_loss = 0.
                    
        # finish all epoch
        print("Training finish!")


    def eval(self):
        print('Start to evaluate on validation datasets...')
        stats_meter = self.inference_one_epoch(0,'val')
        
        for key, value in stats_meter.items():
            print(key, value.avg)
