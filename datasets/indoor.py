import cv2
import random
import os, sys, glob, torch
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
# import open3d as o3d
import glob
from lib.benchmark_utils import to_o3d_pcd, to_tsfm, get_correspondences
from datasets.visualize import make_open3d_point_cloud, draw_registration_result
from PIL import Image, ImageFilter
from torchvision.transforms import transforms
from projection import Projection
from datasets.visualize import viz_supernode, save_ply, draw_registration_result, unproject, adjust_intrinsic

random.seed(0)
_imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class IndoorDataset(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """

    def __init__(self, infos, config, data_augmentation=True):
        super(IndoorDataset, self).__init__()
        for k, v in infos.items():
            infos[k] = infos[k][:]
        self.infos = infos
        self.base_dir = config.root
        self.path = config.img_path
        self.matches_path=config.superglue_matches_path
        self.overlap_radius = config.overlap_radius
        self.data_augmentation = data_augmentation
        self.config = config

        self.image_feature = config.image_feature
        self.img_num = config.img_num
        self.node_overlap = config.node_overlap
        self.quaternion = config.quaternion
        self.window_size=config.window_size

        self.rot_factor = 1.
        self.augment_noise = config.augment_noise
        self.max_points = 30000
        self.image_size = [240, 320]  # [480,640]
        self.depth_img_size = [120, 160]  # [480,640]

        self.transforms = {}
        self.transforms["rgb"] = transforms.Compose([
            transforms.Resize(self.image_size, Image.NEAREST),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize(_imagenet_stats['mean'], std=_imagenet_stats['std'])
        ])
        self.transforms["depth"] = transforms.Compose([
            transforms.Resize(self.depth_img_size, Image.NEAREST),
            transforms.ToTensor(),
        ])
        src_list = self.infos['src']
        tgt_list = self.infos['tgt']

        fnames = os.listdir(self.base_dir)
        scene_names = set([fnames[i].split('@')[0] for i in range(len(fnames))])

        self.src_scene_id_list, self.src_full_scene_id_list, self.src_seq_id_list, self.src_image_id1_list, self.src_image_id2_list = self.split_info(
            src_list)
        self.tgt_scene_id_list, self.tgt_full_scene_id_list, self.tgt_seq_id_list, self.tgt_image_id1_list, self.tgt_image_id2_list = self.split_info(
            tgt_list)

    def __len__(self):
        return len(self.infos['rot'])

    def split_info(self, list, scene_names=None):
        scene_id_list = []
        full_scene_id_list = []
        seq_id_list = []
        image_id1 = []
        image_id2 = []

        for i, fname in enumerate(list):
            phase, scene_id, image_id = fname.split('/')
            # for i in range(len(scene_names)):
            #     if scene_id[:-3] in scene_names[i]:
            #         scene_id=scene_names[i]
            txt_path = image_id[:-4] + '.info.txt'
            with open(os.path.join(self.base_dir, phase, scene_id, txt_path), 'r') as f:
                line = f.readline()
                full_scene_id, seq_id, id1, id2 = line.split()
            f.close()
            scene_id_list.append(scene_id)
            full_scene_id_list.append(full_scene_id)
            seq_id_list.append(seq_id)
            image_id1.append(id1)
            image_id2.append(id2)
        return scene_id_list, full_scene_id_list, seq_id_list, image_id1, image_id2

    def get_coords(self, keypoints):
        detection = np.zeros((len(keypoints), 2))
        for i in range(len(keypoints)):
            detection[i] = (keypoints[i].pt[0], keypoints[i].pt[1])
        return detection

    def __getitem__(self, item):
        # item=190
        # get transformation
        data = {}
        rot = self.infos['rot'][item]
        trans = self.infos['trans'][item]

        # get pointcloud
        src_path = os.path.join(self.base_dir, self.infos['src'][item])
        tgt_path = os.path.join(self.base_dir, self.infos['tgt'][item])
        src_pcd = torch.load(src_path)
        tgt_pcd = torch.load(tgt_path)

        # draw_registration_result(src_pcd,tgt_pcd,Window_name='src_pcd,tgt_pcd 11')

        # save_ply(src_pcd,f'/home/lab507/yjl/frame2/frame{item}_src_load_pcd_before_augmentation')
        # save_ply(tgt_pcd,f'/home/lab507/yjl/frame2/frame{item}_tgt_load_pcd_before_augmentation')

        # if we get too many points, we do some downsampling
        if (src_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(src_pcd.shape[0])[:self.max_points]
            src_pcd = src_pcd[idx]
        if (tgt_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(tgt_pcd.shape[0])[:self.max_points]
            tgt_pcd = tgt_pcd[idx]
        src_bin_pcd = src_pcd  # np.matmul(rot,src_pcd.T).T+trans.T
        tgt_bin_pcd = tgt_pcd
        # add gaussian noise
        if self.data_augmentation:
            # rotate the point cloud
            # print("augment rotation to source point cloud")
            euler_ab = np.random.rand(3) * np.pi * 2 / self.rot_factor  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            aug_src = np.random.rand(1)[0]
            if (aug_src > 0.5):
                src_pcd = np.matmul(rot_ab, src_pcd.T).T
                rot = np.matmul(rot, rot_ab.T)
            else:
                # print("rotate the point cloud")

                tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T
                rot = np.matmul(rot_ab, rot)
                trans = np.matmul(rot_ab, trans)

            src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * self.augment_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * self.augment_noise

        # save_ply(src_pcd,f'/home/lab507/yjl/frame2/frame{item}_src_load_pcd_after_augmentation')
        # save_ply(tgt_pcd,f'/home/lab507/yjl/frame2/frame{item}_tgt_load_pcd_after_augmentation')
        if (trans.ndim == 1):
            trans = trans[:, None]

        # get correspondence at fine level
        tsfm = to_tsfm(rot, trans)
        correspondences = get_correspondences(to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd), tsfm, self.overlap_radius)

        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        rot = rot.astype(np.float32)
        trans = trans.astype(np.float32)
        data['src_pcd'] = src_pcd
        data['tgt_pcd'] = tgt_pcd
        data['src_feats'] = src_feats
        data['tgt_feats'] = tgt_feats
        data['rot'] = rot
        data['trans'] = trans
        data['correspondences'] = correspondences
        data['sample'] = torch.ones(1)

        if self.image_feature:
            if self.img_num == 1:
                src_image_id1 = self.src_image_id1_list[item]
                # src_image_id2 = self.src_image_id2_list[item]
                tgt_image_id1 = self.tgt_image_id1_list[item]
                # tgt_image_id2 = self.tgt_image_id2_list[item]
                src_scene_id1 = self.src_full_scene_id_list[item]
                src_seq_id1 = self.src_seq_id_list[item]
                tgt_seq_id1 = self.tgt_seq_id_list[item]
                tgt_scene_id1 = self.tgt_full_scene_id_list[item]

                # file_name=os.path.join(self.data_dir,f'{src_scene_id1}_{src_seq_id1}_{src_image_id1}_{tgt_image_id1}.pth')
                # if os.path.exists(file_name):
                #     data_pth = torch.load(file_name)
                #     for k in data_pth.keys():
                #         data[k]=data_pth[k]

                src_color_path1 = os.path.join(self.path, src_scene_id1, src_seq_id1,
                                               "frame-" + src_image_id1.zfill(6) + ".color.png")
                # src_color_path2 = os.path.join(self.path, src_scene_id1, src_seq_id1, "frame-"+src_image_id2.zfill(6)+".color.png")
                tgt_color_path1 = os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                               "frame-" + tgt_image_id1.zfill(6) + ".color.png")
                # tgt_color_path2 = os.path.join(self.path, tgt_scene_id1,  tgt_seq_id1, "frame-"+tgt_image_id2.zfill(6)+".color.png")
                if os.path.isfile(src_color_path1):
                    src_color_image1 = Image.open(src_color_path1)
                    # src_color_image2 = Image.open(src_color_path2)
                    tgt_color_image1 = Image.open(tgt_color_path1)
                    # tgt_color_image2 = Image.open(tgt_color_path2)
                else:
                    src_color_path1 = os.path.join(self.path, src_scene_id1, src_seq_id1,
                                                   "frame-" + src_image_id1.zfill(6) + ".color.jpg")
                    # src_color_path2 = os.path.join(self.path, src_scene_id1, src_seq_id1, "frame-"+src_image_id2.zfill(6)+".color.jpg")
                    tgt_color_path1 = os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                                   "frame-" + tgt_image_id1.zfill(6) + ".color.jpg")
                    # tgt_color_path2 = os.path.join(self.path, tgt_scene_id1,  tgt_seq_id1, "frame-"+tgt_image_id2.zfill(6)+".color.jpg")
                    src_color_image1 = Image.open(src_color_path1)
                    # src_color_image2 = Image.open(src_color_path2)
                    tgt_color_image1 = Image.open(tgt_color_path1)
                    # tgt_color_image2 = Image.open(tgt_color_path2)
                src_color_image1 = self.transforms["rgb"](src_color_image1)  # center crop, resize, normalize
                # src_color_image2 = self.transforms["rgb"](src_color_image2)  # center crop, resize, normalize
                src_depth_path1 = os.path.join(self.path, src_scene_id1, src_seq_id1,
                                               "frame-" + src_image_id1.zfill(6) + ".depth.png")
                # src_depth_path2 = os.path.join(self.path, src_scene_id1,src_seq_id1, "frame-"+src_image_id2.zfill(6)+".depth.png")
                src_depth_image1 = Image.open(src_depth_path1)  # 640 480
                # src_depth_image2 = Image.open(src_depth_path2)  # 640 480
                src_depth_image1 = self.transforms["depth"](src_depth_image1) / 1000.0
                # src_depth_image2 = self.transforms["depth"](src_depth_image2)/1000.0

                # tgt image
                tgt_color_image1 = self.transforms["rgb"](tgt_color_image1)  # center crop, resize, normalize
                # tgt_color_image2 = self.transforms["rgb"](tgt_color_image2)  # center crop, resize, normalize
                tgt_depth_path1 = os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                               "frame-" + tgt_image_id1.zfill(6) + ".depth.png")
                # tgt_depth_path2 = os.path.join(self.path, tgt_scene_id1, tgt_seq_id1, "frame-"+tgt_image_id2.zfill(6)+".depth.png")
                tgt_depth_image1 = Image.open(tgt_depth_path1)  # 640 480
                # tgt_depth_image2 = Image.open(tgt_depth_path2)  # 640 480
                tgt_depth_image1 = self.transforms["depth"](tgt_depth_image1) / 1000.0
                # tgt_depth_image2 = self.transforms["depth"](tgt_depth_image2)/1000.0
                #############################superglue
                # draw the box of the overlapped pixel
                path0,path1=src_color_path1.split('/'),tgt_color_path1.split('/')
                # path0_2,path1_2=src_color_path2.split('/'),tgt_color_path2.split('/')
                stem0, stem1 = f'{path0[-3]}_{path0[-2]}_{path0[-1][:-10]}', f'{path1[-3]}_{path1[-2]}_{path1[-1][:-10]}'
                # stem0_2, stem1_2 = f'{path0_2[-3]}_{path0_2[-2]}_{path0_2[-1][:-10]}', f'{path1_2[-3]}_{path1_2[-2]}_{path1_2[-1][:-10]}'
                dump_matches_path=self.matches_path
                path=os.path.join(dump_matches_path,f'{stem0}_{stem1}_matches.npz')
                # path2=os.path.join(dump_matches_path,f'{stem0_2}_{stem1_2}_matches.npz')
                # path = f'/home/lab507/data/SuperGluePretrainedNetwork-master/dump_match_pairs_160/{stem0}_{stem1}_matches.npz'
                # path2 = f'/home/lab507/data/SuperGluePretrainedNetwork-master/dump_match_pairs_160/{stem0_2}_{stem1_2}_matches.npz'
                npz = np.load(path)
                # npz2 = np.load(path2)
                a = npz.files
                src1_keyspts0 = npz['keypoints0']
                # src2_keyspts0 = npz2['keypoints0']
                tgt1_keyspts1 = npz['keypoints1']
                # tgt2_keyspts1 = npz2['keypoints1']
                src1_confidence = npz['match_confidence']
                # src2_confidence = npz2['match_confidence']
                src1_matches = npz['matches']
                # src2_matches = npz2['matches']
                valid1 = src1_matches > -1
                # valid2 = src2_matches > -1
                mkpts0 = src1_keyspts0[valid1]
                mkpts1 = tgt1_keyspts1[src1_matches[valid1]]
                # valid_keypts=np.sum(npz['matches']>-1)
                # top_k=40
                # top_k_idx=confidence.argsort()[::-1][0:top_k]
                # print(top_k_idx)
                # src1_matches[top_k_idx]
                # tgt1_keyspts1[src1_matches[top_k_idx]]
                # src1_keyspts0[top_k_idx]
                w = self.window_size
                src_valid_map1 = np.zeros((160, 120))
                # src_valid_map2 = np.zeros((160, 120))
                tgt_valid_map1 = np.zeros((160, 120))
                # tgt_valid_map2 = np.zeros((160, 120))
                for i in range(len(src1_keyspts0[valid1])):
                    left_up = int(src1_keyspts0[valid1][i][0] - w)
                    right_up = int(src1_keyspts0[valid1][i][0] + w)
                    left_down = int(src1_keyspts0[valid1][i][1] - w)
                    right_down = int(src1_keyspts0[valid1][i][1] + w)
                    src_valid_map1[left_up:right_up, left_down:right_down] = src1_confidence[valid1][i]
                    left_up = int(tgt1_keyspts1[src1_matches[valid1]][i][0] - w)
                    right_up = int(tgt1_keyspts1[src1_matches[valid1]][i][0] + w)
                    left_down = int(tgt1_keyspts1[src1_matches[valid1]][i][1] - w)
                    right_down = int(tgt1_keyspts1[src1_matches[valid1]][i][1] + w)
                    tgt_valid_map1[left_up:right_up, left_down:right_down] = src1_confidence[valid1][i]

                ###################### #############################superglue

                intrinsics = np.loadtxt(os.path.join(self.path, src_scene_id1, 'camera-intrinsics.txt'))
                big_size, image_size = [640, 480], [160, 120]
                # tranpose height and width
                intrinsics = adjust_intrinsic(intrinsics, big_size, image_size)

                if intrinsics.shape[0] == 3:
                    res = np.eye(4)
                    res[:3, :3] = intrinsics
                    intrinsics = res
                src_pose1 = np.loadtxt(os.path.join(self.path, src_scene_id1, src_seq_id1,
                                                    "frame-" + src_image_id1.zfill(6) + ".pose.txt"))
                # src_pose2 = np.loadtxt( os.path.join(self.path, src_scene_id1, src_seq_id1, "frame-"+src_image_id2.zfill(6)+".pose.txt"))
                tgt_pose1 = np.loadtxt(os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                                    "frame-" + tgt_image_id1.zfill(6) + ".pose.txt"))
                # tgt_pose2 = np.loadtxt( os.path.join(self.path, tgt_scene_id1, tgt_seq_id1, "frame-"+tgt_image_id2.zfill(6)+".pose.txt"))
                eye = np.eye(4)

                "back to  image space"
                intrinsics = torch.from_numpy(intrinsics).float()
                projection = Projection(intrinsics)
                if self.data_augmentation:

                    if (aug_src > 0.5):
                        # print("augment rotation to source point cloud")
                        src1_world2camera = np.eye(4)
                        src1_world2camera[:3, :3] = np.linalg.inv(rot_ab)
                        src1_world2camera = torch.from_numpy(src1_world2camera).float()

                        tgt1_world2camera = torch.eye(4).float()
                    else:
                        # print("augment rotation to target point cloud")
                        tgt1_world2camera = np.eye(4)
                        tgt1_world2camera[:3, :3] = np.linalg.inv(rot_ab)
                        tgt1_world2camera = torch.from_numpy(tgt1_world2camera).float()
                        src1_world2camera = torch.eye(4).float()
                else:
                    src1_world2camera = torch.eye(4).float()
                    tgt1_world2camera = torch.eye(4).float()
                src_pose1_rev = np.linalg.inv(src_pose1)
                tgt_pose1_rev = np.linalg.inv(tgt_pose1)

                src_init_pcd = torch.from_numpy(src_pcd).float()
                tgt_init_pcd = torch.from_numpy(tgt_pcd).float()
                src1_inds2d, src1_inds3d = projection.projection(src_init_pcd, src_depth_image1, src1_world2camera)
                # src2_inds2d, src2_inds3d=projection.projection(src_init_pcd,src_depth_image2,src2_world2camera)
                tgt1_inds2d, tgt1_inds3d = projection.projection(tgt_init_pcd, tgt_depth_image1, tgt1_world2camera)

                id_name = f'item_{item}_' + src_scene_id1 + '__src' + src_image_id1 + '__tgt' + tgt_image_id1
                data['src_valid_map1'] = torch.from_numpy(src_valid_map1).float()
                # data['src_valid_map2'] = torch.from_numpy(src_valid_map2).float()
                data['tgt_valid_map1'] = torch.from_numpy(tgt_valid_map1).float()
                # data['tgt_valid_map2'] = torch.from_numpy(tgt_valid_map2).float()
                data['src1_inds2d'] = src1_inds2d.long()
                # data['src2_inds2d']=src2_inds2d.long()
                data['tgt1_inds2d'] = tgt1_inds2d.long()
                # data['tgt2_inds2d']=tgt2_inds2d.long()
                data['src1_inds3d'] = src1_inds3d.long()
                # data['src2_inds3d']=src2_inds3d.long()
                data['tgt1_inds3d'] = tgt1_inds3d.long()
                # data['tgt2_inds3d']=tgt2_inds3d.long()
                data['src_color1'] = src_color_image1
                # data['src_color2']=src_color_image2
                data['tgt_color1'] = tgt_color_image1
                # data['tgt_color2']=tgt_color_image2
                data['id_name'] = id_name
            elif self.img_num == 2:

                img_pair_name = './image-pair-3dmatchtest.txt'
                # sift = cv2.xfeatures2d.SIFT_create()
                src_image_id1 = self.src_image_id1_list[item]
                src_image_id2 = self.src_image_id2_list[item]
                tgt_image_id1 = self.tgt_image_id1_list[item]
                tgt_image_id2 = self.tgt_image_id2_list[item]
                src_scene_id1 = self.src_full_scene_id_list[item]
                src_seq_id1 = self.src_seq_id_list[item]
                tgt_seq_id1 = self.tgt_seq_id_list[item]
                tgt_scene_id1 = self.tgt_full_scene_id_list[item]

                src_color_path1 = os.path.join(self.path, src_scene_id1, src_seq_id1,
                                               "frame-" + src_image_id1.zfill(6) + ".color.png")
                src_color_path2 = os.path.join(self.path, src_scene_id1, src_seq_id1,
                                               "frame-" + src_image_id2.zfill(6) + ".color.png")
                tgt_color_path1 = os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                               "frame-" + tgt_image_id1.zfill(6) + ".color.png")
                tgt_color_path2 = os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                               "frame-" + tgt_image_id2.zfill(6) + ".color.png")

                if os.path.isfile(src_color_path1):
                    src_color_image1 = Image.open(src_color_path1)
                    src_color_image2 = Image.open(src_color_path2)
                    tgt_color_image1 = Image.open(tgt_color_path1)
                    tgt_color_image2 = Image.open(tgt_color_path2)

                    # with open(img_pair_name, 'a') as f:
                    #     f.write(f'{src_color_path1} {tgt_color_path1} \n')
                    #     f.write(f'{src_color_path2} {tgt_color_path2} \n')
                    # src_img1 = cv2.imread(src_color_path1)
                    # src_img2 = cv2.imread(src_color_path2)
                    # tgt_img1 = cv2.imread(tgt_color_path1)
                    # tgt_img2 = cv2.imread(tgt_color_path2)
                    # dim = (160, 120)
                    # src_img1 = cv2.resize(src_img1, dim, interpolation=cv2.INTER_AREA)
                    # src_img2 = cv2.resize(src_img2, dim, interpolation=cv2.INTER_AREA)
                    # tgt_img1 = cv2.resize(tgt_img1, dim, interpolation=cv2.INTER_AREA)
                    # tgt_img2 = cv2.resize(tgt_img2, dim, interpolation=cv2.INTER_AREA)

                    # # gray1 = cv2.cvtColor(src_img1, cv2.COLOR_BGR2GRAY) #灰度处理图像
                    # kp1, des1 = sift.detectAndCompute(src_img1,None)   #des是描述子
                    #
                    # # gray2 = cv2.cvtColor(src_img2, cv2.COLOR_BGR2GRAY)#灰度处理图像
                    # kp2, des2 = sift.detectAndCompute(src_img2,None)  #des是描述子
                    # # gray1 = cv2.cvtColor(tgt_img1, cv2.COLOR_BGR2GRAY) #灰度处理图像
                    # kp3, des3 = sift.detectAndCompute(tgt_img1,None)   #des是描述子
                    #
                    # # gray2 = cv2.cvtColor(tgt_img2, cv2.COLOR_BGR2GRAY)#灰度处理图像
                    # kp4, des4 = sift.detectAndCompute(tgt_img2,None)  #des是描述子
                    # detect_1=self.get_coords(kp1)
                    # detect_2=self.get_coords(kp2)
                    # detect_3=self.get_coords(kp3)
                    # detect_4=self.get_coords(kp4)
                else:
                    src_color_path1 = os.path.join(self.path, src_scene_id1, src_seq_id1,
                                                   "frame-" + src_image_id1.zfill(6) + ".color.jpg")
                    src_color_path2 = os.path.join(self.path, src_scene_id1, src_seq_id1,
                                                   "frame-" + src_image_id2.zfill(6) + ".color.jpg")
                    tgt_color_path1 = os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                                   "frame-" + tgt_image_id1.zfill(6) + ".color.jpg")
                    tgt_color_path2 = os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                                   "frame-" + tgt_image_id2.zfill(6) + ".color.jpg")
                    # with open(img_pair_name, 'a') as f:
                    #     f.write(f'{src_color_path1} {tgt_color_path1} \n')
                    #     f.write(f'{src_color_path2} {tgt_color_path2} \n')

                    # src_img1 = cv2.imread(src_color_path1)
                    # src_img2 = cv2.imread(src_color_path2)
                    # tgt_img1 = cv2.imread(tgt_color_path1)
                    # tgt_img2 = cv2.imread(tgt_color_path2)
                    # dim = (160, 120)
                    # src_img1 = cv2.resize(src_img1, dim, interpolation=cv2.INTER_AREA)
                    # src_img2 = cv2.resize(src_img2, dim, interpolation=cv2.INTER_AREA)
                    # tgt_img1 = cv2.resize(tgt_img1, dim, interpolation=cv2.INTER_AREA)
                    # tgt_img2 = cv2.resize(tgt_img2, dim, interpolation=cv2.INTER_AREA)

                    # gray1 = cv2.cvtColor(src_img1, cv2.COLOR_BGR2GRAY) #灰度处理图像
                    # kp1, des1 = sift.detectAndCompute(src_img1,None)   #des是描述子
                    #
                    # # gray2 = cv2.cvtColor(src_img2, cv2.COLOR_BGR2GRAY)#灰度处理图像
                    # kp2, des2 = sift.detectAndCompute(src_img2,None)  #des是描述子
                    # # gray1 = cv2.cvtColor(tgt_img1, cv2.COLOR_BGR2GRAY) #灰度处理图像
                    # kp3, des3 = sift.detectAndCompute(tgt_img1,None)   #des是描述子
                    #
                    # # gray2 = cv2.cvtColor(tgt_img2, cv2.COLOR_BGR2GRAY)#灰度处理图像
                    # kp4, des4 = sift.detectAndCompute(tgt_img2,None)  #des是描述子
                    # detect_1=self.get_coords(kp1)
                    # detect_2=self.get_coords(kp2)
                    # detect_3=self.get_coords(kp3)
                    # detect_4=self.get_coords(kp4)

                    src_color_image1 = Image.open(src_color_path1)
                    src_color_image2 = Image.open(src_color_path2)
                    tgt_color_image1 = Image.open(tgt_color_path1)
                    tgt_color_image2 = Image.open(tgt_color_path2)
                src_color_image1 = self.transforms["rgb"](src_color_image1)  # center crop, resize, normalize
                src_color_image2 = self.transforms["rgb"](src_color_image2)  # center crop, resize, normalize
                src_depth_path1 = os.path.join(self.path, src_scene_id1, src_seq_id1,
                                               "frame-" + src_image_id1.zfill(6) + ".depth.png")
                src_depth_path2 = os.path.join(self.path, src_scene_id1, src_seq_id1,
                                               "frame-" + src_image_id2.zfill(6) + ".depth.png")
                src_depth_image1 = Image.open(src_depth_path1)  # 640 480
                src_depth_image2 = Image.open(src_depth_path2)  # 640 480
                src_depth_image1 = self.transforms["depth"](src_depth_image1) / 1000.0
                src_depth_image2 = self.transforms["depth"](src_depth_image2) / 1000.0

                # tgt image
                tgt_color_image1 = self.transforms["rgb"](tgt_color_image1)  # center crop, resize, normalize
                tgt_color_image2 = self.transforms["rgb"](tgt_color_image2)  # center crop, resize, normalize
                tgt_depth_path1 = os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                               "frame-" + tgt_image_id1.zfill(6) + ".depth.png")
                tgt_depth_path2 = os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                               "frame-" + tgt_image_id2.zfill(6) + ".depth.png")
                tgt_depth_image1 = Image.open(tgt_depth_path1)  # 640 480
                tgt_depth_image2 = Image.open(tgt_depth_path2)  # 640 480
                tgt_depth_image1 = self.transforms["depth"](tgt_depth_image1) / 1000.0
                tgt_depth_image2 = self.transforms["depth"](tgt_depth_image2) / 1000.0
   #############################superglue
                # draw the box of the overlapped pixel
                path0,path1=src_color_path1.split('/'),tgt_color_path1.split('/')
                path0_2,path1_2=src_color_path2.split('/'),tgt_color_path2.split('/')
                stem0, stem1 = f'{path0[-3]}_{path0[-2]}_{path0[-1][:-10]}', f'{path1[-3]}_{path1[-2]}_{path1[-1][:-10]}'
                stem0_2, stem1_2 = f'{path0_2[-3]}_{path0_2[-2]}_{path0_2[-1][:-10]}', f'{path1_2[-3]}_{path1_2[-2]}_{path1_2[-1][:-10]}'
                dump_matches_path=self.matches_path
                path=os.path.join(dump_matches_path,f'{stem0}_{stem1}_matches.npz')
                path2=os.path.join(dump_matches_path,f'{stem0_2}_{stem1_2}_matches.npz')
                # path = f'/home/lab507/data/SuperGluePretrainedNetwork-master/dump_match_pairs_160/{stem0}_{stem1}_matches.npz'
                # path2 = f'/home/lab507/data/SuperGluePretrainedNetwork-master/dump_match_pairs_160/{stem0_2}_{stem1_2}_matches.npz'
                npz = np.load(path)
                npz2 = np.load(path2)
                a = npz.files
                src1_keyspts0 = npz['keypoints0']
                src2_keyspts0 = npz2['keypoints0']
                tgt1_keyspts1 = npz['keypoints1']
                tgt2_keyspts1 = npz2['keypoints1']
                src1_confidence = npz['match_confidence']
                src2_confidence = npz2['match_confidence']
                src1_matches = npz['matches']
                src2_matches = npz2['matches']
                valid1 = src1_matches > -1
                valid2 = src2_matches > -1
                mkpts0 = src1_keyspts0[valid1]
                mkpts1 = tgt1_keyspts1[src1_matches[valid1]]
                # valid_keypts=np.sum(npz['matches']>-1)
                # top_k=40
                # top_k_idx=confidence.argsort()[::-1][0:top_k]
                # print(top_k_idx)
                # src1_matches[top_k_idx]
                # tgt1_keyspts1[src1_matches[top_k_idx]]
                # src1_keyspts0[top_k_idx]
                w = self.window_size
                src_valid_map1 = np.zeros((160, 120))
                src_valid_map2 = np.zeros((160, 120))
                tgt_valid_map1 = np.zeros((160, 120))
                tgt_valid_map2 = np.zeros((160, 120))
                for i in range(len(src1_keyspts0[valid1])):
                    left_up = int(src1_keyspts0[valid1][i][0] - w)
                    right_up = int(src1_keyspts0[valid1][i][0] + w)
                    left_down = int(src1_keyspts0[valid1][i][1] - w)
                    right_down = int(src1_keyspts0[valid1][i][1] + w)
                    src_valid_map1[left_up:right_up, left_down:right_down] = src1_confidence[valid1][i]
                    left_up = int(tgt1_keyspts1[src1_matches[valid1]][i][0] - w)
                    right_up = int(tgt1_keyspts1[src1_matches[valid1]][i][0] + w)
                    left_down = int(tgt1_keyspts1[src1_matches[valid1]][i][1] - w)
                    right_down = int(tgt1_keyspts1[src1_matches[valid1]][i][1] + w)
                    tgt_valid_map1[left_up:right_up, left_down:right_down] = src1_confidence[valid1][i]
                for i in range(len(src2_keyspts0[valid2])):
                    left_up2 = int(src2_keyspts0[valid2][i][0] - w)
                    right_up2 = int(src2_keyspts0[valid2][i][0] + w)
                    left_down2 = int(src2_keyspts0[valid2][i][1] - w)
                    right_down2 = int(src2_keyspts0[valid2][i][1] + w)
                    src_valid_map2[left_up2:right_up2, left_down2:right_down2] = src2_confidence[valid2][i]
                    left_up2 = int(tgt2_keyspts1[src2_matches[valid2]][i][0] - w)
                    right_up2 = int(tgt2_keyspts1[src2_matches[valid2]][i][0] + w)
                    left_down2 = int(tgt2_keyspts1[src2_matches[valid2]][i][1] - w)
                    right_down2 = int(tgt2_keyspts1[src2_matches[valid2]][i][1] + w)
                    tgt_valid_map2[left_up2:right_up2, left_down2:right_down2] = src2_confidence[valid2][i]
 ###################### #############################superglue
                intrinsics = np.loadtxt(os.path.join(self.path, src_scene_id1, 'camera-intrinsics.txt'))
                big_size, image_size = [640, 480], [160, 120]
                # tranpose height and width
                intrinsics = adjust_intrinsic(intrinsics, big_size, image_size)

                if intrinsics.shape[0] == 3:
                    res = np.eye(4)
                    res[:3, :3] = intrinsics
                    intrinsics = res
                src_pose1 = np.loadtxt(os.path.join(self.path, src_scene_id1, src_seq_id1,
                                                    "frame-" + src_image_id1.zfill(6) + ".pose.txt"))
                src_pose2 = np.loadtxt(os.path.join(self.path, src_scene_id1, src_seq_id1,
                                                    "frame-" + src_image_id2.zfill(6) + ".pose.txt"))
                tgt_pose1 = np.loadtxt(os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                                    "frame-" + tgt_image_id1.zfill(6) + ".pose.txt"))
                tgt_pose2 = np.loadtxt(os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                                    "frame-" + tgt_image_id2.zfill(6) + ".pose.txt"))
                eye = np.eye(4)

                "back to  image space"

                if self.data_augmentation:

                    if (aug_src > 0.5):
                        # print("augment rotation to source point cloud")
                        src1_world2camera = np.eye(4)
                        src1_world2camera[:3, :3] = np.linalg.inv(rot_ab)
                        src1_world2camera = torch.from_numpy(src1_world2camera).float()

                        tgt1_world2camera = torch.eye(4).float()
                    else:
                        # print("augment rotation to target point cloud")
                        tgt1_world2camera = np.eye(4)
                        tgt1_world2camera[:3, :3] = np.linalg.inv(rot_ab)
                        tgt1_world2camera = torch.from_numpy(tgt1_world2camera).float()
                        src1_world2camera = torch.eye(4).float()
                else:
                    src1_world2camera = torch.eye(4).float()
                    tgt1_world2camera = torch.eye(4).float()
                src_pose1_rev = np.linalg.inv(src_pose1)
                src_pose2_rev = np.linalg.inv(src_pose2)
                tgt_pose1_rev = np.linalg.inv(tgt_pose1)
                tgt_pose2_rev = np.linalg.inv(tgt_pose2)
                src2_world2camera = torch.mm(torch.from_numpy(src_pose2_rev).float(),
                                             torch.mm(torch.from_numpy(src_pose1).float(), src1_world2camera))
                tgt2_world2camera = torch.mm(torch.from_numpy(tgt_pose2_rev).float(),
                                             torch.mm(torch.from_numpy(tgt_pose1).float(), tgt1_world2camera))

                src_init_pcd = torch.from_numpy(src_pcd).float()
                tgt_init_pcd = torch.from_numpy(tgt_pcd).float()

                img_fov_src_depth_pcd1 = unproject((src_depth_image1[0]*1000.0).detach().cpu().numpy(), intrinsics, eye)
                img_fov_src_depth_pcd2= unproject((src_depth_image2[0]*1000.0).detach().cpu().numpy(), intrinsics, eye)

                img_fov_tgt_depth_pcd1 = unproject((tgt_depth_image1[0]*1000.0).detach().cpu().numpy(), intrinsics, eye)
                img_fov_tgt_depth_pcd2 = unproject((tgt_depth_image2[0]*1000.0).detach().cpu().numpy(), intrinsics, eye)
                intrinsics = torch.from_numpy(intrinsics).float()
                projection = Projection(intrinsics)
                src_back_pcd=projection.matrix_multiplication(src1_world2camera,src_init_pcd)
                tgt_back_pcd=projection.matrix_multiplication(tgt1_world2camera,tgt_init_pcd)
                # draw_registration_result(img_fov_src_depth_pcd1,torch.from_numpy(img_fov_src_depth_pcd1),Window_name="rotate back source pcd")
                # draw_registration_result(tgt_back_pcd,torch.from_numpy(img_fov_tgt_depth_pcd1),Window_name="rotate back target pcd")



                src1_inds2d, src1_inds3d = projection.projection(src_init_pcd, src_depth_image1, src1_world2camera)
                src2_inds2d, src2_inds3d = projection.projection(src_init_pcd, src_depth_image2, src2_world2camera)
                tgt1_inds2d, tgt1_inds3d = projection.projection(tgt_init_pcd, tgt_depth_image1, tgt1_world2camera)
                tgt2_inds2d, tgt2_inds3d = projection.projection(tgt_init_pcd, tgt_depth_image2, tgt2_world2camera)

                id_name = f'item_{item}_' + src_scene_id1 + '__src' + src_image_id1 + '__tgt' + tgt_image_id1
                # data['detect_1']=torch.from_numpy(detect_1).long()
                # data['detect_2']=torch.from_numpy(detect_2).long()
                # data['detect_3']=torch.from_numpy(detect_3).long()
                # data['detect_4']=torch.from_numpy(detect_4).long()
                # data['des1']=torch.from_numpy(des1).float()
                # data['des2']=torch.from_numpy(des2).float()
                # data['des3']=torch.from_numpy(des3).float()
                # data['des4']=torch.from_numpy(des4).float()
                data['src_valid_map1'] = torch.from_numpy(src_valid_map1).float()
                data['src_valid_map2'] = torch.from_numpy(src_valid_map2).float()
                data['tgt_valid_map1'] = torch.from_numpy(tgt_valid_map1).float()
                data['tgt_valid_map2'] = torch.from_numpy(tgt_valid_map2).float()
                data['src1_inds2d'] = src1_inds2d.long()
                data['src2_inds2d'] = src2_inds2d.long()
                data['tgt1_inds2d'] = tgt1_inds2d.long()
                data['tgt2_inds2d'] = tgt2_inds2d.long()
                data['src1_inds3d'] = src1_inds3d.long()
                data['src2_inds3d'] = src2_inds3d.long()
                data['tgt1_inds3d'] = tgt1_inds3d.long()
                data['tgt2_inds3d'] = tgt2_inds3d.long()
                data['src_color1'] = src_color_image1
                data['src_color2'] = src_color_image2
                data['tgt_color1'] = tgt_color_image1
                data['tgt_color2'] = tgt_color_image2
                data['id_name'] = id_name
            else:

                src_image_id1 = self.src_image_id1_list[item]
                src_image_id2 = self.src_image_id2_list[item]
                tgt_image_id1 = self.tgt_image_id1_list[item]
                tgt_image_id2 = self.tgt_image_id2_list[item]
                src_scene_id1 = self.src_full_scene_id_list[item]
                src_seq_id1 = self.src_seq_id_list[item]
                tgt_seq_id1 = self.tgt_seq_id_list[item]
                tgt_scene_id1 = self.tgt_full_scene_id_list[item]

                src_image_id3 = str((int(src_image_id1) + int(src_image_id2)) // 2)
                tgt_image_id3 = str((int(tgt_image_id1) + int(tgt_image_id2)) // 2)
                src_color_path1 = os.path.join(self.path, src_scene_id1, src_seq_id1,
                                               "frame-" + src_image_id1.zfill(6) + ".color.png")
                src_color_path2 = os.path.join(self.path, src_scene_id1, src_seq_id1,
                                               "frame-" + src_image_id2.zfill(6) + ".color.png")
                src_color_path3 = os.path.join(self.path, src_scene_id1, src_seq_id1,
                                               "frame-" + src_image_id3.zfill(6) + ".color.png")
                tgt_color_path1 = os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                               "frame-" + tgt_image_id1.zfill(6) + ".color.png")
                tgt_color_path2 = os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                               "frame-" + tgt_image_id2.zfill(6) + ".color.png")
                tgt_color_path3 = os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                               "frame-" + tgt_image_id3.zfill(6) + ".color.png")
                if os.path.isfile(src_color_path1):
                    src_color_image1 = Image.open(src_color_path1)
                    src_color_image2 = Image.open(src_color_path2)
                    src_color_image3 = Image.open(src_color_path3)
                    tgt_color_image1 = Image.open(tgt_color_path1)
                    tgt_color_image2 = Image.open(tgt_color_path2)
                    tgt_color_image3 = Image.open(tgt_color_path3)
                else:
                    src_color_path1 = os.path.join(self.path, src_scene_id1, src_seq_id1,
                                                   "frame-" + src_image_id1.zfill(6) + ".color.jpg")
                    src_color_path2 = os.path.join(self.path, src_scene_id1, src_seq_id1,
                                                   "frame-" + src_image_id2.zfill(6) + ".color.jpg")
                    src_color_path3 = os.path.join(self.path, src_scene_id1, src_seq_id1,
                                                   "frame-" + src_image_id3.zfill(6) + ".color.jpg")
                    tgt_color_path1 = os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                                   "frame-" + tgt_image_id1.zfill(6) + ".color.jpg")
                    tgt_color_path2 = os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                                   "frame-" + tgt_image_id2.zfill(6) + ".color.jpg")
                    tgt_color_path3 = os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                                   "frame-" + tgt_image_id3.zfill(6) + ".color.jpg")
                    src_color_image1 = Image.open(src_color_path1)
                    src_color_image2 = Image.open(src_color_path2)
                    src_color_image3 = Image.open(src_color_path3)
                    tgt_color_image1 = Image.open(tgt_color_path1)
                    tgt_color_image2 = Image.open(tgt_color_path2)
                    tgt_color_image3 = Image.open(tgt_color_path3)
                src_color_image1 = self.transforms["rgb"](src_color_image1)  # center crop, resize, normalize
                src_color_image2 = self.transforms["rgb"](src_color_image2)  # center crop, resize, normalize
                src_color_image3 = self.transforms["rgb"](src_color_image3)  # center crop, resize, normalize
                src_depth_path1 = os.path.join(self.path, src_scene_id1, src_seq_id1,
                                               "frame-" + src_image_id1.zfill(6) + ".depth.png")
                src_depth_path2 = os.path.join(self.path, src_scene_id1, src_seq_id1,
                                               "frame-" + src_image_id2.zfill(6) + ".depth.png")
                src_depth_path3 = os.path.join(self.path, src_scene_id1, src_seq_id1,
                                               "frame-" + src_image_id3.zfill(6) + ".depth.png")
                src_depth_image1 = Image.open(src_depth_path1)  # 640 480
                src_depth_image2 = Image.open(src_depth_path2)  # 640 480
                src_depth_image3 = Image.open(src_depth_path3)  # 640 480
                src_depth_image1 = self.transforms["depth"](src_depth_image1) / 1000.0
                src_depth_image2 = self.transforms["depth"](src_depth_image2) / 1000.0
                src_depth_image3 = self.transforms["depth"](src_depth_image3) / 1000.0

                # tgt image
                tgt_color_image1 = self.transforms["rgb"](tgt_color_image1)  # center crop, resize, normalize
                tgt_color_image2 = self.transforms["rgb"](tgt_color_image2)  # center crop, resize, normalize
                tgt_color_image3 = self.transforms["rgb"](tgt_color_image3)  # center crop, resize, normalize
                tgt_depth_path1 = os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                               "frame-" + tgt_image_id1.zfill(6) + ".depth.png")
                tgt_depth_path2 = os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                               "frame-" + tgt_image_id2.zfill(6) + ".depth.png")
                tgt_depth_path3 = os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                               "frame-" + tgt_image_id3.zfill(6) + ".depth.png")
                tgt_depth_image1 = Image.open(tgt_depth_path1)  # 640 480
                tgt_depth_image2 = Image.open(tgt_depth_path2)  # 640 480
                tgt_depth_image3 = Image.open(tgt_depth_path3)  # 640 480
                tgt_depth_image1 = self.transforms["depth"](tgt_depth_image1) / 1000.0
                tgt_depth_image2 = self.transforms["depth"](tgt_depth_image2) / 1000.0
                tgt_depth_image3 = self.transforms["depth"](tgt_depth_image3) / 1000.0

                intrinsics = np.loadtxt(os.path.join(self.path, src_scene_id1, 'camera-intrinsics.txt'))
                big_size, image_size = [640, 480], [160, 120]
                # tranpose height and width
                intrinsics = adjust_intrinsic(intrinsics, big_size, image_size)

                if intrinsics.shape[0] == 3:
                    res = np.eye(4)
                    res[:3, :3] = intrinsics
                    intrinsics = res
                src_pose1 = np.loadtxt(os.path.join(self.path, src_scene_id1, src_seq_id1,
                                                    "frame-" + src_image_id1.zfill(6) + ".pose.txt"))
                src_pose2 = np.loadtxt(os.path.join(self.path, src_scene_id1, src_seq_id1,
                                                    "frame-" + src_image_id2.zfill(6) + ".pose.txt"))
                src_pose3 = np.loadtxt(os.path.join(self.path, src_scene_id1, src_seq_id1,
                                                    "frame-" + src_image_id3.zfill(6) + ".pose.txt"))
                tgt_pose1 = np.loadtxt(os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                                    "frame-" + tgt_image_id1.zfill(6) + ".pose.txt"))
                tgt_pose2 = np.loadtxt(os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                                    "frame-" + tgt_image_id2.zfill(6) + ".pose.txt"))
                tgt_pose3 = np.loadtxt(os.path.join(self.path, tgt_scene_id1, tgt_seq_id1,
                                                    "frame-" + tgt_image_id3.zfill(6) + ".pose.txt"))

                "get 2d-3d correspondece relationship"

                # save_ply(src_pcd,f'/home/lab507/yjl/frame2/frame{item}_src_aligned_pcd')
                # save_ply(tgt_aligned_pcd,f'/home/lab507/yjl/frame2/frame{item}_tgt_aligned_pcd')
                eye = np.eye(4)

                "back to  image space"
                intrinsics = torch.from_numpy(intrinsics).float()
                projection = Projection(intrinsics)
                if self.data_augmentation:

                    if (aug_src > 0.5):
                        # print("augment rotation to source point cloud")
                        src1_world2camera = np.eye(4)
                        src1_world2camera[:3, :3] = np.linalg.inv(rot_ab)
                        src1_world2camera = torch.from_numpy(src1_world2camera).float()

                        tgt1_world2camera = torch.eye(4).float()
                    else:
                        # print("augment rotation to target point cloud")
                        tgt1_world2camera = np.eye(4)
                        tgt1_world2camera[:3, :3] = np.linalg.inv(rot_ab)
                        tgt1_world2camera = torch.from_numpy(tgt1_world2camera).float()
                        src1_world2camera = torch.eye(4).float()
                else:
                    src1_world2camera = torch.eye(4).float()
                    tgt1_world2camera = torch.eye(4).float()
                src_pose1_rev = np.linalg.inv(src_pose1)
                src_pose2_rev = np.linalg.inv(src_pose2)
                src_pose3_rev = np.linalg.inv(src_pose3)
                tgt_pose1_rev = np.linalg.inv(tgt_pose1)
                tgt_pose2_rev = np.linalg.inv(tgt_pose2)
                tgt_pose3_rev = np.linalg.inv(tgt_pose3)
                src2_world2camera = torch.mm(torch.from_numpy(src_pose2_rev).float(),
                                             torch.mm(torch.from_numpy(src_pose1).float(), src1_world2camera))
                src3_world2camera = torch.mm(torch.from_numpy(src_pose3_rev).float(),
                                             torch.mm(torch.from_numpy(src_pose1).float(), src1_world2camera))
                tgt2_world2camera = torch.mm(torch.from_numpy(tgt_pose2_rev).float(),
                                             torch.mm(torch.from_numpy(tgt_pose1).float(), tgt1_world2camera))
                tgt3_world2camera = torch.mm(torch.from_numpy(tgt_pose3_rev).float(),
                                             torch.mm(torch.from_numpy(tgt_pose1).float(), tgt1_world2camera))

                src_init_pcd = torch.from_numpy(src_pcd).float()
                tgt_init_pcd = torch.from_numpy(tgt_pcd).float()
                #  add aug r aug t

                # src_back_pcd=projection.matrix_multiplication(src1_world2camera,src_init_pcd)
                # tgt_back_pcd=projection.matrix_multiplication(tgt1_world2camera,tgt_init_pcd)
                # draw_registration_result(src_back_pcd,torch.from_numpy(img_fov_src_depth_pcd1),Window_name="rotate back source pcd")
                # draw_registration_result(tgt_back_pcd,torch.from_numpy(img_fov_tgt_depth_pcd1),Window_name="rotate back target pcd")

                # img_fov_src_depth_pcd2=projection.matrix_multiplication(torch.from_numpy(src_pose1_rev),projection.matrix_multiplication(torch.from_numpy(src_pose2),torch.from_numpy(img_fov_src_depth_pcd2)))
                # img_fov_tgt_depth_pcd2=projection.matrix_multiplication(torch.from_numpy(tgt_pose1_rev),projection.matrix_multiplication(torch.from_numpy(tgt_pose2),torch.from_numpy(img_fov_tgt_depth_pcd2)))
                src1_inds2d, src1_inds3d = projection.projection(src_init_pcd, src_depth_image1, src1_world2camera)
                src2_inds2d, src2_inds3d = projection.projection(src_init_pcd, src_depth_image2, src2_world2camera)
                src3_inds2d, src3_inds3d = projection.projection(src_init_pcd, src_depth_image3, src3_world2camera)
                tgt1_inds2d, tgt1_inds3d = projection.projection(tgt_init_pcd, tgt_depth_image1, tgt1_world2camera)
                tgt2_inds2d, tgt2_inds3d = projection.projection(tgt_init_pcd, tgt_depth_image2, tgt2_world2camera)
                tgt3_inds2d, tgt3_inds3d = projection.projection(tgt_init_pcd, tgt_depth_image3, tgt3_world2camera)
                id_name = f'item_{item}_' + src_scene_id1 + '__src' + src_image_id1 + '__tgt' + tgt_image_id1

                data['src1_inds2d'] = src1_inds2d.long()
                data['src2_inds2d'] = src2_inds2d.long()
                data['src3_inds2d'] = src3_inds2d.long()
                data['tgt1_inds2d'] = tgt1_inds2d.long()
                data['tgt2_inds2d'] = tgt2_inds2d.long()
                data['tgt3_inds2d'] = tgt3_inds2d.long()
                data['src1_inds3d'] = src1_inds3d.long()
                data['src2_inds3d'] = src2_inds3d.long()
                data['src3_inds3d'] = src3_inds3d.long()
                data['tgt1_inds3d'] = tgt1_inds3d.long()
                data['tgt2_inds3d'] = tgt2_inds3d.long()
                data['tgt3_inds3d'] = tgt3_inds3d.long()
                data['src_color1'] = src_color_image1
                data['src_color2'] = src_color_image2
                data['src_color3'] = src_color_image3
                data['tgt_color1'] = tgt_color_image1
                data['tgt_color2'] = tgt_color_image2
                data['tgt_color3'] = tgt_color_image3
                data['id_name'] = id_name

        return data
