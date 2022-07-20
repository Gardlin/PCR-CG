import os
import torch

class Projection(object):
    def __init__(self, intrinsic_matrix=0, thresh=0.1):
        """
        intrinsic_matrix is a 4x4 matrix, torch.FloatTensor

        """
        self.intrinsics = intrinsic_matrix
        self.thresh = thresh

    @staticmethod
    def matrix_multiplication(matrix, points):
        """
        matrix: 4x4, torch.FloatTensor
        points: nx3, torch.FloatTensor
        reutrn: nx3, torch.FloatTensor
        """
        device = torch.device("cuda" if matrix.get_device() != -1 else "cpu")
        points = torch.cat([points.t(), torch.ones((1, points.shape[0]), device=device)])
        if matrix.shape[0] ==3:
            mat=torch.eye(4).to(device)
            mat[:3,:3]=matrix
            matrix=mat

        return torch.mm(matrix, points).t()[:, :3]



    def projection(self, points, depth_map, world2camera):
        """
        points: nx3 point cloud xyz in world space, torch.FloatTensor
        depth_map: height x width, torch.FloatTensor
        world2camera: 4x4 matrix, torch.FloatTensor

        return: mapping of 2d pixel coordinates to 3d point cloud indices
            inds2d: n x 2 array, torch.LongTensor (notice the order xy == width, height)
            inds3d: n x 1 array, index of point cloud
        """
        depth_map=depth_map.squeeze(0)
        height = depth_map.size(0)
        width = depth_map.size(1)

        xyz_in_camera_space = Projection.matrix_multiplication(world2camera, points)
        xyz_in_image_space = Projection.matrix_multiplication(self.intrinsics, xyz_in_camera_space)

        projected_depth = xyz_in_image_space[:,2]
        xy_in_image_space = (xyz_in_image_space[:,:2] / projected_depth.repeat(2,1).T[:,:]).long()

        mask_height = (xy_in_image_space[:,1] >= 0) & (xy_in_image_space[:,1] < height)
        mask_width = (xy_in_image_space[:,0] >= 0) & (xy_in_image_space[:,0] < width)
        mask_spatial = mask_height & mask_width
        depth = depth_map[xy_in_image_space[mask_spatial,1], 
                          xy_in_image_space[mask_spatial,0]]
        mask_depth = torch.abs(projected_depth[mask_spatial] - depth) < self.thresh

        inds2d = xy_in_image_space[mask_spatial][mask_depth]
        inds3d = torch.arange(points.size(0))[mask_spatial][mask_depth]

        return inds2d, inds3d
    def get_mask(self, img_fov_points, img_foc_depth_pcd,depth_map):
        """
        points: nx3 point cloud xyz in world space, torch.FloatTensor
        depth_map: height x width, torch.FloatTensor
        world2camera: 4x4 matrix, torch.FloatTensor

        return: mapping of 2d pixel coordinates to 3d point cloud indices
            inds2d: n x 2 array, torch.LongTensor (notice the order xy == width, height)
            inds3d: n x 1 array, index of point cloud
        """
        depth_map=depth_map.squeeze(0)
        height = depth_map.size(0)
        width = depth_map.size(1)

        xyz_in_camera_space = img_fov_points
        xyz_in_image_space = Projection.matrix_multiplication(self.intrinsics, xyz_in_camera_space)

        projected_depth = xyz_in_image_space[:,2]
        xy_in_image_space = (xyz_in_image_space[:,:2] / projected_depth.repeat(2,1).T[:,:]).long()

        mask_height = (xy_in_image_space[:,1] >= 0) & (xy_in_image_space[:,1] < height)
        mask_width = (xy_in_image_space[:,0] >= 0) & (xy_in_image_space[:,0] < width)
        mask_spatial = mask_height & mask_width
        depth = depth_map[xy_in_image_space[mask_spatial,1],
                          xy_in_image_space[mask_spatial,0]]
        mask_depth = torch.abs(projected_depth[mask_spatial] - depth) < self.thresh

        inds2d = xy_in_image_space[mask_spatial][mask_depth]
        inds3d = torch.arange(img_fov_points.size(0))[mask_spatial][mask_depth]

        return inds2d, inds3d

if __name__ == "__main__":
    intrinsic_matrix = torch.zeros((4,4))
    depth_map = torch.zeros((240, 320))
    world2camera = torch.zeros((4,4))

    projection = Projection(intrinsic_matrix)
    inds2d, inds3d = projection.projection(points, depth_map, world2camera)






