import os.path as osp
import torch
import tqdm
from utils.utils import load_ply, project,mesh_project
import numpy as np
import cv2
from scipy import spatial


cuda = torch.cuda.is_available()

diameters = {
    "cat": 15.2633,
    "ape": 9.74298,
    "benchvise": 28.6908,
    "bowl": 17.1185,
    "cam": 17.1593,
    "can": 19.3416,
    "cup": 12.5961,
    "driller": 25.9425,
    "duck": 10.7131,
    "eggbox": 17.6364,
    "glue": 16.4857,
    "holepuncher": 14.8204,
    "iron": 30.3153,
    "lamp": 28.5155,
    "phone": 20.8394,
}


class evaluator:
    def __init__(self, args, model, test_loader, device):

        self.args = args
        self.mesh_model = load_ply(
            osp.join(
                args.data_path,
                "linemod",
                args.class_type,
                "{}_new.ply".format(args.class_type),
            )
        )

        self.pts_3d = self.mesh_model["pts"] * 1000
        self.device = device
        self.model = model
        self.proj_2d = []
        self.add = []
        self.diameter = diameters[args.class_type] / 100.0
        self.data_loader = test_loader

    def evaluate(self):

        self.model.eval()
        print("model class type:{}".format(self.args.class_type))

        with torch.no_grad():
            for data in tqdm.tqdm(self.data_loader, leave=False, desc="val"):
                if cuda:
                    img, mask, pose, K = [x.to(self.device) for x in data]
                else:
                    img, mask, pose, K = data

                S, seg = self.model(img)

                self.calculate_projection2d_add(
                    S, seg, mask, pose, K, self.args.class_type
                )

        proj2d = np.mean(self.proj_2d)
        add = np.mean(self.add)

        print("2d projections metric: {}".format(proj2d))
        print("ADD metric: {}".format(add))

    def calculate_projection2d_add(self, S, seg, mask, pose, K, cls):

        mask_pred = torch.argmax(seg, dim=1).detach().cpu().numpy().astype(np.uint8)
        mask_gt = mask.detach().cpu().numpy()

        S = S.permute(1, 0).detach().cpu().numpy()

        pose = pose.detach().cpu().numpy()
        K = K.detach().cpu().numpy()

        valid_mask = np.where(mask_pred)
        batch = valid_mask[0]
        v = valid_mask[1]
        u = valid_mask[2]

        S_key_points = S

        batch_size = mask_pred.shape[0]
        match_loss_ = []
        total_error = []
        total_iou = []
        org_pose = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=-1)
        for i in range(batch_size):
            K_s = K[i]
            gt_pose = pose[i]
            mesh_pts_2d = mesh_project(self.mesh_model["pts"], K_s, gt_pose)
            # iou = (mask_pred[i] & mask_gt[i]).sum() / (mask_pred[i] | mask_gt[i]).sum()
            gd_v = v[batch == i]
            gd_u = u[batch == i]
            S_mat = S_key_points[:, batch == i]
            if S_mat.size is 0:
                self.proj_2d.append(False)
                self.add.append(False)
                continue
            point_idx = np.argmax(S_mat, axis=0)
            pts_2d = np.array([gd_u, gd_v], dtype=np.int32).transpose((1, 0))

            if pts_2d.shape[0] <= 5:
                self.proj_2d.append(False)
                self.add.append(False)
                continue

            pts_3d = self.pts_3d[point_idx, :]
            pred_pose = self.pnp(pts_3d, pts_2d, K_s)

            self.projection_2d(pred_pose, gt_pose, K_s)
            if cls in ["eggbox", "glue"]:
                self.add_metric(pred_pose, gt_pose, syn=True)
            else:
                self.add_metric(pred_pose, gt_pose)

    def pnp(self, points_3d, points_2d, camera_matrix, method=cv2.SOLVEPNP_ITERATIVE):

        try:
            dist_coeffs = pnp.dist_coeffs
        except:
            dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")

        assert (
            points_3d.shape[0] == points_2d.shape[0]
        ), "points 3D and points 2D must have same number of vertices"
        if method == cv2.SOLVEPNP_EPNP:
            points_3d = np.expand_dims(points_3d, 0)
            points_2d = np.expand_dims(points_2d, 0)

        points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
        points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
        camera_matrix = camera_matrix.astype(np.float64)

        if points_2d.shape[0] < 30:
            _, R_exp, t = cv2.solvePnP(
                points_3d, points_2d, camera_matrix, dist_coeffs, flags=method
            )
        else:

            _, R_exp, t, inliers = cv2.solvePnPRansac(
                points_3d, points_2d, camera_matrix, dist_coeffs, reprojectionError=1.2
            )

        R, _ = cv2.Rodrigues(R_exp)

        return np.concatenate([R, t / 1000], axis=-1)

    def projection_2d(self, pose_pred, pose_targets, K, threshold=5):
        model_2d_pred = project(self.mesh_model["pts"], K, pose_pred)
        model_2d_targets = project(self.mesh_model["pts"], K, pose_targets)
        proj_mean_diff = np.mean(
            np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1)
        )

        self.proj_2d.append(proj_mean_diff < threshold)

    def add_metric(self, pose_pred, pose_targets, syn=False, percentage=0.1):
        diameter = self.diameter * percentage
        model_pred = (
            np.dot(self.mesh_model["pts"], pose_pred[:, :3].T) + pose_pred[:, 3]
        )
        model_targets = (
            np.dot(self.mesh_model["pts"], pose_targets[:, :3].T) + pose_targets[:, 3]
        )

        if syn:
            mean_dist_index = spatial.cKDTree(model_pred)
            mean_dist, _ = mean_dist_index.query(model_targets, k=1)
            mean_dist = np.mean(mean_dist)
        else:
            mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))

        self.add.append(mean_dist < diameter)

    def cm_degree_5_metric(self, pose_pred, pose_targets, icp=False):
        translation_distance = (
            np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 100
        )
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.0) / 2.0))
        if icp:
            self.icp_cmd5.append(translation_distance < 5 and angular_distance < 5)
        else:
            self.cmd5.append(translation_distance < 5 and angular_distance < 5)
