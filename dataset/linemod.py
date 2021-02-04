from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image
from utils.transforms import rotate_img
from utils.utils import load_ply
import glob, pickle
import torch_geometric.transforms as T
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt


linemod_K = np.array(
    [[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]]
)

blender_K = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])

linemod_cls_names = [
    "ape",
    "cam",
    "cat",
    "duck",
    "glue",
    "iron",
    "phone",
    "benchvise",
    "can",
    "driller",
    "eggbox",
    "holepuncher",
    "lamp",
]


class LineModDataset(Dataset):
    def __init__(self, root, cls, is_train=True, occ=False):

        super(LineModDataset, self).__init__()
        self.root = root
        self.cls = cls
        self.data_paths = []
        self.is_training = is_train
        if is_train:
            self.data_paths = self.get_train_data_path(self.root, self.cls)

        else:
            if occ:
                self.data_paths = self.get_occ_data_path(self.root, self.cls)
            else:
                self.data_paths = self.get_test_data_path(self.root, self.cls)

        self.mesh_model = load_ply(
            osp.join(root, "linemod", cls, "{}_new.ply".format(cls))
        )

    def get_occ_data_path(self, root, cls):

        linemod_root = osp.join(root, "linemod")
        occ_root = osp.join(root, "occlusion_linemod")
        paths_list = []
        paths = {}
        occ_inds = np.loadtxt(osp.join(linemod_root, cls, "test_occlusion.txt"), np.str)
        occ_inds = [int(osp.basename(ind).replace(".jpg", "")) for ind in occ_inds]

        img_path = osp.join(occ_root, "RGB-D/rgb_noseg")
        pose_path = osp.join(occ_root, "blender_poses", cls)
        mask_path = osp.join(occ_root, "masks", cls)

        for idx in occ_inds:
            img_name = "color_{:05d}.png".format(idx)
            mask_name = "{}.png".format(idx)
            pose_name = "pose{}.npy".format(idx)
            paths["img_path"] = osp.join(img_path, img_name)
            paths["mask_path"] = osp.join(mask_path, mask_name)
            paths["pose_path"] = osp.join(pose_path, pose_name)
            paths["type"] = "occ"
            paths_list.append(paths.copy())

        return paths_list

    def get_test_data_path(self, root, cls):

        root = osp.join(root, "linemod")

        paths_list = []
        paths = {}
        train_inds = np.loadtxt(osp.join(root, cls, "test.txt"), np.str)
        train_inds = [int(osp.basename(ind).replace(".jpg", "")) for ind in train_inds]

        img_path = osp.join(root, cls, "JPEGImages")
        pose_path = osp.join(root, cls, "pose")
        mask_path = osp.join(root, cls, "mask")

        for idx in train_inds:
            img_name = "{:06}.jpg".format(idx)
            mask_name = "{:04}.png".format(idx)
            pose_name = "pose{}.npy".format(idx)
            paths["img_path"] = osp.join(img_path, img_name)
            paths["mask_path"] = osp.join(mask_path, mask_name)
            paths["pose_path"] = osp.join(pose_path, pose_name)
            paths["type"] = "true"
            paths_list.append(paths.copy())

        return paths_list

    def get_train_data_path(self, root, cls):

        paths_list = []
        paths = {}
        root = osp.join(root, "linemod")
        render_dir = osp.join(root, "renders", cls)
        fuse_dir = osp.join(root, "fuse")
        fuse_num = len(glob.glob(osp.join(fuse_dir, "*.pkl")))
        cls_idx = linemod_cls_names.index(cls)
        render_num = len(glob.glob(osp.join(render_dir, "*.pkl")))
        train_inds = np.loadtxt(osp.join(root, cls, "train.txt"), np.str)
        train_inds = [int(osp.basename(ind).replace(".jpg", "")) for ind in train_inds]

        train_img_path = osp.join(root, cls, "JPEGImages")
        pose_path = osp.join(root, cls, "pose")
        mask_path = osp.join(root, cls, "mask")

        for ind in range(fuse_num):

            img_name = "{}_rgb.jpg".format(ind)
            rgb_path = osp.join(fuse_dir, img_name)
            pose_dir = osp.join(fuse_dir, "{}_info.pkl".format(ind))
            mask_dir = osp.join(fuse_dir, "{}_mask.png".format(ind))
            mask = (np.asarray(Image.open(mask_dir)) == cls_idx).astype(np.uint8)

            if np.sum(mask) < 400:
                continue

            paths["img_path"] = rgb_path
            paths["mask_path"] = mask_dir
            paths["pose_path"] = pose_dir
            paths["type"] = "fuse"
            paths_list.append(paths.copy())

        for idx in train_inds:
            img_name = "{:06}.jpg".format(idx)
            mask_name = "{:04}.png".format(idx)
            pose_name = "pose{}.npy".format(idx)
            paths["img_path"] = osp.join(train_img_path, img_name)
            paths["mask_path"] = osp.join(mask_path, mask_name)
            paths["pose_path"] = osp.join(pose_path, pose_name)
            paths["type"] = "true"
            paths_list.append(paths.copy())

        for idx in range(render_num):
            img_name = "{}.jpg".format(idx)
            mask_name = "{}_depth.png".format(idx)
            pose_name = "{}_RT.pkl".format(idx)

            paths["img_path"] = osp.join(render_dir, img_name)
            paths["mask_path"] = osp.join(render_dir, mask_name)
            paths["pose_path"] = osp.join(render_dir, pose_name)
            paths["type"] = "render"
            paths_list.append(paths.copy())

        return paths_list

    def get_data(self, path):

        img = np.array(Image.open(path["img_path"]))

        if path["type"] == "true":
            pose = np.load(path["pose_path"])
            K = linemod_K
            mask = (np.asarray(Image.open(path["mask_path"]))[:, :, 0] != 0).astype(
                np.uint8
            )

        elif path["type"] == "occ":
            pose = np.load(path["pose_path"])
            K = linemod_K
            mask = (np.asarray(Image.open(path["mask_path"]))).astype(np.uint8)

        elif path["type"] == "render":
            with open(path["pose_path"], "rb") as f:
                pose = pickle.load(f)["RT"]

            K = blender_K
            mask = (np.asarray(Image.open(path["mask_path"]))).astype(np.uint8)

        else:
            with open(path["pose_path"], "rb") as f:
                begins, poses = pickle.load(f)

            cls_idx = linemod_cls_names.index(self.cls)
            pose = poses[cls_idx]
            K = linemod_K.copy()
            K[0, 2] += begins[cls_idx, 1]
            K[1, 2] += begins[cls_idx, 0]
            mask = (np.asarray(Image.open(path["mask_path"])) == cls_idx + 1).astype(
                np.uint8
            )

        return img, mask, pose.astype(np.float32), K

    def augment(self, img, mask, pose, K):

        img = np.asarray(img).astype(np.uint8)
        foreground = np.sum(mask)
        if foreground > 400:
            # randomly mask out to add occlusion
            R = np.eye(3, dtype=np.float32)
            R_orig = pose[:3, :3]
            T_orig = pose[:3, 3]

            img, mask, R = rotate_img(img, mask, T_orig, K, -30, 30)

            new_R = np.dot(R, R_orig)
            pose[:3, :3] = new_R

        return img, mask, pose

    def __getitem__(self, index):

        path = self.data_paths[index]

        img, mask, pose, K = self.get_data(path)

        if self.is_training:
            img, mask, pose = self.augment(img, mask, pose, K)

        img = img / 255.0
        img -= [0.485, 0.456, 0.406]
        img /= [0.229, 0.224, 0.225]
        img = torch.tensor(img, dtype=torch.float32).permute((2, 0, 1))
        mask = np.asarray(mask).astype(np.uint8)

        return img, mask, pose, K

    def __len__(self):
        return len(self.data_paths)
