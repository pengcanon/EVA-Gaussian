from torch.utils.data import Dataset

import numpy as np
import os
from PIL import Image
import cv2
import torch
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from pathlib import Path
import logging
import json
from tqdm import tqdm


def save_np_to_json(parm, save_name):
    for key in parm.keys():
        parm[key] = parm[key].tolist()
    with open(save_name, 'w') as file:
        json.dump(parm, file, indent=1)


def load_json_to_np(parm_name):
    with open(parm_name, 'r') as f:
        parm = json.load(f)
    for key in parm.keys():
        parm[key] = np.array(parm[key])
    return parm


def depth2pts(depth, extrinsic, intrinsic):
    # depth H W extrinsic 3x4 intrinsic 3x3 pts map H W 3
    rot = extrinsic[:3, :3]
    trans = extrinsic[:3, 3:]
    S, S = depth.shape

    y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device),
                          torch.linspace(0.5, S-0.5, S, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # H W 3

    pts_2d[..., 2] = 1.0 / (depth + 1e-8)
    pts_2d[..., 0] -= intrinsic[0, 2]
    pts_2d[..., 1] -= intrinsic[1, 2]
    pts_2d_xy = pts_2d[..., :2] * pts_2d[..., 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[0, 0]
    pts_2d[..., 1] /= intrinsic[1, 1]
    pts_2d = pts_2d.reshape(-1, 3).T
    pts = rot.T @ pts_2d - rot.T @ trans
    return pts.T.view(S, S, 3)


def pts2depth(ptsmap, extrinsic, intrinsic):
    S, S, _ = ptsmap.shape
    pts = ptsmap.view(-1, 3).T
    calib = intrinsic @ extrinsic
    pts = calib[:3, :3] @ pts
    pts = pts + calib[:3, 3:4]
    pts[:2, :] /= (pts[2:, :] + 1e-8)
    depth = 1.0 / (pts[2, :].view(S, S) + 1e-8)
    return depth


def stereo_pts2flow(pts0, pts1, rectify0, rectify1, Tf_x):
    new_extr0, new_intr0, rectify_mat0_x, rectify_mat0_y = rectify0
    new_extr1, new_intr1, rectify_mat1_x, rectify_mat1_y = rectify1
    new_depth0 = pts2depth(torch.FloatTensor(pts0), torch.FloatTensor(new_extr0), torch.FloatTensor(new_intr0))
    new_depth1 = pts2depth(torch.FloatTensor(pts1), torch.FloatTensor(new_extr1), torch.FloatTensor(new_intr1))
    new_depth0 = new_depth0.detach().numpy()
    new_depth1 = new_depth1.detach().numpy()
    new_depth0 = cv2.remap(new_depth0, rectify_mat0_x, rectify_mat0_y, cv2.INTER_LINEAR)
    new_depth1 = cv2.remap(new_depth1, rectify_mat1_x, rectify_mat1_y, cv2.INTER_LINEAR)

    offset0 = new_intr1[0, 2] - new_intr0[0, 2]
    disparity0 = -new_depth0 * Tf_x
    flow0 = offset0 - disparity0

    offset1 = new_intr0[0, 2] - new_intr1[0, 2]
    disparity1 = -new_depth1 * (-Tf_x)
    flow1 = offset1 - disparity1

    flow0[new_depth0 < 0.05] = 0
    flow1[new_depth1 < 0.05] = 0

    return flow0, flow1


def read_img(name):
    img = np.array(Image.open(name))
    return img


def read_depth(name):
    return cv2.imread(name, cv2.IMREAD_UNCHANGED).astype(np.float32) / 2.0 ** 15


class DepthHumanDataset(Dataset):
    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.val_novel_id = opt.val_novel_id
        self.use_processed_data = opt.use_processed_data
        self.phase = phase
        if self.phase == 'train':
            self.data_root = os.path.join(opt.data_root, 'train')
        elif self.phase == 'val':
            self.data_root = os.path.join(opt.data_root, 'val')
        elif self.phase == 'test':
            self.data_root = opt.test_data_root

        self.img_path = os.path.join(self.data_root, 'img/%s/%d.jpg')
        self.img_hr_path = os.path.join(self.data_root, 'img/%s/%d_hr.jpg')
        self.mask_path = os.path.join(self.data_root, 'mask/%s/%d.png')
        self.depth_path = os.path.join(self.data_root, 'depth/%s/%d.png')
        self.intr_path = os.path.join(self.data_root, 'parm/%s/%d_intrinsic.npy')
        self.extr_path = os.path.join(self.data_root, 'parm/%s/%d_extrinsic.npy')
        self.sample_list = sorted(list(os.listdir(os.path.join(self.data_root, 'img'))))

        if self.phase == 'train' and self.opt.anchor:
            with open(os.path.join(opt.data_root, 'landmark.json'), 'r') as file:
                self.landmark = json.load(file)

    def load_single_view(self, sample_name, source_id, hr_img=False, require_mask=True, require_pts=True):
        img_name = self.img_path % (sample_name, source_id)
        image_hr_name = self.img_hr_path % (sample_name, source_id)
        mask_name = self.mask_path % (sample_name, source_id)
        depth_name = self.depth_path % (sample_name, source_id)
        intr_name = self.intr_path % (sample_name, source_id)
        extr_name = self.extr_path % (sample_name, source_id)

        intr, extr = np.load(intr_name), np.load(extr_name)
        mask, pts = None, None
        if hr_img:
            img = read_img(image_hr_name)
            intr[:2] *= 2
        else:
            img = read_img(img_name)
        if require_mask:
            mask = read_img(mask_name)
        depth = read_depth(depth_name)
        pts = depth2pts(torch.FloatTensor(depth), torch.FloatTensor(extr), torch.FloatTensor(intr))

        return img, mask, intr, extr, pts, depth

    def get_novel_view_tensor(self, sample_name, view_id):
        img, mask, intr, extr, _, depth = self.load_single_view(sample_name, view_id, hr_img=self.opt.use_hr_img,
                                                      require_mask=True, require_pts=False)
        width, height = img.shape[:2]
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img / 255.0

        R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(extr[:3, 3], np.float32)

        FovX = focal2fov(intr[0, 0], width)
        FovY = focal2fov(intr[1, 1], height)
        projection_matrix = getProjectionMatrix(znear=self.opt.znear, zfar=self.opt.zfar, K=intr, h=height, w=width).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(self.opt.trans), self.opt.scale)).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        novel_view_data = {
            'scence': sample_name,
            'view_id': torch.IntTensor([view_id]),
            'img': img,
            'mask': mask,
            'extr': torch.FloatTensor(extr),
            'FovX': FovX,
            'FovY': FovY,
            'width': width,
            'height': height,
            'world_view_transform': world_view_transform,
            'full_proj_transform': full_proj_transform,
            'camera_center': camera_center
        }

        return novel_view_data

    def get_origin_data(self, sample_name, source_id_0, source_id_1, require_mask=True):

        img_name = self.img_path % (sample_name, source_id_0)
        mask_name = self.mask_path % (sample_name, source_id_0)
        depth_name = self.depth_path % (sample_name, source_id_0)
        intr_name = self.intr_path % (sample_name, source_id_0)
        extr_name = self.extr_path % (sample_name, source_id_0)
        
        try:
            intr_0, extr_0 = np.load(intr_name), np.load(extr_name)
        except:
            intr_0, extr_0 = camera["intr0"], camera["extr0"]

        mask_0, pts_0 = None, None
        left_0, right_0, face_0 = torch.zeros(21,2),  torch.zeros(21,2),  torch.zeros(19,3)
        img_0 = read_img(img_name)
        if require_mask:
            mask_0 = read_img(mask_name)
        depth_0 = read_depth(depth_name)
        if self.phase == 'train' and self.opt.anchor:
            if 'Left' in self.landmark[sample_name][f"{source_id_0}.jpg"]:
                left_0=[]
                for i in self.landmark[sample_name][f"{source_id_0}.jpg"]['Left'].keys():
                    left_0.append([self.landmark[sample_name][f"{source_id_0}.jpg"]['Left'][i]["x"], self.landmark[sample_name][f"{source_id_0}.jpg"]['Left'][i]["y"]])
                left_0 = torch.tensor(left_0)
            if 'Right' in self.landmark[sample_name][f"{source_id_0}.jpg"]:
                right_0=[]
                for i in self.landmark[sample_name][f"{source_id_0}.jpg"]['Right'].keys():
                    right_0.append([self.landmark[sample_name][f"{source_id_0}.jpg"]['Right'][i]["x"], self.landmark[sample_name][f"{source_id_0}.jpg"]['Right'][i]["y"]])
                right_0 = torch.tensor(right_0)
            if self.landmark[sample_name][f"{source_id_0}.jpg"]['face']:
                face_0 = torch.tensor(self.landmark[sample_name][f"{source_id_0}.jpg"]['face'])

        img_name = self.img_path % (sample_name, source_id_1)
        mask_name = self.mask_path % (sample_name, source_id_1)
        depth_name = self.depth_path % (sample_name, source_id_1)
        intr_name = self.intr_path % (sample_name, source_id_1)
        extr_name = self.extr_path % (sample_name, source_id_1)

        try:
            intr_1, extr_1 = np.load(intr_name), np.load(extr_name)
        except:
            intr_1, extr_1 = camera["intr1"], camera["extr1"]

        mask_1, pts_1 = None, None
        left_1, right_1, face_1 = torch.zeros(21,2),  torch.zeros(21,2),  torch.zeros(19,3)
        img_1 = read_img(img_name)
        if require_mask:
            mask_1 = read_img(mask_name)
        depth_1 = read_depth(depth_name)
        if self.phase == 'train' and self.opt.anchor:
            if 'Left' in self.landmark[sample_name][f"{source_id_1}.jpg"]:
                left_1=[]
                for i in self.landmark[sample_name][f"{source_id_1}.jpg"]['Left'].keys():
                    left_1.append([self.landmark[sample_name][f"{source_id_1}.jpg"]['Left'][i]["x"], self.landmark[sample_name][f"{source_id_1}.jpg"]['Left'][i]["y"]])
                left_1 = torch.tensor(left_1)
            if 'Right' in self.landmark[sample_name][f"{source_id_1}.jpg"]:
                right_1=[]
                for i in self.landmark[sample_name][f"{source_id_1}.jpg"]['Right'].keys():
                    right_1.append([self.landmark[sample_name][f"{source_id_1}.jpg"]['Right'][i]["x"], self.landmark[sample_name][f"{source_id_1}.jpg"]['Right'][i]["y"]])
                right_1 = torch.tensor(right_1)
            if self.landmark[sample_name][f"{source_id_1}.jpg"]['face']:
                face_1 = torch.tensor(self.landmark[sample_name][f"{source_id_1}.jpg"]['face'])

        camera = {
            'intr0': intr_0,
            'intr1': intr_1,
            'extr0': extr_0,
            'extr1': extr_1,
        }

        stereo_data = {
            'img0': img_0,
            'mask0': mask_0,
            'depth0': depth_0,
            'left0': left_0,
            'right0': right_0,
            'face0': face_0,
            'img1': img_1,
            'mask1': mask_1,
            'camera': camera,
            'depth1': depth_1,
            'left1': left_1,
            'right1': right_1,
            'face1': face_1,
        }

        return stereo_data

    def stereo_to_dict_tensor(self, stereo_data, subject_name):
        img_tensor, mask_tensor = [], []
        for (img_view, mask_view) in [('img0', 'mask0'), ('img1', 'mask1')]:
            img = torch.from_numpy(stereo_data[img_view]).permute(2, 0, 1)
            img = 2 * (img / 255.0) - 1.0
            mask = torch.from_numpy(stereo_data[mask_view]).permute(2, 0, 1).float()
            mask = mask / 255.0

            img = img * mask
            mask[mask < 0.5] = 0.0
            mask[mask >= 0.5] = 1.0
            img_tensor.append(img)
            mask_tensor.append(mask)

        lmain_data = {
            'img': img_tensor[0],
            'mask': mask_tensor[0],
            'intr': torch.FloatTensor(stereo_data['camera']['intr0']),
            'ref_intr': torch.FloatTensor(stereo_data['camera']['intr1']),
            'extr': torch.FloatTensor(stereo_data['camera']['extr0']),
            'left': stereo_data['left0'],
            'right': stereo_data['right0'],
            'face': stereo_data['face0'],
        }

        rmain_data = {
            'img': img_tensor[1],
            'mask': mask_tensor[1],
            'intr': torch.FloatTensor(stereo_data['camera']['intr1']),
            'ref_intr': torch.FloatTensor(stereo_data['camera']['intr0']),
            'extr': torch.FloatTensor(stereo_data['camera']['extr1']),
            'left': stereo_data['left1'],
            'right': stereo_data['right1'],
            'face': stereo_data['face1'],
        }

        if 'flow0' in stereo_data:
            flow_tensor, valid_tensor = [], []
            for (flow_view, valid_view) in [('flow0', 'valid0'), ('flow1', 'valid1')]:
                flow = torch.from_numpy(stereo_data[flow_view])
                flow = torch.unsqueeze(flow, dim=0)
                flow_tensor.append(flow)

                valid = torch.from_numpy(stereo_data[valid_view])
                valid = torch.unsqueeze(valid, dim=0)
                valid = valid / 255.0
                valid_tensor.append(valid)

            lmain_data['flow'], lmain_data['valid'] = flow_tensor[0], valid_tensor[0]
            rmain_data['flow'], rmain_data['valid'] = flow_tensor[1], valid_tensor[1]

        if 'depth0' in stereo_data:
            depth_tensor = []
            for depth_view in ['depth0', 'depth1']:
                depth = torch.from_numpy(stereo_data[depth_view])
                depth = torch.unsqueeze(depth, dim=0)
                depth_tensor.append(depth)

            lmain_data['depth'] = depth_tensor[0]
            rmain_data['depth'] = depth_tensor[1]

        return {'name': subject_name, 'lmain': lmain_data, 'rmain': rmain_data}

    def get_item(self, index, novel_id=None):
        sample_id = index % len(self.sample_list)
        sample_name = self.sample_list[sample_id]

        stereo_np = self.get_origin_data(sample_name, self.opt.source_id[0], self.opt.source_id[1],
                                               require_mask=True)
        dict_tensor = self.stereo_to_dict_tensor(stereo_np, sample_name)

        if novel_id:
            novel_id = np.random.choice(novel_id)
            dict_tensor.update({
                'novel_view': self.get_novel_view_tensor(sample_name, novel_id)
            })

        return dict_tensor

    def get_test_item(self, index, source_id):
        sample_id = index % len(self.sample_list)
        sample_name = self.sample_list[sample_id]

        if self.use_processed_data:
            logging.error('test data loader not support processed data')

        view0_data = self.load_single_view(sample_name, source_id[0], hr_img=False, require_mask=True, require_pts=False)
        view1_data = self.load_single_view(sample_name, source_id[1], hr_img=False, require_mask=True, require_pts=False)
        lmain_intr_ori, lmain_extr_ori = view0_data[2], view0_data[3]
        rmain_intr_ori, rmain_extr_ori = view1_data[2], view1_data[3]
        stereo_np = self.get_rectified_stereo_data(main_view_data=view0_data, ref_view_data=view1_data)
        dict_tensor = self.stereo_to_dict_tensor(stereo_np, sample_name)

        dict_tensor['lmain']['intr_ori'] = torch.FloatTensor(lmain_intr_ori)
        dict_tensor['rmain']['intr_ori'] = torch.FloatTensor(rmain_intr_ori)
        dict_tensor['lmain']['extr_ori'] = torch.FloatTensor(lmain_extr_ori)
        dict_tensor['rmain']['extr_ori'] = torch.FloatTensor(rmain_extr_ori)

        img_len = 2048 if self.opt.use_hr_img else 1024
        novel_dict = {
            'height': torch.IntTensor([img_len]),
            'width': torch.IntTensor([img_len])
        }

        dict_tensor.update({
            'novel_view': novel_dict
        })

        return dict_tensor

    def __getitem__(self, index):
        if self.phase == 'train':
            return self.get_item(index, novel_id=self.opt.train_novel_id)
        elif self.phase == 'val':
            return self.get_item(index, novel_id=self.val_novel_id)

    def __len__(self):
        self.train_boost = 50
        self.val_boost = 200
        if self.phase == 'train':
            return len(self.sample_list) * self.train_boost
        elif self.phase == 'val':
            return len(self.sample_list) * self.val_boost
        else:
            return len(self.sample_list)
