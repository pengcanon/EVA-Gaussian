
import torch
from gaussian_renderer import render_f

from einops import rearrange
import time

def pts2render_feature(data, bg_color):
    bs = data['lmain']['img'].shape[0]
    render_novel_list = []
    render_feat_list = []
    for i in range(bs):
        xyz_i_valid = []
        rgb_i_valid = []
        rot_i_valid = []
        scale_i_valid = []
        opacity_i_valid = []
        feature_i_valid = []
        covariances_i_valid = []
        for view in ['lmain', 'rmain']:
            valid_i = data[view]['pts_valid'][i, :]
            xyz_i = data[view]['xyz'][i, :, :]
            rgb_i = data[view]['img'][i, :, :, :].permute(1, 2, 0).view(-1, 3)
            rot_i = data[view]['rot_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 4)
            scale_i = data[view]['scale_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 3)
            opacity_i = data[view]['opacity_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 1)
            if data[view]['features'] is not None:
                feature_i = data[view]['features'][i, :, :, :].permute(1, 2, 0).view(-1, data[view]['features'].shape[1])
                feature_i_valid.append(feature_i[valid_i].view(-1, data[view]['features'].shape[1]))

            xyz_i_valid.append(xyz_i[valid_i].view(-1, 3))
            rgb_i_valid.append(rgb_i[valid_i].view(-1, 3))
            rot_i_valid.append(rot_i[valid_i].view(-1, 4))
            scale_i_valid.append(scale_i[valid_i].view(-1, 3))
            opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))

        pts_xyz_i = torch.concat(xyz_i_valid, dim=0)
        pts_rgb_i = torch.concat(rgb_i_valid, dim=0)
        pts_rgb_i = pts_rgb_i * 0.5 + 0.5
        rot_i = torch.concat(rot_i_valid, dim=0)
        scale_i = torch.concat(scale_i_valid, dim=0)
        opacity_i = torch.concat(opacity_i_valid, dim=0)
        if data[view]['features'] is not None:
            feature_i = torch.concat(feature_i_valid, dim=0)

        render_novel_i, features = render_f(data, i, pts_xyz_i, pts_rgb_i, rot_i, scale_i, opacity_i, bg_color=bg_color, pts_features=feature_i if data[view]['features'] is not None else None)
        render_feat_list.append(features.unsqueeze(0))
        render_novel_list.append(render_novel_i.unsqueeze(0))

    data['novel_view']['img_pred'] = torch.concat(render_novel_list, dim=0)
    features = torch.concat(render_feat_list, dim=0)
    return data, features

