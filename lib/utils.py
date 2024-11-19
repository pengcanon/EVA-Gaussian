
import torch
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
import matplotlib.pyplot as plt
import numpy as np


def get_novel_calib(data, opt, ratio=0.5, intr_key='intr', extr_key='extr'):
    bs = data['lmain'][intr_key].shape[0]
    fovx_list, fovy_list, world_view_transform_list, full_proj_transform_list, camera_center_list = [], [], [], [], []
    for i in range(bs):
        intr0 = data['lmain'][intr_key][i, ...].cpu().numpy()
        intr1 = data['rmain'][intr_key][i, ...].cpu().numpy()
        extr0 = data['lmain'][extr_key][i, ...].cpu().numpy()
        extr1 = data['rmain'][extr_key][i, ...].cpu().numpy()

        rot0 = extr0[:3, :3]
        rot1 = extr1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot0, rot1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        npose = np.diag([1.0, 1.0, 1.0, 1.0])
        npose = npose.astype(np.float32)
        npose[:3, :3] = rot.as_matrix()
        npose[:3, 3] = ((1.0 - ratio) * extr0 + ratio * extr1)[:3, 3]
        extr_new = npose[:3, :]
        intr_new = ((1.0 - ratio) * intr0 + ratio * intr1)

        if opt.use_hr_img:
            intr_new[:2] *= 2
        width, height = data['novel_view']['width'][i], data['novel_view']['height'][i]
        R = np.array(extr_new[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(extr_new[:3, 3], np.float32)

        FovX = focal2fov(intr_new[0, 0], width)
        FovY = focal2fov(intr_new[1, 1], height)
        projection_matrix = getProjectionMatrix(znear=opt.znear, zfar=opt.zfar, K=intr_new, h=height, w=width).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(opt.trans), opt.scale)).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        fovx_list.append(FovX)
        fovy_list.append(FovY)
        world_view_transform_list.append(world_view_transform.unsqueeze(0))
        full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
        camera_center_list.append(camera_center.unsqueeze(0))

    data['novel_view']['FovX'] = torch.FloatTensor(np.array(fovx_list)).cuda()
    data['novel_view']['FovY'] = torch.FloatTensor(np.array(fovy_list)).cuda()
    data['novel_view']['world_view_transform'] = torch.concat(world_view_transform_list).cuda()
    data['novel_view']['full_proj_transform'] = torch.concat(full_proj_transform_list).cuda()
    data['novel_view']['camera_center'] = torch.concat(camera_center_list).cuda()
    return data


def depth2pc(depth, extrinsic, intrinsic):
    B, C, S, S = depth.shape
    depth = depth[:, 0, :, :]
    rot = extrinsic[:, :3, :3]
    trans = extrinsic[:, :3, 3:]

    y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device), torch.linspace(0.5, S-0.5, S, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # B S S 3

    pts_2d[..., 2] = 1.0 / (depth + 1e-8)
    pts_2d[:, :, :, 0] -= intrinsic[:, None, None, 0, 2]
    pts_2d[:, :, :, 1] -= intrinsic[:, None, None, 1, 2]
    pts_2d_xy = pts_2d[:, :, :, :2] * pts_2d[:, :, :, 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[:, 0, 0][:, None, None]
    pts_2d[..., 1] /= intrinsic[:, 1, 1][:, None, None]

    pts_2d = pts_2d.view(B, -1, 3).permute(0, 2, 1)
    rot_t = rot.permute(0, 2, 1)
    pts = torch.bmm(rot_t, pts_2d) - torch.bmm(rot_t, trans)

    return pts.permute(0, 2, 1)


def absolutedepth2pc(depth, extrinsic, intrinsic):
    B, C, S, S = depth.shape
    depth = depth[:, 0, :, :]
    rot = extrinsic[:, :3, :3]
    trans = extrinsic[:, :3, 3:]

    y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device), torch.linspace(0.5, S-0.5, S, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # B S S 3

    pts_2d[..., 2] = depth
    pts_2d[:, :, :, 0] -= intrinsic[:, None, None, 0, 2]
    pts_2d[:, :, :, 1] -= intrinsic[:, None, None, 1, 2]
    pts_2d_xy = pts_2d[:, :, :, :2] * pts_2d[:, :, :, 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[:, 0, 0][:, None, None]
    pts_2d[..., 1] /= intrinsic[:, 1, 1][:, None, None]

    pts_2d = pts_2d.view(B, -1, 3).permute(0, 2, 1)
    rot_t = rot.permute(0, 2, 1)
    pts = torch.bmm(rot_t, pts_2d) - torch.bmm(rot_t, trans)

    return pts.permute(0, 2, 1)

def add_gaussian_noise(data):
    for view in ['lmain', 'rmain']:
        b,c,h,w = data[view]['depth_pred'].shape
        gaussian_noise = 0.002*torch.randn(b, c, h, w).to(data[view]['depth_pred'])

        data[view]['depth_pred'] = data[view]['depth_pred']+gaussian_noise
    
    return data

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]

C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * x * sh[..., 1] +
                C1 * y * sh[..., 2] -
                C1 * z * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xz * sh[..., 4] +
                    C2[1] * xy * sh[..., 5] +
                    C2[2] * (2.0 * yy - zz - xx) * sh[..., 6] +
                    C2[3] * yz * sh[..., 7] +
                    C2[4] * (zz - xx) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * x * (3 * zz - xx) * sh[..., 9] +
                C3[1] * xz * y * sh[..., 10] +
                C3[2] * x * (4 * yy - zz - xx)* sh[..., 11] +
                C3[3] * y * (2 * yy - 3 * zz - 3 * xx) * sh[..., 12] +
                C3[4] * z * (4 * yy - zz - xx) * sh[..., 13] +
                C3[5] * z * (zz - xx) * sh[..., 14] +
                C3[6] * z * (zz - 3 * xx) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xz * (zz - xx) * sh[..., 16] +
                            C4[1] * xy * (3 * zz - xx) * sh[..., 17] +
                            C4[2] * xz * (7 * yy - 1) * sh[..., 18] +
                            C4[3] * xy * (7 * yy - 3) * sh[..., 19] +
                            C4[4] * (yy * (35 * yy - 30) + 3) * sh[..., 20] +
                            C4[5] * yz * (7 * yy - 3) * sh[..., 21] +
                            C4[6] * (zz - xx) * (7 * yy - 1) * sh[..., 22] +
                            C4[7] * yz * (zz - 3 * xx) * sh[..., 23] +
                            C4[8] * (zz * (zz - 3 * xx) - xx * (3 * zz - xx)) * sh[..., 24])
    return result
