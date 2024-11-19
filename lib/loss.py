import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from einops import rearrange

def sequence_depth_loss(depth_pred, depth_gt, loss_gamma=0.9):
    """ Loss function defined over sequence of flow predictions """

    valid = (depth_gt != 0.0)
    assert not torch.isinf(depth_gt[valid.bool()]).any()
    bound = (torch.max(depth_gt)-torch.min(depth_gt))

    i_loss = (depth_pred[0] - depth_gt).abs()
    depth_loss =  i_loss[valid.bool()].sum()

    epe = torch.sum((depth_pred[-1] - depth_gt)**2, dim=2).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    

    metrics = {
        'train_epe': epe.mean().item(),
        'train_1pc': (epe < 0.01*bound).float().mean().item(),
        'train_3pc': (epe < 0.03*bound).float().mean().item()
    }

    return depth_loss, metrics

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt)**2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).contiguous().view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def discriminator_loss(real_output, fake_output):
    real_loss = -torch.mean(torch.log(real_output + 1e-12))
    fake_loss = -torch.mean(torch.log(1 - fake_output + 1e-12))
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    g_loss = -torch.mean(torch.log(fake_output + 1e-12))
    return g_loss

def opacity_regular(data):
    loss = 0
    for view in ['lmain', 'rmain']:
        opacity = data[view]['opacity_maps'].permute(0, 2, 3, 1).view(-1, 1)
        loss = loss + (opacity*torch.log(opacity)).mean()
    return loss

def scale_regular(data):
    loss = 0
    for view in ['lmain', 'rmain']:
        scale = data[view]['scale_maps'].permute(0, 2, 3, 1).reshape(-1, 3)
        loss = loss + scale.mean()
    return loss

def anchor_loss(data):
    t1 = 0.1
    t2 = 0.05
    loss = torch.tensor(0.)
    bs =  data['lmain']['img'].shape[0]
    width = data['lmain']['img'].shape[3]
    for i in range(bs):
        if (data['lmain']['face'][i].sum() != 0) and (data['rmain']['face'][i].sum() != 0):
            for (anchor_left, anchor_right) in torch.stack((data['lmain']['face'][i], data['rmain']['face'][i]), dim=1):
                dist = data['lmain']['xyz'][i, int(anchor_left[0])+width*int(anchor_left[1]), :] - data['rmain']['xyz'][i, int(anchor_right[0])+width*int(anchor_right[1]), :]
                loss = loss + (torch.abs(torch.sum(dist**2).sqrt()-t2)+torch.abs(torch.sum(dist**2).sqrt()+t2))/2
        
    return loss

