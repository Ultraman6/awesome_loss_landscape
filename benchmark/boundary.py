import os
import random
import time
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from torch import flip
from torch.autograd import grad
from tqdm import tqdm

def check_shape(x):
    return x if len(x.shape) > 0 else x.unsqueeze(0)

def replicate_input(x):
    return x.detach().clone()

def get_func_value_grads_batch(x, y, model, func, **kwargs_func):
    # df: bs, the func values
    # dg: bs * img_size, the input grads of func

    im = x.clone().requires_grad_()
    df = func(model, im, y, **kwargs_func)
    dg = grad(outputs=df, inputs=im, grad_outputs=torch.ones_like(df))[0].detach()

    return df.detach(), dg.detach()

DEFAULT_EPS_DICT_BY_NORM = {'Linf': .3, 'L2': 1., 'L1': 5.0}
def perturb(x, y, device, model, func, norm='Linf', n_restarts=1, n_iter=100,
            alpha_max=0.1, eta=1.05, beta=0.9, return_all=False, verbose=True):
    """
    :param x:    clean images
    :param y:    clean labels
    :param model: NN with num_classes logits
    :param func: the scalar function; needs to be func(model, x, y, **kwargs_func)

    :kwargs_func: other parameters of the func, which includes
        :param temperature (when using CE)


    Output: hat(x) s.t. fun(hat(x), y, model, **kwargs_func) = 0

    :param return_all:
    if True, return fab samples for all correct points (even if fab sample is not adv example)
    if False, only return fab samples that 1. original pt is correct 2. fab sample is adv example.

    Note: here 'correct' point means func > 0, 'adv_example' means func < 0
    """

    orig_dim = list(x.shape[1:])
    ndims = len(orig_dim)
    eps = DEFAULT_EPS_DICT_BY_NORM[norm]
    x = x.detach().clone().float().to(device)

    with torch.no_grad():
        func_value = func(model, x, y)
    pred = (func_value > 0)  # 'correct' points
    corr_classified = pred.float().sum()  # only for statistics
    if verbose:
        print('FAB_soft_bdr: \'Clean\' accuracy: {:.2%}'.format(pred.float().mean()))
    if pred.sum() == 0:
        return x
    pred = check_shape(pred.nonzero().squeeze())

    startt = time.time()
    # runs the attack only on 'correctly' classified points
    im2 = replicate_input(x[pred])  # 'correctly' classified points
    la2 = replicate_input(y[pred])
    if len(im2.shape) == ndims:
        im2 = im2.unsqueeze(0)
    bs = im2.shape[0]  # number of 'correct' pts
    u1 = torch.arange(bs)
    adv = im2.clone()
    adv_c = x.clone()
    res2 = 1e10 * torch.ones([bs]).to(device)
    res_c = torch.zeros([x.shape[0]]).to(device)
    x1 = im2.clone()
    x0 = im2.clone().reshape([bs, -1])
    counter_restarts = 0

    while counter_restarts < n_restarts:
        #             if counter_restarts > 0:
        if counter_restarts > -1:
            # perturb with radius eps/2 (as in the paper)
            if norm == 'Linf':
                t = 2 * torch.rand(x1.shape).to(device) - 1
                x1 = im2 + (
                    torch.min(
                        res2,
                        eps * torch.ones(res2.shape).to(device)
                    ).reshape([-1, *([1] * ndims)])
                ) * t / (t.reshape([t.shape[0], -1]).abs()
                         .max(dim=1, keepdim=True)[0]
                         .reshape([-1, *([1] * ndims)])) * .5
            elif norm == 'L2':
                t = torch.randn(x1.shape).to(device)
                x1 = im2 + (
                    torch.min(
                        res2,
                        eps * torch.ones(res2.shape).to(device)
                    ).reshape([-1, *([1] * ndims)])
                ) * t / ((t ** 2)
                         .view(t.shape[0], -1)
                         .sum(dim=-1)
                         .sqrt()
                         .view(t.shape[0], *([1] * ndims))) * .5
            elif norm == 'L1':
                t = torch.randn(x1.shape).to(device)
                x1 = im2 + (torch.min(
                    res2,
                    eps * torch.ones(res2.shape).to(device)
                ).reshape([-1, *([1] * ndims)])
                            ) * t / (t.abs().view(t.shape[0], -1)
                                     .sum(dim=-1)
                                     .view(t.shape[0], *([1] * ndims))) / 2

            x1 = x1.clamp(0.0, 1.0)

        counter_iter = 0
        while counter_iter < n_iter:
            df, dg = get_func_value_grads_batch(x=x1, y=la2, model=model, func=func)
            with torch.no_grad():

                b = (- df +
                     (dg * x1).view(x1.shape[0], -1).sum(dim=-1))
                w = dg.reshape([bs, -1])

                if norm == 'Linf':
                    d3 = projection_linf(
                        torch.cat((x1.reshape([bs, -1]), x0), 0),
                        torch.cat((w, w), 0),
                        torch.cat((b, b), 0), device)
                elif norm == 'L2':
                    d3 = projection_l2(
                        torch.cat((x1.reshape([bs, -1]), x0), 0),
                        torch.cat((w, w), 0),
                        torch.cat((b, b), 0), device)
                elif norm == 'L1':
                    d3 = projection_l1(
                        torch.cat((x1.reshape([bs, -1]), x0), 0),
                        torch.cat((w, w), 0),
                        torch.cat((b, b), 0), device)
                # d3: (2*bs) * (3*32*32); d1: bs*3*32*32
                # x1+d1, x_orig+d2 are on the plane: (w*(x1+d1).view(x1.size()[0],-1)).sum(1) - b = 0
                d1 = torch.reshape(d3[:bs], x1.shape)
                d2 = torch.reshape(d3[-bs:], x1.shape)

                # biased graident
                if norm == 'Linf':
                    a0 = d3.abs().max(dim=1, keepdim=True)[0] \
                        .view(-1, *([1] * ndims))
                elif norm == 'L2':
                    a0 = (d3 ** 2).sum(dim=1, keepdim=True).sqrt() \
                        .view(-1, *([1] * ndims))
                elif norm == 'L1':
                    a0 = d3.abs().sum(dim=1, keepdim=True) \
                        .view(-1, *([1] * ndims))
                a0 = torch.max(a0, 1e-8 * torch.ones(
                    a0.shape).to(device))
                a1 = a0[:bs]
                a2 = a0[-bs:]
                alpha = torch.min(torch.max(a1 / (a1 + a2),
                                            torch.zeros(a1.shape)
                                            .to(device))[0],
                                  alpha_max * torch.ones(a1.shape)
                                  .to(device))
                x1 = ((x1 + eta * d1) * (1 - alpha) +
                      (im2 + d2 * eta) * alpha).clamp(0.0, 1.0)

                with torch.no_grad():
                    func_value = func(model, x1, la2)
                is_adv = (func_value < 0)  # 'adv' points

                # backward step
                if is_adv.sum() > 0:
                    ind_adv = is_adv.nonzero().squeeze()
                    ind_adv = check_shape(ind_adv)
                    if norm == 'Linf':
                        t = (x1[ind_adv] - im2[ind_adv]).reshape(
                            [ind_adv.shape[0], -1]).abs().max(dim=1)[0]
                    elif norm == 'L2':
                        t = ((x1[ind_adv] - im2[ind_adv]) ** 2) \
                            .view(ind_adv.shape[0], -1).sum(dim=-1).sqrt()
                    elif norm == 'L1':
                        t = (x1[ind_adv] - im2[ind_adv]) \
                            .abs().view(ind_adv.shape[0], -1).sum(dim=-1)
                    adv[ind_adv] = x1[ind_adv] * (t < res2[ind_adv]). \
                        float().reshape([-1, *([1] * ndims)]) \
                                   + adv[ind_adv] \
                                   * (t >= res2[ind_adv]).float().reshape(
                        [-1, *([1] * ndims)])
                    res2[ind_adv] = t * (t < res2[ind_adv]).float() \
                                    + res2[ind_adv] * (t >= res2[ind_adv]).float()
                    x1[ind_adv] = im2[ind_adv] + (
                            x1[ind_adv] - im2[ind_adv]) * beta

                counter_iter += 1

        counter_restarts += 1

    ind_succ = res2 < 1e10
    if verbose:
        print('success rate: {:.0f}/{:.0f}'
              .format(ind_succ.float().sum(), corr_classified) +
              ' (on correctly classified points) in {:.1f} s'
              .format(time.time() - startt))

    res_c[pred] = res2 * ind_succ.float() + 1e10 * (1 - ind_succ.float())
    ind_succ_ = check_shape(ind_succ.nonzero().squeeze())
    adv_c[pred[ind_succ_]] = adv[ind_succ_].clone()

    if return_all:
        ind_fail = ~ind_succ
        inf_fail_ = check_shape(ind_fail.nonzero().squeeze())
        adv_c[pred[inf_fail_]] = x1[inf_fail_].clone()

    # pred: index of correctly classified points
    # ind_succ: index of successfully attacked points, inside pred

    return adv_c.detach(), pred, ind_succ

def projection_linf(points_to_project, w_hyperplane, b_hyperplane, device):
    # the plane is: {z: <w,z>-b=0}
    t = points_to_project.clone()
    w = w_hyperplane.clone()
    b = b_hyperplane.clone()

    ind2 = ((w * t).sum(1) - b < 0).nonzero().squeeze()
    ind2 = check_shape(ind2)
    w[ind2] *= -1
    b[ind2] *= -1

    c5 = (w < 0).float()
    a = torch.ones(t.shape).to(device)
    d = (a * c5 - t) * (w != 0).float()
    a -= a * (1 - c5)

    p = torch.ones(t.shape).to(device) * c5 - t * (2 * c5 - 1)
    _, indp = torch.sort(p, dim=1)

    b = b - (w * t).sum(1)
    b0 = (w * d).sum(1)
    b1 = b0.clone()

    counter = 0
    indp2 = flip(indp.unsqueeze(-1), dims=(1, 2)).squeeze()
    u = torch.arange(0, w.shape[0])
    ws = w[u.unsqueeze(1), indp2]
    bs2 = - ws * d[u.unsqueeze(1), indp2]

    s = torch.cumsum(ws.abs(), dim=1)
    sb = torch.cumsum(bs2, dim=1) + b0.unsqueeze(1)

    c = b - b1 > 0
    b2 = sb[u, -1] - s[u, -1] * p[u, indp[u, 0]]
    c_l = (b - b2 > 0).nonzero().squeeze()
    c2 = ((b - b1 > 0) * (b - b2 <= 0)).nonzero().squeeze()
    c_l = check_shape(c_l)
    c2 = check_shape(c2)

    lb = torch.zeros(c2.shape[0])
    ub = torch.ones(c2.shape[0]) * (w.shape[1] - 1)
    nitermax = torch.ceil(torch.log2(torch.tensor(w.shape[1]).float()))

    while counter < nitermax:
        counter4 = torch.floor((lb + ub) / 2)
        counter2 = counter4.long()
        indcurr = indp[c2, -counter2 - 1]
        b2 = sb[c2, counter2] - s[c2, counter2] * p[c2, indcurr]
        c = b[c2] - b2 > 0
        ind3 = c.nonzero().squeeze()
        ind32 = (~c).nonzero().squeeze()
#             ind3 = self.check_shape(ind3)
#             ind32 = self.check_shape(ind32)
        ind3 = check_shape(ind3).cpu()
        ind32 = check_shape(ind32).cpu()
        lb[ind3] = counter4[ind3]
        ub[ind32] = counter4[ind32]
        counter += 1

    lb = lb.long()

    if c_l.nelement() != 0:
        lmbd_opt = (torch.max((b[c_l] - sb[c_l, -1]) / (-s[c_l, -1]),
                              torch.zeros(sb[c_l, -1].shape)
                              .to(device))).unsqueeze(-1)
        d[c_l] = (2 * a[c_l] - 1) * lmbd_opt

    lmbd_opt = (torch.max((b[c2] - sb[c2, lb]) / (-s[c2, lb]),
                          torch.zeros(sb[c2, lb].shape)
                          .to(device))).unsqueeze(-1)
    d[c2] = torch.min(lmbd_opt, d[c2]) * c5[c2]\
        + torch.max(-lmbd_opt, d[c2]) * (1 - c5[c2])

    return d * (w != 0).float()

def projection_l2(points_to_project, w_hyperplane, b_hyperplane, device):
    t = points_to_project.clone()
    w = w_hyperplane.clone()
    b = b_hyperplane.clone()

    c = (w * t).sum(1) - b
    ind2 = (c < 0).nonzero().squeeze()
    ind2 = check_shape(ind2)
    w[ind2] *= -1
    c[ind2] *= -1

    u = torch.arange(0, w.shape[0]).unsqueeze(1)

    r = torch.max(t / w, (t - 1) / w)
    u2 = torch.ones(r.shape).to(device)
    r = torch.min(r, 1e12 * u2)
    r = torch.max(r, -1e12 * u2)
    r[w.abs() < 1e-8] = 1e12
    r[r == -1e12] = -r[r == -1e12]
    rs, indr = torch.sort(r, dim=1)
    rs2 = torch.cat((rs[:, 1:],
                     torch.zeros(rs.shape[0], 1).to(device)), 1)
    rs[rs == 1e12] = 0
    rs2[rs2 == 1e12] = 0

    w3 = w ** 2
    w3s = w3[u, indr]
    w5 = w3s.sum(dim=1, keepdim=True)
    ws = w5 - torch.cumsum(w3s, dim=1)
    d = -(r * w).clone()
    d = d * (w.abs() > 1e-8).float()
    s = torch.cat(((-w5.squeeze() * rs[:, 0]).unsqueeze(1),
                   torch.cumsum((-rs2 + rs) * ws, dim=1) -
                   w5 * rs[:, 0].unsqueeze(-1)), 1)

    c4 = (s[:, 0] + c < 0)
    c3 = ((d * w).sum(dim=1) + c > 0)
    c6 = c4.nonzero().squeeze()
    c2 = ((1 - c4.float()) * (1 - c3.float())).nonzero().squeeze()
    c6 = check_shape(c6)
    c2 = check_shape(c2)

    counter = 0
    lb = torch.zeros(c2.shape[0])
    ub = torch.ones(c2.shape[0]) * (w.shape[1] - 1)
    nitermax = torch.ceil(torch.log2(torch.tensor(w.shape[1]).float()))
    counter2 = torch.zeros(lb.shape).long()

    while counter < nitermax:
        counter4 = torch.floor((lb + ub) / 2)
        counter2 = counter4.long()
        c3 = s[c2, counter2] + c[c2] > 0
        ind3 = c3.nonzero().squeeze()
        ind32 = (~c3).nonzero().squeeze()
        ind3 = check_shape(ind3)
        ind32 = check_shape(ind32)
        lb[ind3] = counter4[ind3]
        ub[ind32] = counter4[ind32]
        counter += 1

    lb = lb.long()
    alpha = torch.zeros([1])

    if c6.nelement() != 0:
        alpha = c[c6] / w5[c6].squeeze(-1)
        d[c6] = -alpha.unsqueeze(-1) * w[c6]

    if c2.nelement() != 0:
        alpha = (s[c2, lb] + c[c2]) / ws[c2, lb] + rs[c2, lb]
        if torch.sum(ws[c2, lb] == 0) > 0:
            ind = (ws[c2, lb] == 0).nonzero().squeeze().long()
            ind = check_shape(ind)
            alpha[ind] = 0
        c5 = (alpha.unsqueeze(-1) > r[c2]).float()
        d[c2] = d[c2] * c5 - alpha.unsqueeze(-1) * w[c2] * (1 - c5)

    return d * (w.abs() > 1e-8).float()

def projection_l1(points_to_project, w_hyperplane, b_hyperplane, device):
    t = points_to_project.clone()
    w = w_hyperplane.clone()
    b = b_hyperplane.clone()

    c = (w * t).sum(1) - b
    ind2 = (c < 0).nonzero().squeeze()
    ind2 = check_shape(ind2)
    w[ind2] *= -1
    c[ind2] *= -1

    r = torch.max(1 / w, -1 / w)
    r = torch.min(r, 1e12 * torch.ones(r.shape).to(device))
    rs, indr = torch.sort(r, dim=1)
    _, indr_rev = torch.sort(indr)

    u = torch.arange(0, w.shape[0]).unsqueeze(1)
    u2 = torch.arange(0, w.shape[1]).repeat(w.shape[0], 1)
    c6 = (w < 0).float()
    d = (-t + c6) * (w != 0).float()
    d2 = torch.min(-w * t, w * (1 - t))
    ds = d2[u, indr]
    ds2 = torch.cat((c.unsqueeze(-1), ds), 1)
    s = torch.cumsum(ds2, dim=1)

    c4 = s[:, -1] < 0
    c2 = c4.nonzero().squeeze(-1)
    c2 = check_shape(c2)

    counter = 0
    lb = torch.zeros(c2.shape[0])
    ub = torch.ones(c2.shape[0]) * (s.shape[1])
    nitermax = torch.ceil(torch.log2(torch.tensor(s.shape[1]).float()))
    counter2 = torch.zeros(lb.shape).long()

    while counter < nitermax:
        counter4 = torch.floor((lb + ub) / 2)
        counter2 = counter4.long()
        c3 = s[c2, counter2] > 0
        ind3 = c3.nonzero().squeeze()
        ind32 = (~c3).nonzero().squeeze()
        ind3 = check_shape(ind3)
        ind32 = check_shape(ind32)
        lb[ind3] = counter4[ind3]
        ub[ind32] = counter4[ind32]
        counter += 1

    lb2 = lb.long()

    if c2.nelement() != 0:
        alpha = -s[c2, lb2] / w[c2, indr[c2, lb2]]
        c5 = u2[c2].float() < lb.unsqueeze(-1).float()
        u3 = c5[u[:c5.shape[0]], indr_rev[c2]]
        d[c2] = d[c2] * u3.float().to(device)
        d[c2, indr[c2, lb2]] = alpha

    return d * (w.abs() > 1e-8).float()

# class plane_dataset(torch.utils.data.Dataset):
#     def __init__(self, base_img, vec1, vec2, coords, resolution=0.2,
#                     range_l=.1, range_r=.1):
#         self.base_img = base_img
#         self.vec1 = vec1
#         self.vec2 = vec2
#         self.coords = coords
#         x_bounds = [coord[0] for coord in coords]
#         y_bounds = [coord[1] for coord in coords]
#
#         self.bound1 = [torch.min(torch.tensor(x_bounds)), torch.max(torch.tensor(x_bounds))]
#         self.bound2 = [torch.min(torch.tensor(y_bounds)), torch.max(torch.tensor(y_bounds))]
#
#         len1 = self.bound1[-1] - self.bound1[0]
#         len2 = self.bound2[-1] - self.bound2[0]
#
#         #list1 = torch.linspace(self.bound1[0] - 0.1*len1, self.bound1[1] + 0.1*len1, int(resolution))
#         #list2 = torch.linspace(self.bound2[0] - 0.1*len2, self.bound2[1] + 0.1*len2, int(resolution))
#         # linspace为()搜索，步长为 resolution
#         list1 = torch.linspace(self.bound1[0] - range_l*len1, self.bound1[1] + range_r*len1, int(resolution))
#         list2 = torch.linspace(self.bound2[0] - range_l*len2, self.bound2[1] + range_r*len2, int(resolution))
#
#         grid = torch.meshgrid([list1, list2])
#
#         self.coefs1 = grid[0].flatten()
#         self.coefs2 = grid[1].flatten()
#
#     def __len__(self):
#         return self.coefs1.shape[0]
#
#     def __getitem__(self, idx):
#         return self.base_img + self.coefs1[idx] * self.vec1 + self.coefs2[idx] * self.vec2
#
# def decision_boundary(net, loader, device):
#     net.eval().to(device)
#     predicted_labels = []
#     with torch.no_grad():
#         for inputs in tqdm(loader, desc="Computing decision boundary"):
#             inputs = inputs.to(device)
#             outputs = net(inputs)
#             for output in outputs:
#                 predicted_labels.append(output)
#     return predicted_labels
#
# def get_plane(img1, img2, img3):
#     ''' Calculate the plane (basis vecs) spanned by 3 images
#     Input: 3 image tensors of the same size
#     Output: two (orthogonal) basis vectors for the plane spanned by them, and
#     the second vector (before being made orthogonal)
#     '''
#     a = img2 - img1
#     b = img3 - img1
#     a_norm = torch.dot(a.flatten(), a.flatten()).sqrt()
#     a = a / a_norm
#     first_coef = torch.dot(a.flatten(), b.flatten())
#     #first_coef = torch.dot(a.flatten(), b.flatten()) / torch.dot(a.flatten(), a.flatten())
#     b_orthog = b - first_coef * a
#     b_orthog_norm = torch.dot(b_orthog.flatten(), b_orthog.flatten()).sqrt()
#     b_orthog = b_orthog / b_orthog_norm
#     second_coef = torch.dot(b.flatten(), b_orthog.flatten())
#     #second_coef = torch.dot(b_orthog.flatten(), b.flatten()) / torch.dot(b_orthog.flatten(), b_orthog.flatten())
#     coords = [[0,0], [a_norm,0], [first_coef, second_coef]]
#     return a, b_orthog, b, coords
#
# def make_planeloader(images, resolution=0.2, range_l=.1, range_r=.1):
#     a, b_orthog, b, coords = get_plane(images[0], images[1], images[2])
#
#     planeset = plane_dataset(images[0], a, b_orthog, coords, resolution=resolution, range_l=range_l, range_r=range_r)
#
#     planeloader = torch.utils.data.DataLoader(
#         planeset, batch_size=256, shuffle=False, num_workers=2)
#     return planeloader
#
# def get_random_images(trainset):
#     imgs = []
#     labels = []
#     ids = []
#     while len(imgs) < 3:
#         idx = random.randint(0, len(trainset)-1)
#         img, label = trainset[idx]
#         if label not in labels:
#             imgs.append(img)
#             labels.append(label)
#             ids.append(idx)
#
#     return imgs, labels, ids
#
# def get_noisy_images(dummy_imgs, dataset, net, device, from_scratch=False):
#     dm = (0.5 * torch.ones(dummy_imgs.shape[0])).unsqueeze(-1).unsqueeze(-1)
#     ds = (0.25 * torch.ones(dummy_imgs.shape[0])).unsqueeze(-1).unsqueeze(-1)
#     new_imgs = []
#     new_labels = []
#     net.eval()
#     with torch.no_grad():
#         while len(new_labels) < dummy_imgs.shape[0]:
#             imgs = torch.rand(dummy_imgs.shape)
#             imgs = (imgs - dm) / ds
#             imgs = imgs.to(device)
#             outputs = net(imgs)
#             _, labels = outputs.max(1)
#             if from_scratch:
#                 new_imgs = [img.cpu() for img in imgs]
#                 new_labels = [label.cpu() for label in labels]
#                 break
#             for i, label in enumerate(labels):
#                 ''' LF this takes too long for random training dynamics...
#                 if label.cpu() not in new_labels:
#                     new_imgs.append(imgs[i].cpu())
#                     new_labels.append(label.cpu())
#                 '''
#                 new_imgs.append(imgs[i].cpu())
#                 new_labels.append(label.cpu())
#     return new_imgs, new_labels
#
# def simple_lapsed_time(text, lapsed):
#     hours, rem = divmod(lapsed, 3600)
#     minutes, seconds = divmod(rem, 60)
#     print(text+": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
#
# def plot(net, plot_path, dataset, trainloader, imgs=None, device='cuda', **kwargs):
#     start = time.time()
#     end = time.time()
#     simple_lapsed_time("Time taken to train/load the model", end - start)
#     data = [(x, y) for x, y in zip(*dataset)]
#     start = time.time()
#     if imgs is None:
#         images, labels, _ = get_random_images(data)
#     elif -1 in imgs:
#         dummy_imgs, _, _ = get_random_images(data)
#         images, labels = get_noisy_images(torch.stack(dummy_imgs), data, net, device)
#     elif -10 in imgs:
#         image_ids = imgs[0]
#         images = [data[image_ids][0]]
#         labels = [data[image_ids][1]]
#         for i in list(range(2)):
#             temp = torch.zeros_like(images[0])
#             if i == 0:
#                 temp[0, 0, 0] = 1
#             else:
#                 temp[0, -1, -1] = 1
#             images.append(temp)
#             labels.append(0)
#     else:
#         image_ids = imgs
#         images = [data[i][0] for i in image_ids]
#         labels = [data[i][1] for i in image_ids]
#
#     planeloader = make_planeloader(images, **kwargs)
#     preds = decision_boundary(net, planeloader, device)
#
#     os.makedirs(plot_path, exist_ok=True)
#     plot_path = os.path.join(plot_path, '_'.join(f'{k}={v}' for k, v in kwargs.items()))
#     produce_plot_alt(plot_path, preds, planeloader, images, labels, trainloader)
#
#     end = time.time()
#     simple_lapsed_time("Time taken to plot the image", end - start)
#
# def produce_plot_alt(path, preds, planeloader, images, labels, trainloader, epoch='best', temp=1.0):
#     from matplotlib import cm
#     from matplotlib.colors import LinearSegmentedColormap
#     col_map = cm.get_cmap('tab10')
#     cmaplist = [col_map(i) for i in range(col_map.N)]
#     classes = ['airpl', 'autom', 'bird', 'cat', 'deer',
#                    'dog', 'frog', 'horse', 'ship', 'truck']
#
#     cmaplist = cmaplist[:len(classes)]
#     col_map = LinearSegmentedColormap.from_list('custom_colormap', cmaplist, N=len(classes))
#     fig, ax1 = plt.subplots()
#     import torch.nn as nn
#     preds = torch.stack((preds))
#     preds = nn.Softmax(dim=1)(preds / temp)
#     val = torch.max(preds,dim=1)[0].cpu().numpy()
#     class_pred = torch.argmax(preds, dim=1).cpu().numpy()
#     x = planeloader.dataset.coefs1.cpu().numpy()
#     y = planeloader.dataset.coefs2.cpu().numpy()
#     label_color_dict = dict(zip([*range(10)], cmaplist))
#
#     color_idx = [label_color_dict[label] for label in class_pred]
#     scatter = ax1.scatter(x, y, c=color_idx, alpha=val, s=0.1)
#     markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in label_color_dict.values()]
#     legend1 = plt.legend(markers, classes, numpoints=1,bbox_to_anchor=(1.01, 1))
#     ax1.add_artist(legend1)
#     coords = planeloader.dataset.coords
#
#     dm = torch.tensor(trainloader.dataset.transform.transforms[-1].mean)[:, None, None]
#     ds = torch.tensor(trainloader.dataset.transform.transforms[-1].std)[:, None, None]
#     for i, image in enumerate(images):
#         # import ipdb; ipdb.set_trace()
#         img = torch.clamp(image * ds + dm, 0, 1)
#         img = img.cpu().numpy().transpose(1,2,0)
#         if img.shape[0] > 32:
#             from PIL import Image
#             img = img*255
#             img = img.astype(np.uint8)
#             img = Image.fromarray(img).resize(size=(32, 32))
#             img = np.array(img)
#
#         coord = coords[i]
#         imscatter(coord[0], coord[1], img, ax1)
#
#     red_patch = mpatches.Patch(color =cmaplist[labels[0]] , label=f'{classes[labels[0]]}')
#     blue_patch = mpatches.Patch(color =cmaplist[labels[1]], label=f'{classes[labels[1]]}')
#     green_patch = mpatches.Patch(color =cmaplist[labels[2]], label=f'{classes[labels[2]]}')
#     plt.legend(handles=[red_patch, blue_patch, green_patch], loc='upper center', bbox_to_anchor=(0.5, 1.05),
#               ncol=3, fancybox=True, shadow=True)
#     plt.title(f'Epoch: {epoch}')
#     if path is not None:
#         plt.savefig(f'{path}.png',bbox_extra_artists=(legend1,), bbox_inches='tight')
#         print(f"Saved plot to {path}.png")
#     plt.close(fig)
#     return
#
# def imscatter(x, y, image, ax=None, zoom=1):
#     im = OffsetImage(image, zoom=zoom)
#     x, y = np.atleast_1d(x, y)
#     artists = []
#     ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
#     artists.append(ax.add_artist(ab))
#     ax.update_datalim(np.column_stack([x, y]))
#     ax.autoscale()
#     return artists