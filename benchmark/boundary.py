import os
import random
import time
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm

class plane_dataset(torch.utils.data.Dataset):
    def __init__(self, base_img, vec1, vec2, coords, resolution=0.2,
                    range_l=.1, range_r=.1):
        self.base_img = base_img
        self.vec1 = vec1
        self.vec2 = vec2
        self.coords = coords
        x_bounds = [coord[0] for coord in coords]
        y_bounds = [coord[1] for coord in coords]

        self.bound1 = [torch.min(torch.tensor(x_bounds)), torch.max(torch.tensor(x_bounds))]
        self.bound2 = [torch.min(torch.tensor(y_bounds)), torch.max(torch.tensor(y_bounds))]

        len1 = self.bound1[-1] - self.bound1[0]
        len2 = self.bound2[-1] - self.bound2[0]

        #list1 = torch.linspace(self.bound1[0] - 0.1*len1, self.bound1[1] + 0.1*len1, int(resolution))
        #list2 = torch.linspace(self.bound2[0] - 0.1*len2, self.bound2[1] + 0.1*len2, int(resolution))
        # linspace为()搜索，步长为 resolution
        list1 = torch.linspace(self.bound1[0] - range_l*len1, self.bound1[1] + range_r*len1, int(resolution))
        list2 = torch.linspace(self.bound2[0] - range_l*len2, self.bound2[1] + range_r*len2, int(resolution))

        grid = torch.meshgrid([list1, list2])

        self.coefs1 = grid[0].flatten()
        self.coefs2 = grid[1].flatten()

    def __len__(self):
        return self.coefs1.shape[0]

    def __getitem__(self, idx):
        return self.base_img + self.coefs1[idx] * self.vec1 + self.coefs2[idx] * self.vec2

def decision_boundary(net, loader, device):
    net.eval().to(device)
    predicted_labels = []
    with torch.no_grad():
        for inputs in tqdm(loader, desc="Computing decision boundary"):
            inputs = inputs.to(device)
            outputs = net(inputs)
            for output in outputs:
                predicted_labels.append(output)
    return predicted_labels

def get_plane(img1, img2, img3):
    ''' Calculate the plane (basis vecs) spanned by 3 images
    Input: 3 image tensors of the same size
    Output: two (orthogonal) basis vectors for the plane spanned by them, and
    the second vector (before being made orthogonal)
    '''
    a = img2 - img1
    b = img3 - img1
    a_norm = torch.dot(a.flatten(), a.flatten()).sqrt()
    a = a / a_norm
    first_coef = torch.dot(a.flatten(), b.flatten())
    #first_coef = torch.dot(a.flatten(), b.flatten()) / torch.dot(a.flatten(), a.flatten())
    b_orthog = b - first_coef * a
    b_orthog_norm = torch.dot(b_orthog.flatten(), b_orthog.flatten()).sqrt()
    b_orthog = b_orthog / b_orthog_norm
    second_coef = torch.dot(b.flatten(), b_orthog.flatten())
    #second_coef = torch.dot(b_orthog.flatten(), b.flatten()) / torch.dot(b_orthog.flatten(), b_orthog.flatten())
    coords = [[0,0], [a_norm,0], [first_coef, second_coef]]
    return a, b_orthog, b, coords

def make_planeloader(images, resolution=0.2, range_l=.1, range_r=.1):
    a, b_orthog, b, coords = get_plane(images[0], images[1], images[2])

    planeset = plane_dataset(images[0], a, b_orthog, coords, resolution=resolution, range_l=range_l, range_r=range_r)

    planeloader = torch.utils.data.DataLoader(
        planeset, batch_size=256, shuffle=False, num_workers=2)
    return planeloader

def get_random_images(trainset):
    imgs = []
    labels = []
    ids = []
    while len(imgs) < 3:
        idx = random.randint(0, len(trainset)-1)
        img, label = trainset[idx]
        if label not in labels:
            imgs.append(img)
            labels.append(label)
            ids.append(idx)

    return imgs, labels, ids

def get_noisy_images(dummy_imgs, dataset, net, device, from_scratch=False):
    dm = (0.5 * torch.ones(dummy_imgs.shape[0])).unsqueeze(-1).unsqueeze(-1)
    ds = (0.25 * torch.ones(dummy_imgs.shape[0])).unsqueeze(-1).unsqueeze(-1)
    new_imgs = []
    new_labels = []
    net.eval()
    with torch.no_grad():
        while len(new_labels) < dummy_imgs.shape[0]:
            imgs = torch.rand(dummy_imgs.shape)
            imgs = (imgs - dm) / ds
            imgs = imgs.to(device)
            outputs = net(imgs)
            _, labels = outputs.max(1)
            if from_scratch:
                new_imgs = [img.cpu() for img in imgs]
                new_labels = [label.cpu() for label in labels]
                break
            for i, label in enumerate(labels):
                ''' LF this takes too long for random training dynamics... 
                if label.cpu() not in new_labels:
                    new_imgs.append(imgs[i].cpu())
                    new_labels.append(label.cpu())
                '''
                new_imgs.append(imgs[i].cpu())
                new_labels.append(label.cpu())
    return new_imgs, new_labels

def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text+": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def plot(net, plot_path, dataset, trainloader, imgs=None, device='cuda', **kwargs):
    start = time.time()
    end = time.time()
    simple_lapsed_time("Time taken to train/load the model", end - start)
    data = [(x, y) for x, y in zip(*dataset)]
    start = time.time()
    if imgs is None:
        images, labels, _ = get_random_images(data)
    elif -1 in imgs:
        dummy_imgs, _, _ = get_random_images(data)
        images, labels = get_noisy_images(torch.stack(dummy_imgs), data, net, device)
    elif -10 in imgs:
        image_ids = imgs[0]
        images = [data[image_ids][0]]
        labels = [data[image_ids][1]]
        for i in list(range(2)):
            temp = torch.zeros_like(images[0])
            if i == 0:
                temp[0, 0, 0] = 1
            else:
                temp[0, -1, -1] = 1
            images.append(temp)
            labels.append(0)
    else:
        image_ids = imgs
        images = [data[i][0] for i in image_ids]
        labels = [data[i][1] for i in image_ids]

    planeloader = make_planeloader(images, **kwargs)
    preds = decision_boundary(net, planeloader, device)

    os.makedirs(plot_path, exist_ok=True)
    plot_path = os.path.join(plot_path, '_'.join(f'{k}={v}' for k, v in kwargs.items()))
    produce_plot_alt(plot_path, preds, planeloader, images, labels, trainloader)

    end = time.time()
    simple_lapsed_time("Time taken to plot the image", end - start)

def produce_plot_alt(path, preds, planeloader, images, labels, trainloader, epoch='best', temp=1.0):
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap
    col_map = cm.get_cmap('tab10')
    cmaplist = [col_map(i) for i in range(col_map.N)]
    classes = ['airpl', 'autom', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    cmaplist = cmaplist[:len(classes)]
    col_map = LinearSegmentedColormap.from_list('custom_colormap', cmaplist, N=len(classes))
    fig, ax1 = plt.subplots()
    import torch.nn as nn
    preds = torch.stack((preds))
    preds = nn.Softmax(dim=1)(preds / temp)
    val = torch.max(preds,dim=1)[0].cpu().numpy()
    class_pred = torch.argmax(preds, dim=1).cpu().numpy()
    x = planeloader.dataset.coefs1.cpu().numpy()
    y = planeloader.dataset.coefs2.cpu().numpy()
    label_color_dict = dict(zip([*range(10)], cmaplist))

    color_idx = [label_color_dict[label] for label in class_pred]
    scatter = ax1.scatter(x, y, c=color_idx, alpha=val, s=0.1)
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in label_color_dict.values()]
    legend1 = plt.legend(markers, classes, numpoints=1,bbox_to_anchor=(1.01, 1))
    ax1.add_artist(legend1)
    coords = planeloader.dataset.coords

    dm = torch.tensor(trainloader.dataset.transform.transforms[-1].mean)[:, None, None]
    ds = torch.tensor(trainloader.dataset.transform.transforms[-1].std)[:, None, None]
    for i, image in enumerate(images):
        # import ipdb; ipdb.set_trace()
        img = torch.clamp(image * ds + dm, 0, 1)
        img = img.cpu().numpy().transpose(1,2,0)
        if img.shape[0] > 32:
            from PIL import Image
            img = img*255
            img = img.astype(np.uint8)
            img = Image.fromarray(img).resize(size=(32, 32))
            img = np.array(img)

        coord = coords[i]
        imscatter(coord[0], coord[1], img, ax1)

    red_patch = mpatches.Patch(color =cmaplist[labels[0]] , label=f'{classes[labels[0]]}')
    blue_patch = mpatches.Patch(color =cmaplist[labels[1]], label=f'{classes[labels[1]]}')
    green_patch = mpatches.Patch(color =cmaplist[labels[2]], label=f'{classes[labels[2]]}')
    plt.legend(handles=[red_patch, blue_patch, green_patch], loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)
    plt.title(f'Epoch: {epoch}')
    if path is not None:
        plt.savefig(f'{path}.png',bbox_extra_artists=(legend1,), bbox_inches='tight')
        print(f"Saved plot to {path}.png")
    plt.close(fig)
    return

def imscatter(x, y, image, ax=None, zoom=1):
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
    artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists