"""Example."""
import argparse
import os.path
from src.main import train, plot

def train_save_folder(args):
    save_folder = '_'.join([args.dataset, args.model, args.optimizer])
    save_folder += '_lr=' + str(args.lr)
    save_folder += '_ld=' + str(args.lr_decay)
    save_folder += '_bs=' + str(args.batch_size)
    save_folder += '_wd=' + str(args.weight_decay)
    save_folder += '_mo=' + str(args.momentum)
    save_folder += '_ls=' + str(args.criterion)
    if args.n_examples > 0:
        save_folder += '_num=' + str(args.n_examples)
    else:
        save_folder += '_num=all'
    save_folder += '_sd=' + str(args.seed)
    save_folder += '_es=' + str(args.epochs)
    return save_folder

def plot_save_folder(args):
    save_folder = ''
    save_folder += '_sd=' + str(args.seed)
    save_folder += '_be=' + str(args.begin_epoch)
    save_folder += '_es=' + str(args.end_epoch)
    save_folder += '_se=' + str(args.save_epoch)
    save_folder += '_ce=' + str(args.center_epoch)
    return save_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_root', default='./records')
    parser.add_argument('--plot_root', default=None)
    parser.add_argument('--train', default=True)
    parser.add_argument('--landscape', default=False)
    # train
    parser.add_argument('--gpu', default=True,  help='number of gpus')
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10', 'cifar100'], help='dataset name')
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--model', default='cnn', help='models name')
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--n_examples', default=-1, help='number of examples to train')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--criterion', default='ce', choices=['ce', 'bce', 'mse'])
    parser.add_argument('--optimizer', default='sgd', help='optimizer: sgd | adam')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=0.1, type=float, help='learning rate decay rate')
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--save_epoch', default=1, type=int, help='save every save_epochs')
    parser.add_argument('--resume_epoch', default=3, help='whether to resume training')
    # landscape
    parser.add_argument('--begin_epoch', default=0, type=int)
    parser.add_argument('--center_epoch', default=100, type=int)
    parser.add_argument('--end_epoch', default=100, type=int)
    parser.add_argument('--res', default=50, help='A string with format xmin:x_max:xnum')
    parser.add_argument('--margin', default=0.3, help='A string with format ymin:ymax:ynum')
    parser.add_argument('--xnorm', default='filter', help='direction normalization: filter | layer | weight')
    parser.add_argument('--ynorm', default='filter', help='direction normalization: filter | layer | weight')
    parser.add_argument('--xignore', default='biasbn', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--yignore', default='biasbn', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--dir_type', default='weights', help='direction type: weights | states (including BN\'s running_mean/var)')
    parser.add_argument('--reduction_method', default='random', choices=['random', 'pca', 'custom', 'class'])

    args = parser.parse_args()
    args.save_root = os.path.join(args.save_root, train_save_folder(args))
    args.plot_root = os.path.join(args.save_root, plot_save_folder(args))

    if args.train:
        train(args)
    if args.plot:
        plot(args)

