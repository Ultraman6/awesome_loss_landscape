import argparse, os

def train_save_folder(args):
    save_folder = '_'.join([args.dataset, args.model, args.optimizer])
    save_folder += '_lr=' + str(args.lr)
    save_folder += '_ld=' + str(args.lr_decay)
    save_folder += '_ls=' + str(args.lr_step)
    save_folder += '_bs=' + str(args.batch_size)
    save_folder += '_wd=' + str(args.weight_decay)
    save_folder += '_mo=' + str(args.momentum)
    save_folder += '_cr=' + str(args.criterion)
    save_folder += '_sd=' + str(args.seed)
    save_folder += '_es=' + str(args.epochs)
    return save_folder

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='mps')
parser.add_argument('--num_workers', default=4)
parser.add_argument('--seed', default=0)
parser.add_argument('--save_root', default='../records')
parser.add_argument('--eval', default=True)
parser.add_argument('--num_classes', default=10)
parser.add_argument('--input_dim', default=())
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--data_root', default='./data')
parser.add_argument('--data_fix', default='r-20')
parser.add_argument('--model', default='vgg9')
parser.add_argument('--pretrained', default=True)
parser.add_argument('--batch_size', default=64)
parser.add_argument('--criterion', default='ce', choices=['ce', 'bce', 'mse'])
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--lr', default=0.01)
parser.add_argument('--lr_decay', default=0.1)
parser.add_argument('--lr_step', default=10)
parser.add_argument('--weight_decay', default=0.0005)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--epochs', default=10)
parser.add_argument('--resume_epoch', default=-1)
parser.add_argument('--save_epoch', default=10)

args = parser.parse_args()
args.save_root = os.path.join(args.save_root, train_save_folder(args))
