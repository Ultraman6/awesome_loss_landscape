import argparse, os

def train_save_folder(args):
    fix = ''
    if args.dataset in ['cifar10-LT', 'cifar100-LT']:
        fix = str(args.cifar_imb_ratio)
    save_folder = '_'.join([args.dataset + fix, args.model, args.optimizer])
    save_folder += '_lr=' + str(args.lr)
    save_folder += '_ld=' + str(args.lr_decay)
    save_folder += '_bs=' + str(args.batch_size)
    save_folder += '_wd=' + str(args.weight_decay)
    save_folder += '_mo=' + str(args.momentum)
    save_folder += '_cr=' + str(args.criterion)
    save_folder += '_sd=' + str(args.seed)
    save_folder += '_es=' + str(args.epochs)
    save_folder += '_ie=' + str(args.interval_epoch)
    return save_folder

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=100)
parser.add_argument('--resume_epoch', default=0,
                    help='0 means not resuming -1 means the last, -2 means the best')
parser.add_argument('--interval_epoch', default=10)
parser.add_argument('--device', default='cuda')
parser.add_argument('--num_workers', default=4)
parser.add_argument('--seed', default=0)
parser.add_argument('--save_root', default='../../records')
parser.add_argument('--eval', default=True)
parser.add_argument('--save', default=True)
parser.add_argument('--record_cls', default=True)
parser.add_argument('--num_classes', default=10)
parser.add_argument('--input_dim', default=())
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--cifar_imb_ratio', default=0.01)
parser.add_argument('--data_root', default='../../data')
parser.add_argument('--model', default='resnet18_noshort')
parser.add_argument('--pretrained', default=True)
parser.add_argument('--batch_size', default=64)
parser.add_argument('--criterion', default='ce', choices=['ce', 'bce', 'mse'])
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--lr', default=0.1)
parser.add_argument('--lr_decay', default=1.0)
parser.add_argument('--weight_decay', default=0.0005)
parser.add_argument('--momentum', default=0.9)

args = parser.parse_args()
args.save_root = os.path.join(args.save_root, train_save_folder(args))