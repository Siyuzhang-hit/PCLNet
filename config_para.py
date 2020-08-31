# coding=utf-8
import argparse
import socket
import os

LABEL_NAME = ("building", "rapeseed", "beet", "stembeans", "peas", "forest", "lucerne", "potatoes", "baresoil",
              "grasses", "barley", "water", "wheatone", "wheattwo", "wheatthree")

color_value = [[255, 255, 255],
                [255, 200, 128],
                [255, 128, 0],
                [128, 0, 128],
                [255, 0, 0],
                [128, 0, 255],
                [0, 128, 0],
                [0, 255, 255],
                [255, 255, 0],
                [128, 128, 0],
                [0, 255, 0],
                [128, 0, 0],
                [0, 0, 255],
                [255, 128, 255],
                [128, 128, 255],
                [128, 255, 128]]


def parse_option_train():

    hostname = socket.gethostname()

    argparser = argparse.ArgumentParser('argument for training')

    argparser.add_argument('--print_freq', type=int, help='print frequency', default=10)
    argparser.add_argument('--tb_freq', type=int, help='tb frequency', default=10)
    argparser.add_argument('--save_freq', type=int, help='pretrain save frequency', default=10)
    argparser.add_argument('--batch_size', type=int, help='batch_size', default=32)
    argparser.add_argument('--num_workers', type=int, help='num of workers to use', default=2)
    argparser.add_argument('--epochs', type=int, help='pretrain number of training epochs', default=800)
    argparser.add_argument('--set_num_threads', type=int, help='cpu', default=1)

    # optimization
    argparser.add_argument('--learning_rate', type=float, help='learning rate', default=0.1)
    argparser.add_argument('--lr_decay_epochs', type=str, help='where to decay lr, can be a list', default='300,500')
    argparser.add_argument('--lr_decay_rate', type=float, help='decay rate for learning rate', default=0.5)
    argparser.add_argument('--beta1', type=float, help='beta1 for adam', default=0.5)
    argparser.add_argument('--beta2', type=float, help='beta2 for Adam', default=0.999)
    argparser.add_argument('--weight_decay', type=float, help='weight decay', default=1e-4)
    argparser.add_argument('--momentum', type=float, help='momentum', default=0.9)

    # resume
    argparser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # loss function
    argparser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    argparser.add_argument('--nce_k', type=int, default=8192)
    argparser.add_argument('--nce_t', type=float, default=0.4)

    # memory setting
    argparser.add_argument('--alpha', type=float, default=0.999, help='exponential moving average weight')

    argparser.add_argument('--stride', type=int, help='classification stride', default=1)
    argparser.add_argument('--character_num', type=int, help='character_num', default=9)
    argparser.add_argument('--patch_size', type=int, help='patch_size', default=15)
    argparser.add_argument('--outDim', type=int, help='feature number', default=32)

    args = argparser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    args.method = 'softmax' if args.softmax else 'nce'
    prefix = 'PCLNet{}'.format(args.alpha)

    if hostname.startswith('Mywork'):
        args.data_folder = ''
        args.model_path = ''
        args.tb_path = ''
    else:
        raise NotImplementedError('server invalid: {}'.format(hostname))

    args.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_{}'.format(prefix, args.method, args.nce_k, args.learning_rate,
                                                                   args.weight_decay, args.batch_size, args.nce_t)

    args.model_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    args.tb_folder = os.path.join(args.tb_path, args.model_name)
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    return args


def parse_option_eval():

    hostname = socket.gethostname()

    argparser = argparse.ArgumentParser('argument for evalution')

    argparser.add_argument('--print_freq', type=int, help='print frequency', default=10)
    argparser.add_argument('--tb_freq', type=int, help='tb frequency', default=10)
    argparser.add_argument('--save_freq', type=int, help='finetune save frequency', default=10)
    argparser.add_argument('--batch_size', type=int, help='batch_size', default=32)
    argparser.add_argument('--num_workers', type=int, help='num of workers to use', default=2)
    argparser.add_argument('--epochs', type=int, help='finetune number of training epochs', default=500)
    argparser.add_argument('--set_num_threads', type=int, help='cpu', default=1)

    # optimization
    argparser.add_argument('--learning_rate', type=float, help='learning rate', default=0.01)
    argparser.add_argument('--lr_decay_epochs', type=str, help='where to decay lr, can be a list', default='300')
    argparser.add_argument('--lr_decay_rate', type=float, help='decay rate for learning rate', default=0.1)
    argparser.add_argument('--beta1', type=float, help='beta1 for adam', default=0.5)
    argparser.add_argument('--beta2', type=float, help='beta2 for Adam', default=0.999)
    argparser.add_argument('--weight_decay', type=float, help='weight decay', default=0)
    argparser.add_argument('--momentum', type=float, help='momentum', default=0.9)

    # model definition
    argparser.add_argument('--model_path', type=str, default='', help='the model to test')
    argparser.add_argument('--layer', type=int, default=3, help='which layer to evaluate')

    # resume
    argparser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    argparser.add_argument('--adam', action='store_true', help='use adam optimizer')

    argparser.add_argument('--stride', type=int, help='classification stride', default=1)
    argparser.add_argument('--character_num', type=int, help='character_num', default=9)
    argparser.add_argument('--patch_size', type=int, help='patch_size', default=15)
    argparser.add_argument('--label', type=int, help='label number', default=15)

    args = argparser.parse_args()
    args.label_name = LABEL_NAME

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    if hostname.startswith('Mywork'):
        args.train_data_folder = ''
        args.valid_data_folder = ''
        args.save_path = ''
        args.tb_path = ''

    args.model_name = args.model_path.split('/')[-2]
    args.model_name = '{}_bsz_{}_lr_{}_decay_{}'.format(args.model_name, args.batch_size, args.learning_rate, args.weight_decay)

    if args.adam:
        args.model_name = '{}_useAdam'.format(args.model_name)

    args.save_folder = os.path.join(args.save_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    args.tb_folder = os.path.join(args.tb_path, args.model_name)
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    return args


def parse_option_test():
    hostname = socket.gethostname()

    argparser = argparse.ArgumentParser('argument for test')

    argparser.add_argument('--batch_size', type=int, help='batch_size', default=32)
    argparser.add_argument('--num_workers', type=int, help='num_workers', default=2)
    argparser.add_argument('--set_num_threads', type=int, help='cpu', default=1)
    argparser.add_argument('--stride', type=int, help='classification stride', default=1)
    argparser.add_argument('--update_lr', type=float, help='learning rate', default=0.01)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=500)
    argparser.add_argument('--character_num', type=int, help='character_num', default=9)
    argparser.add_argument('--patch_size', type=int, help='patch_size', default=15)
    argparser.add_argument('--char_height', type=int, help='char_height', default=750)
    argparser.add_argument('--char_width', type=int, help='char_width', default=1024)
    argparser.add_argument('--label', type=int, help='label number', default=15)
    argparser.add_argument('--layer', type=int, default=3, help='which layer to evaluate')

    argparser.add_argument('--test_path', type=str, help='test_path',
                           default='')
    argparser.add_argument('--model_path', type=str, default='', help='the model to test')
    argparser.add_argument('--linear_path', type=str, default='', help='the linear classifier to test')
    argparser.add_argument('--confusion_path', type=str, help='classification result', default='')
    argparser.add_argument('--cha_path', type=str, help='cha_path', default='')
    argparser.add_argument('--result_path', type=str, help='classification result', default='')

    args = argparser.parse_args()
    args.label_name = LABEL_NAME
    args.color_value = color_value
    if hostname.startswith('Mywork'):
        args.data_folder = ''
    return args
