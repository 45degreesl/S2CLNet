import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='S2CLNet training and testing')
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--dataset', default='refsegrs', choices=['rrsisd', 'refsegrs'],
                        help='dataset adapter to use')
    parser.add_argument('--device', default='cuda:0', help='device')  # only used when testing on a single machine
    parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--img_size', default=480, type=int, help='input image size')
    parser.add_argument('--lr', default=0.00005, type=float, help='the initial learning rate')  # refsegsr: 5e-5, rrsisd: 3e-5
    parser.add_argument('--model', default='lavt_one', help='model: lavt, lavt_one')
    parser.add_argument('--model_id', default='S2CLNet', help='name to identify the model')
    parser.add_argument('--output-dir', default='./checkpoints/S2CLNet/', help='path where to save checkpoint weights')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--pretrained_clip_weights', default='./pretrained_weights/RN101.pt',
                        help='path to the OpenAI CLIP RN101 checkpoint')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--refer_data_root', default='../datasets/', help='dataset root directory')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--split', default='test', help='only used when testing')
    parser.add_argument('--splitBy', default='unc', help='dataset split convention used by the REFER adapter')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
