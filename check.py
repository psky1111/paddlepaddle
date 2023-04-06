from models.pretrain import CVLP_r50
from clip import load_model,tokenize
import argparse
import paddle
from PIL import Image
def get_args_parser():
    parser = argparse.ArgumentParser('VL-LTR training and evaluation script', add_help=False)
    parser.add_argument('--fp32-resume', action='store_true', default=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--config', type=str, help='config')
    parser.add_argument('--pretrained-clip', default="./pretrained_models/RN50.pdparams", type=str)
    parser.add_argument('--txt-embed-path', type=str, default=None, help='config')
    parser.add_argument('--vis-backbone-path', type=str, default=None, help='config')
    parser.add_argument('--two-branch', action='store_true', help='two branch output')
    parser.set_defaults(two_branch=False)
    parser.add_argument('--debug', action='store_true', help='cls and img txt contrastive learning')
    parser.set_defaults(debug=False)

    # NLP parameters
    parser.add_argument('--desc-path', default='', type=str)
    parser.add_argument('--context-length', default=0, type=int, help='max length of text description')
    parser.add_argument('--sent-length', default=64, type=int, help='max number of selected sentences')
    parser.add_argument('--cls-token-length', default=1, type=int, help='the length of cls token')
    parser.add_argument('--loss-type', default='CE', type=str, help='loss type')
    parser.add_argument('--pretrain-cvlp', action='store_true', help='sentence-level pretraining')
    parser.set_defaults(pretrain_cvlp=False)
    parser.add_argument('--pretrain-cvlp-path', default='', type=str,
                        help='path of sentence-level pretraining task ckpt')

    # Model parameters
    parser.add_argument('--model', default='pvt_small', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--img-grad', action='store_true', default=True)
    parser.add_argument('--no-img-grad', action='store_false', dest='img_grad')
    parser.add_argument('--train-mode', action='store_true', default=True)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--text-lr', type=float, default=0, metavar='LR',
                        help='learning rate for text model (default: 0)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    parser.add_argument('--clip-ms', action='store_true', help='use clip mean & std for initialization')
    parser.set_defaults(clip_ms=False)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default=None, type=str, metavar='MODEL',
                        help='Name of teacher model to train')
    parser.add_argument('--teacher-path', type=str, default=None)
    parser.add_argument('--distillation-type', default='none', choices=['none', 'feat', 'logits', 'logits_kl'],
                        type=str, help="")
    parser.add_argument('--distillation-alpha', default=0, type=float, help="")
    parser.add_argument('--distillation-beta', default=0, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    parser.add_argument('--distillation-training-mode', action='store_true', help="")
    parser.set_defaults(distillation_training_mode=False)

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--pretrained', action='store_true')

    # Sampler Parameters
    parser.add_argument('--weight-sample', action='store_true')
    parser.add_argument('--no-weight-sample', action='store_false', dest='weight_sample')
    parser.set_defaults(weight_sample=False)
    parser.add_argument('--use-sqrt-freq', action='store_true')
    parser.set_defaults(use_sqrt_freq=False)

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['PLACES_LT', 'CIFAR', 'IMNET', 
                                                                'IMNET_LT', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')

    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output-dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only', default=False)
    parser.add_argument('--test', action='store_true', help='Perform test only', default=False)
    parser.set_defaults(test=False)
    parser.add_argument('--test-p', action='store_true', help='Calculate acc for each class', default=False)
    parser.add_argument('--select', action='store_true', help='Perform test only', default=False)
    parser.set_defaults(select=False)
    parser.add_argument('--eval-pretrain', action='store_true', help='Perform evaluation for pretraining')
    parser.set_defaults(eval_pretrain=False)
    parser.add_argument('--ensemble', action='store_true', help='Perform zero-shot evaluation for pretraining like CLIP')
    parser.set_defaults(ensemble=False)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--drop-last', action='store_true')
    parser.add_argument('--no-drop-last', action='store_false', dest='drop_last')
    parser.set_defaults(drop_last=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communication during distributed "
                             "training")
    parser.add_argument("--local_rank", default=0, type=int)
    return parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser('VL-LTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    cilp_model = load_model("RN50",pretrained=True)
    clip,transforms = cilp_model
    image = transforms(Image.open("CLIP.png")).unsqueeze(0)
    text = tokenize(["a diagram", "a dog", "a cat"])
    #logits_per_image, logits_per_text = clip(image, text)
    model = CVLP_r50(pretrained=True,args=args)
    logits_per_image, logits_per_test = model([image, text])