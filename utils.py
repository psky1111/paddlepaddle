"""
Misc functions, including distributed helpers.
Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime

import paddle
import paddle.distributed as dist
import os.path as osp
import paddle.vision as vis
from models.pretrain import CVLP_r50
from models.finetune import LGR_r50
from paddle import nn
import mmcv


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = paddle.to_tensor([self.count, self.total], dtype=paddle.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = paddle.to_tensor(list(self.deque))
        
        return paddle.median(d).item()

    @property
    def avg(self):
        d = paddle.to_tensor(list(self.deque), dtype=paddle.float32)
        return paddle.mean(d).item()

    @property
    def global_avg(self):
        if self.count == 0: return 0.0
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, paddle.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]

        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            if len(iterable) == 0:
                break
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    paddle.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        paddle.save(args[0], args[1].as_posix())


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    args.distributed = True
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}, word {}): {}'.format(
        args.rank, args.world_size, args.dist_url), flush=True)
    paddle.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    paddle.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def create_model(method = "pretrain",args=None,dataset=None):
    if args.pretrain_cvlp:
        models = CVLP_r50(pretrained=True,args=args)
    else:
        models = LGR_r50(pretrained=True,args=args,dataset=dataset)
    return models



class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = paddle.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False):
        assert clip_grad == None
        scaled = self._scaler.scale(loss)
        scaled.backward()
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

class SoftTargetCrossEntropy(nn.Layer):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
        loss = paddle.sum(-target * paddle.nn.functional.log_softmax(x, axis=-1), axis=-1)
        return paddle.mean(loss)
    
def update_from_config(args):
    cfg = mmcv.Config.fromfile(args.config)
    for _, cfg_item in cfg._cfg_dict.items():
        for k, v in cfg_item.items():
            setattr(args, k, v)
    if args.output_dir == '':
        config_name = args.config.split('/')[-1].replace('.py', '')
        args.output_dir = osp.join('checkpoints', config_name)
    if args.resume == '':
        args.resume = osp.join(args.output_dir, 'checkpoint.pth')

    return args


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.shape[1])
    batch_size = target.shape[0]
    _, pred = paddle.topk(output,maxk,1,True,True)
    pred = pred.t()
    buff = target.reshape((1, -1)).expand_as(pred)
    pred = paddle.cast(pred,buff.dtype)
    correct = paddle.equal(pred, buff)
    return [ paddle.sum(correct[:min(k, maxk)].reshape([-1]),axis=0) * 100. / batch_size for k in topk]