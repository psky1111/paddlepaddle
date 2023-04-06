import paddle
import paddle.distributed as dist
from paddle.autograd import PyLayer

class GatherLayer(PyLayer):
    '''
        Gather tensors from all process, support backward propagation.
    '''

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [paddle.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)
    
    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = paddle.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()] * dist.get_world_size()
        return 