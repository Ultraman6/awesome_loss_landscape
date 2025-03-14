from pytorch_lightning.callbacks import Callback
import torch
import time
import numpy as np
from torch.autograd import Variable
from scipy.sparse.linalg import LinearOperator, eigsh

class CustomCallback(Callback):
    def __init__(self, start_epoch, checkpoint_callback):
        self.start_epoch = start_epoch
        self.checkpoint_callback = checkpoint_callback

    def on_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch >= self.start_epoch:
            self.checkpoint_callback.on_epoch_end(trainer, pl_module)

def npvec_to_tensorlist(vec, params):
    """ Convert a numpy vector to a list of tensor with the same dimensions as params

        Args:
            vec: a 1D numpy vector
            params: a list of parameters from net

        Returns:
            rval: a list of tensors with the same shape as params
    """
    loc = 0
    rval = []
    for p in params:
        numel = p.data.numel()
        rval.append(torch.from_numpy(vec[loc:loc+numel]).view(p.data.shape).float())
        loc += numel
    assert loc == vec.size, 'The vector has more elements than the net has parameters'
    return rval


def gradtensor_to_npvec(net, include_bn=False):
    """ Extract gradients from net, and return a concatenated numpy vector.

        Args:
            net: trained model
            include_bn: If include_bn, then gradients w.r.t. BN parameters and bias
            values are also included. Otherwise only gradients with dim > 1 are considered.

        Returns:
            a concatenated numpy vector containing all gradients
    """
    filter = lambda p: include_bn or len(p.data.size()) > 1
    return np.concatenate([p.grad.data.cpu().numpy().ravel() for p in net.parameters() if filter(p)])


################################################################################
#                  For computing Hessian-vector products
################################################################################
def eval_hess_vec_prod(vec, params, net, criterion, dataloader, use_cuda=True):
    """
    Evaluate product of the Hessian of the loss function with a direction vector "vec".
    The product result is saved in the grad of net.

    Args:
        vec: a list of tensor with the same dimensions as "params".
        params: the parameter list of the net (ignoring biases and BN parameters).
        net: model with trained parameters.
        criterion: loss function.
        dataloader: dataloader for the dataset.
        use_cuda: use GPU.
    """

    if use_cuda:
        net.cuda()
        vec = [v.cuda() for v in vec]

    net.eval()
    net.zero_grad() # clears grad for every parameter in the net

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True)

        # Compute inner product of gradient with the direction vector
        prod = Variable(torch.zeros(1)).type(type(grad_f[0].data))
        for (g, v) in zip(grad_f, vec):  # 权重参数扰动
            prod = prod + (g * v).cpu().sum()

        # Compute the Hessian-vector product, H*v
        # prod.backward() computes dprod/dparams for every parameter in params and
        # accumulate the gradients into the params.grad attributes
        prod.backward()
        # 并没有 zero_grad，累计所有样本的梯度

################################################################################
#                  For computing Eigenvalues of Hessian
################################################################################
def min_max_hessian_eigs(net, dataloader):

    params = [p for p in net.parameters() if len(p.size()) > 1]
    N = sum(p.numel() for p in params)

    def hess_vec_prod(vec):
        hess_vec_prod.count += 1  # simulates a static variable
        vec = npvec_to_tensorlist(vec, params)
        start_time = time.time()
        eval_hess_vec_prod(vec, params, net, net.loss_fn, dataloader)
        prod_time = time.time() - start_time
        print("Iter: %d  time: %f" % (hess_vec_prod.count, prod_time))
        return gradtensor_to_npvec(net)

    hess_vec_prod.count = 0
    # HVP 方法，随机向量内积得到近似的 hessian，做奇异分解得到奇异值与奇异向量
    A = LinearOperator((N, N), matvec=hess_vec_prod)
    eigvals, eigvecs = eigsh(A, k=1, tol=1e-2)
    maxeig = eigvals[0]
    print('max eigenvalue = %f' % maxeig)

    # If the largest eigenvalue is positive, shift matrix so that any negative eigenvalue is now the largest
    # We assume the smallest eigenvalue is zero or less, and so this shift is more than what we need
    shift = maxeig*.51
    def shifted_hess_vec_prod(vec):
        return hess_vec_prod(vec) - shift*vec

    A = LinearOperator((N, N), matvec=shifted_hess_vec_prod)
    eigvals, eigvecs = eigsh(A, k=1, tol=1e-2)
    eigvals = eigvals + shift
    mineig = eigvals[0]
    print('min eigenvalue = ' + str(mineig))

    if maxeig <= 0 and mineig > 0:
        maxeig, mineig = mineig, maxeig

    return maxeig, mineig, hess_vec_prod.count
