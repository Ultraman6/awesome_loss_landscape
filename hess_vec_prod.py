import torch
import time
import numpy as np
from sympy.physics.units import length
from torch.autograd import Variable
from scipy.sparse.linalg import LinearOperator, eigsh
from tqdm import tqdm


################################################################################
#                              Supporting Functions
################################################################################
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
    res = np.concatenate([p.grad.data.cpu().numpy().ravel() for p in net.parameters() if filter(p)])
    return res


################################################################################
#                  For computing Hessian-vector products
################################################################################
def eval_hess_vec_prod(vec, params, net, criterion, dataloader, device='cuda'):
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
    net.to(device).eval().zero_grad()
    vec = [v.to(device) for v in vec]

    for inputs, targets in tqdm(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True)
        prod = torch.zeros(1, dtype=grad_f[0].data.dtype).to(grad_f[0].device)
        for (g, v) in zip(grad_f, vec):
            prod = prod + (g * v).sum()
        prod.backward()

################################################################################
#                  For computing Eigenvalues of Hessian
################################################################################
def min_max_hessian_eigs(net, dataloader, criterion, top=1, device='cuda', verbose=False):
    """
        Compute the largest and the smallest eigenvalues of the Hessian marix.

        Args:
            net: the trained model.
            dataloader: dataloader for the dataset, may use a subset of it.
            criterion: loss function.
            rank: rank of the working node.
            use_cuda: use GPU
            verbose: print more information

        Returns:
            maxeig: max eigenvalue
            mineig: min eigenvalue
            hess_vec_prod.count: number of iterations for calculating max and min eigenvalues
    """
    params = [p for p in net.parameters() if len(p.size()) > 1]
    N = sum(p.numel() for p in params)

    def hess_vec_prod(vec):
        hess_vec_prod.count += 1  # simulates a static variable
        vec = npvec_to_tensorlist(vec, params)
        start_time = time.time()
        eval_hess_vec_prod(vec, params, net, criterion, dataloader, device)
        prod_time = time.time() - start_time
        if verbose: print("   Iter: %d  time: %f" % (hess_vec_prod.count, prod_time))
        return gradtensor_to_npvec(net)

    hess_vec_prod.count = 0
    if verbose: print("computing max eigenvalue")
    # 模型维度空间的 HVP 近似 hessian
    A = LinearOperator((N, N), matvec=hess_vec_prod)
    eigvals, eigvecs = eigsh(A, k=top, tol=1e-2)
    for i, v in enumerate(eigvals):
        if verbose: print(f'max {i}th eigenvalue = %f' % eigvals[i])
    # If the largest eigenvalue is positive, shift matrix so that any negative eigenvalue is now the largest
    # We assume the smallest eigenvalue is zero or less, and so this shift is more than what we need
    # shift = maxeig*.51
    # def shifted_hess_vec_prod(vec):
    #     return hess_vec_prod(vec) - shift*vec
    # if verbose and rank == 0: print("Rank %d: Computing shifted eigenvalue" % rank)
    # A = LinearOperator((N, N), matvec=shifted_hess_vec_prod)
    #
    # eigvals, eigvecs = eigsh(A, k=1, tol=1e-2)
    # eigvals = eigvals + shift
    # mineig = eigvals[0]
    # if verbose and rank == 0: print('min eigenvalue = ' + str(mineig))
    #
    # if maxeig <= 0 and mineig > 0:
    #     maxeig, mineig = mineig, maxeig
    return eigvals, eigvecs

