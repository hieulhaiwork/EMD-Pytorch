import torch
import os
import sys
import importlib.util

def _try_import_emd_cuda():
    """Try multiple ways to import emd_cuda"""
    
    # Method 1: Direct import
    try:
        import emd_cuda
        return emd_cuda
    except ImportError:
        pass
    
    # Method 2: Add current directory to path
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        import emd_cuda
        return emd_cuda
    except ImportError:
        pass
    
    # Method 3: Load .pyd directly
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pyd_files = [f for f in os.listdir(current_dir) if f.startswith('emd_cuda') and f.endswith('.pyd')]
        
        if pyd_files:
            pyd_file = pyd_files[0]
            pyd_path = os.path.join(current_dir, pyd_file)
            
            spec = importlib.util.spec_from_file_location("emd_cuda", pyd_path)
            emd_cuda = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(emd_cuda)
            return emd_cuda
    except Exception:
        pass
    
    raise ImportError("Failed to import emd_cuda. Please make sure the CUDA extension is properly compiled.")

# Load the CUDA extension
try:
    emd_cuda = _try_import_emd_cuda()
except ImportError as e:
    raise ImportError(f"{e}\n\nTo fix this issue:\n"
                     "1. Make sure CUDA is installed and working\n"
                     "2. Run: python setup.py build_ext --inplace\n"
                     "3. Check that PyTorch CUDA version matches your system CUDA")


class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd_cuda.approxmatch_forward(xyz1, xyz2)
        cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = emd_cuda.matchcost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2


def earth_mover_distance(xyz1, xyz2, transpose=True):
    """Earth Mover Distance (Approx)

    Args:
        xyz1 (torch.Tensor): (b, 3, n1)
        xyz2 (torch.Tensor): (b, 3, n1)
        transpose (bool): whether to transpose inputs as it might be BCN format.
            Extensions only support BNC format.

    Returns:
        cost (torch.Tensor): (b)

    """
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1 = xyz1.transpose(1, 2)
        xyz2 = xyz2.transpose(1, 2)
    cost = EarthMoverDistanceFunction.apply(xyz1, xyz2)
    return cost


# Aliases for compatibility
EMDFunction = EarthMoverDistanceFunction
EMDLoss = earth_mover_distance


class EMDLoss(torch.nn.Module):
    """EMD Loss as a PyTorch Module"""
    
    def __init__(self, transpose=True):
        super(EMDLoss, self).__init__()
        self.transpose = transpose
    
    def forward(self, xyz1, xyz2):
        return earth_mover_distance(xyz1, xyz2, transpose=self.transpose)

