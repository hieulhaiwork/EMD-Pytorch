// emd.cpp
#ifndef _EMD
#define _EMD

#include <vector>
#include <torch/extension.h>

// CUDA declarations
torch::Tensor ApproxMatchForward(
    const torch::Tensor xyz1,
    const torch::Tensor xyz2);

torch::Tensor MatchCostForward(
    const torch::Tensor xyz1,
    const torch::Tensor xyz2,
    const torch::Tensor match);

std::vector<torch::Tensor> MatchCostBackward(
    const torch::Tensor grad_cost,
    const torch::Tensor xyz1,
    const torch::Tensor xyz2,
    const torch::Tensor match);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("approxmatch_forward", &ApproxMatchForward, "ApproxMatch forward (CUDA)");
  m.def("matchcost_forward", &MatchCostForward, "MatchCost forward (CUDA)");
  m.def("matchcost_backward", &MatchCostBackward, "MatchCost backward (CUDA)");
}

#endif
