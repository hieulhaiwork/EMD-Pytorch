"""
Simple comprehensive test without Unicode symbols for Windows compatibility
"""

import torch
import numpy as np
import time
from emd import earth_mover_distance, EMDLoss

def test_basic_functionality():
    """Test basic EMD functionality"""
    print("=== Basic Functionality Test ===")
    
    # Simple test case
    batch_size = 2
    n_points = 100
    
    xyz1 = torch.randn(batch_size, n_points, 3).cuda()
    xyz2 = torch.randn(batch_size, n_points, 3).cuda()
    
    # Test forward pass
    cost = earth_mover_distance(xyz1, xyz2, transpose=False)
    print(f"[PASS] Forward pass successful")
    print(f"  Input shape: {xyz1.shape}")
    print(f"  Output shape: {cost.shape}")
    print(f"  Cost values: {cost}")
    
    # Test gradient computation
    xyz1.requires_grad_(True)
    xyz2.requires_grad_(True)
    cost = earth_mover_distance(xyz1, xyz2, transpose=False)
    loss = cost.sum()
    loss.backward()
    
    print(f"[PASS] Gradient computation successful")
    print(f"  xyz1 grad shape: {xyz1.grad.shape}")
    print(f"  xyz2 grad shape: {xyz2.grad.shape}")
    print(f"  xyz1 grad norm: {xyz1.grad.norm():.4f}")
    print(f"  xyz2 grad norm: {xyz2.grad.norm():.4f}")
    
    return True

def test_transpose_functionality():
    """Test transpose functionality (BCN vs BNC format)"""
    print("\n=== Transpose Functionality Test ===")
    
    batch_size = 2
    n_points = 50
    
    # BNC format (batch, num_points, channels)
    xyz1_bnc = torch.randn(batch_size, n_points, 3).cuda()
    xyz2_bnc = torch.randn(batch_size, n_points, 3).cuda()
    
    # BCN format (batch, channels, num_points)
    xyz1_bcn = xyz1_bnc.transpose(1, 2)
    xyz2_bcn = xyz2_bnc.transpose(1, 2)
    
    # Test both formats
    cost_bnc = earth_mover_distance(xyz1_bnc, xyz2_bnc, transpose=False)
    cost_bcn = earth_mover_distance(xyz1_bcn, xyz2_bcn, transpose=True)
    
    print(f"[PASS] BNC format cost: {cost_bnc}")
    print(f"[PASS] BCN format cost: {cost_bcn}")
    print(f"[PASS] Difference: {(cost_bnc - cost_bcn).abs().max():.6f}")
    
    # They should be very close (within numerical precision)
    assert torch.allclose(cost_bnc, cost_bcn, rtol=1e-5), "BNC and BCN results should match"
    print("[PASS] Transpose functionality working correctly")
    
    return True

def test_emd_loss_module():
    """Test EMDLoss as a PyTorch module"""
    print("\n=== EMDLoss Module Test ===")
    
    # Create EMDLoss module
    emd_loss = EMDLoss(transpose=False).cuda()
    
    batch_size = 2
    n_points = 80
    
    xyz1 = torch.randn(batch_size, n_points, 3).cuda().requires_grad_(True)
    xyz2 = torch.randn(batch_size, n_points, 3).cuda().requires_grad_(True)
    
    # Forward pass
    loss = emd_loss(xyz1, xyz2)
    
    print(f"[PASS] EMDLoss module forward pass successful")
    print(f"  Loss shape: {loss.shape}")
    print(f"  Loss values: {loss}")
    
    # Backward pass
    total_loss = loss.sum()
    total_loss.backward()
    
    print(f"[PASS] EMDLoss module backward pass successful")
    print(f"  xyz1 grad norm: {xyz1.grad.norm():.4f}")
    print(f"  xyz2 grad norm: {xyz2.grad.norm():.4f}")
    
    return True

def test_performance():
    """Test performance with larger point clouds"""
    print("\n=== Performance Test ===")
    
    batch_size = 4
    n_points = 1000
    
    xyz1 = torch.randn(batch_size, n_points, 3).cuda()
    xyz2 = torch.randn(batch_size, n_points, 3).cuda()
    
    # Warmup
    for _ in range(3):
        _ = earth_mover_distance(xyz1, xyz2, transpose=False)
    
    # Timing
    torch.cuda.synchronize()
    start_time = time.time()
    
    n_iterations = 10
    for _ in range(n_iterations):
        cost = earth_mover_distance(xyz1, xyz2, transpose=False)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_iterations
    
    print(f"[PASS] Performance test completed")
    print(f"  Point clouds: {batch_size} x {n_points} points")
    print(f"  Average time per forward pass: {avg_time:.4f} seconds")
    print(f"  Throughput: {batch_size * n_points / avg_time:.0f} points/second")
    
    return True

def main():
    """Run all tests"""
    print("EMD CUDA Extension Comprehensive Test")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    
    tests = [
        test_basic_functionality,
        test_transpose_functionality,
        test_emd_loss_module,
        test_performance,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print("[PASS] Test completed successfully\n")
        except Exception as e:
            failed += 1
            print(f"[FAIL] Test failed: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All tests passed! EMD CUDA extension is working perfectly!")
    else:
        print("Some tests failed. Please check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)