"""Install a PyTorch allocation ceiling before model construction.

This bounds PyTorch's CUDA allocator, not driver/library allocations or other
processes. Free-memory headroom is a startup check, not a system reservation.
"""
import math


def budget_bytes(total, free, limit_gib, headroom_gib):
    """Use the smaller of the requested ceiling and available startup budget."""
    if not math.isfinite(limit_gib) or limit_gib <= 0:
        raise ValueError('GPU memory limit must be finite and positive')
    if not math.isfinite(headroom_gib) or headroom_gib < 0:
        raise ValueError('GPU headroom must be finite and nonnegative')
    budget = min(int(limit_gib * 1024**3), int(free-headroom_gib * 1024**3), total)
    if budget <= 0:
        raise RuntimeError('Insufficient free GPU memory for the requested headroom; model not loaded')
    return budget


def install_budget(torch, limit_gib=5.0, headroom_gib=1.0):
    """Set and verify the allocator ceiling on the current CUDA device once."""
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    budget = budget_bytes(total, free, limit_gib, headroom_gib)
    if torch.cuda.memory_reserved(device) >= budget:
        raise RuntimeError('Existing PyTorch reservations already reach the GPU budget; model not loaded')
    fraction = budget / total
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    observed = torch.cuda.get_per_process_memory_fraction(device)
    if not math.isclose(observed, fraction, rel_tol=1e-6, abs_tol=1e-9):
        raise RuntimeError('GPU allocator ceiling could not be verified; model not loaded')
    report = dict(requested_gib=limit_gib, headroom_gib=headroom_gib,
                  effective_gib=budget/1024**3, total_gib=total/1024**3,
                  startup_free_gib=free/1024**3, fraction=fraction,
                  device_index=device, scope='PyTorch CUDA allocator')
    print(f'[gpu-budget] PyTorch cap={report["effective_gib"]:.2f} GiB; '
          f'GPU total={report["total_gib"]:.2f} GiB; '
          f'startup free={report["startup_free_gib"]:.2f} GiB', flush=True)
    return report


def smoke_test():
    gib = 1024**3
    assert budget_bytes(8*gib, 7*gib, 5, 1) == 5*gib
    assert budget_bytes(8*gib, 4*gib, 5, 1) == 3*gib
    for limit in (0, -1, float('nan'), float('inf')):
        try:
            budget_bytes(8*gib, 7*gib, limit, 1)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid limit accepted')
    print('GPU budget smoke test passed (no GPU allocation)')


if __name__ == '__main__':
    smoke_test()
