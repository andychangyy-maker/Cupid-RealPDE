"""Two-pass L2 SAM with identical stochastic inputs and exact weight restoration."""
import torch


def matched_sam_backward(backward, model, *args, rho, **kwargs):
    if rho == 0:
        return backward(model, *args, **kwargs)
    if rho != .05:
        raise ValueError('this experiment fixes SAM rho to .05')
    cuda = any(p.is_cuda for p in model.parameters())
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state_all() if cuda else []
    first = backward(model, *args, **kwargs)
    cpu_after = torch.get_rng_state()
    cuda_after = torch.cuda.get_rng_state_all() if cuda else []
    parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in parameters]))
    if not torch.isfinite(norm) or norm <= 0:
        raise FloatingPointError('SAM first gradient must be finite and nonzero')
    originals = [p.detach().clone() for p in parameters]
    scale = rho / norm
    try:
        with torch.no_grad():
            for p in parameters:
                p.add_(p.grad * scale)
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_before)
        if cuda:
            torch.cuda.set_rng_state_all(cuda_before)
        second = backward(model, *args, **kwargs)
        if not torch.equal(torch.get_rng_state(), cpu_after):
            raise RuntimeError('SAM passes consumed different CPU RNG streams')
        if cuda and not all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda_after)):
            raise RuntimeError('SAM passes consumed different CUDA RNG streams')
    finally:
        with torch.no_grad():
            for p, original in zip(parameters, originals):
                p.copy_(original)
        torch.set_rng_state(cpu_after)
        if cuda:
            torch.cuda.set_rng_state_all(cuda_after)
    second.update(sam_unperturbed_loss=float(first['real_loss']),
                  sam_perturbed_loss=float(second['real_loss']),
                  sam_first_gradient_norm=float(norm), sam_rho=rho)
    return second
