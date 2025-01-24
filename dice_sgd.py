import torch
from opacus.optimizers.optimizer import DPOptimizer

class DiceSGD(DPOptimizer):
    """
    A DP-ified DiceSGD optimizer that follows the pseudocode:

        v^t = (1/B) * [ sum_i clip(g_i^t, C1) ] + clip(e^t, C2)
        x^{t+1} = x^t - eta^t [ v^t + noise ]
        e^{t+1} = e^t + (1/B)*sum_i g_i^t - v^t

    where g_i^t = ∇f(x^t; ξ_i), and C1, C2 are clipping thresholds for
    the per-sample gradients and for the error feedback buffer, respectively.
    """
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            *,
            noise_multiplier: float,
            max_grad_norm: float,  # C1
            max_ef_norm: float,    # C2
            expected_batch_size: int,
            **kwargs
    ):
        """
        :param optimizer: A base torch optimizer (e.g. SGD) whose step we'll override
        :param noise_multiplier: The std of the noise relative to max_grad_norm
        :param max_grad_norm: Clipping bound for the per-sample gradient (C1)
        :param max_ef_norm: Clipping bound for the error feedback buffer (C2)
        :param expected_batch_size: Batch size B (used in noise scaling, etc.)
        """
        super().__init__(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            **kwargs
        )
        self.max_ef_norm = max_ef_norm

        # We store a buffer e^t (error feedback) per parameter
        for pg in self.param_groups:
            for p in pg["params"]:
                self.state[p]["e_buffer"] = torch.zeros_like(p, device=p.device)

    def _clip_vector(self, vec: torch.Tensor, max_norm: float) -> torch.Tensor:
        """
        Helper: Clips `vec` to have norm at most `max_norm`.
        """
        norm = vec.norm()
        if norm > max_norm:
            vec = vec * (max_norm / (norm + 1e-6))
        return vec

    def step(self, closure=None):
        """
        Perform one DiceSGD step:

          1) For each parameter p:
             - We have per-sample grads in p.grad_sample of shape [batch_size, ...]
             - Clip them per-sample to norm C1, then sum over batch & average
          2) Let e^t be the error buffer from the previous step
             Clip e^t to norm C2, add it to the clipped gradient to get v^t
          3) Add noise if noise_multiplier>0
          4) Update parameters: x^{t+1} = x^t - lr * (v^t + noise)
          5) Update e^{t+1} = e^t + [avg g_i^t - v^t]
        """
        loss = None
        if closure is not None:
            loss = closure()

        # We'll assume the batch size is consistent:
        batch_size = self.expected_batch_size

        for pg in self.param_groups:
            lr = pg["lr"]
            for p in pg["params"]:
                # If no gradients have been computed for this param, skip
                if not hasattr(p, "grad_sample") or p.grad_sample is None:
                    continue

                # p.grad_sample: [batch_size, ...]
                grad_samples = p.grad_sample

                # 1) Clip each sample in grad_samples to norm <= C1, then average
                per_sample_norms = grad_samples.view(batch_size, -1).norm(2, dim=1)
                factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(max=1.0)
                # Reshape factor so it can broadcast
                factor = factor.view([-1] + [1]*(grad_samples.dim()-1))
                grad_samples = grad_samples * factor
                grad_mean = grad_samples.sum(dim=0) / batch_size

                # 2) Add clipped error buffer e^t
                e_buf = self.state[p]["e_buffer"]
                e_clipped = self._clip_vector(e_buf, self.max_ef_norm)
                v = grad_mean + e_clipped  # v^t

                # 3) Add DP noise if needed
                if self.noise_multiplier > 0:
                    # Standard DP-SGD noise has std = noise_multiplier * max_grad_norm
                    # We also scale by 1 / batch_size (like p.grad was)
                    noise = torch.randn_like(v) * (
                            self.noise_multiplier * self.max_grad_norm / batch_size
                    )
                else:
                    noise = 0.0

                # 4) Update x: x^{t+1} = x^t - lr * (v + noise)
                p.data = p.data - lr * (v + noise)

                # 5) Update error buffer: e^{t+1} = e^t + [ grad_mean - v ]
                self.state[p]["e_buffer"] = e_buf + (grad_mean - v)

                # Clear out per-sample grads so they won't accumulate
                p.grad_sample = None

        return loss