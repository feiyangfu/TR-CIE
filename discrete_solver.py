from contextlib import nullcontext
from math import ceil
from typing import Callable, Optional, Union

import os
import torch
from torch import Tensor

from torch.nn import functional as F

from flow_matching.path import MixtureDiscreteProbPath

from flow_matching.solver.solver import Solver
from flow_matching.utils import categorical, ModelWrapper
from .utils import get_nearest_times
import math
from contextlib import nullcontext
from math import ceil
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm


try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class TRCIESolver(Solver):

    def __init__(
        self,
        model: ModelWrapper,
        path: MixtureDiscreteProbPath,
        vocabulary_size: int,
        source_distribution_p: Optional[Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.path = path
        self.vocabulary_size = vocabulary_size
        self.source_distribution_p = source_distribution_p

    def _calculate_velocity(self, x_current: Tensor, p_1t_pred: Tensor, t: Tensor, div_free_t: float) -> Tensor:
        scheduler_output = self.path.scheduler(t=t)
        k_t = scheduler_output.alpha_t
        d_k_t = scheduler_output.d_alpha_t
        u = d_k_t / (1 - k_t) * p_1t_pred 
        
        if div_free_t > 0 and self.source_distribution_p is not None:
            p_0 = self.source_distribution_p[(None,) * x_current.dim()].to(u.device)
            u = u + div_free_t * d_k_t / (k_t * (1 - k_t)) * (
                (1 - k_t) * p_0 + k_t * p_1t_pred
            )

        delta_t = F.one_hot(x_current, num_classes=self.vocabulary_size)
        u = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(u), u)
        return u

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        div_free: Union[float, Callable[[float], float]] = 0.0,
        dtype_categorical: torch.dtype = torch.float32,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        final_step_denoise: bool = True,
        verbose: bool = False,
        **model_extras,
    ) -> Tensor:

        if not div_free == 0.0:
            assert self.source_distribution_p is not None
        time_grid = time_grid.to(device=x_init.device)
        
        if step_size is None:
            t_discretization = time_grid
            n_steps = len(time_grid) - 1
        else:
            t_init = time_grid[0].item()
            t_final = time_grid[-1].item()
            n_steps = ceil((t_final - t_init) / step_size)
            t_discretization = torch.linspace(t_init, t_final, n_steps + 1, device=x_init.device)
            
            if return_intermediates:
                # Logic to map intermediates if needed...
                pass 

        x_t = x_init.clone()
        steps_counter = 0
        res = [x_init.clone()] if return_intermediates else []

        if verbose:
            ctx = tqdm(total=n_steps, desc=f"NFE: {steps_counter}")
        else:
            ctx = nullcontext()

        u_tau_prev = None 
        delta_tau_prev = None

        with ctx:
            for i in range(n_steps):
                t_curr = t_discretization[i:i+1]
                t_next = t_discretization[i+1:i+2]
            
                term_curr = 1.0 - t_curr.item()
                term_next = max(1.0 - t_next.item(), 1e-10) # Safety epsilon
                
                delta_tau_64 = torch.log(torch.tensor(term_curr / term_next, device=x_init.device, dtype=torch.float64))
                delta_tau = delta_tau_64.to(dtype=torch.float32)

                p_1t_current = self.model(x=x_t, t=t_curr.repeat(x_t.shape[0]), **model_extras)
                div_free_current = div_free(t_curr.item()) if callable(div_free) else div_free
                
                u_t_current = self._calculate_velocity(x_t, p_1t_current, t_curr, div_free_current)
                
                u_tau_current = u_t_current * term_curr
                
                if i == 0:

                    lambda_coeff = 0.0

                elif t_curr.item() > 0.99:

                    lambda_coeff = 0.0

                else:
                    # r = h_n / h_{n-1}
                    r = (delta_tau / delta_tau_prev).item()
                    #  lambda = r / 2
                    lambda_coeff = (r / 2.0)
                
                # Apply Extrapolation
                # u_pred = (1 + lambda) * u_curr - lambda * u_prev
                if lambda_coeff == 0.0:
                    u_tau_used = u_tau_current
                else:
                    
                    u_tau_used = (1 + lambda_coeff) * u_tau_current - lambda_coeff * u_tau_prev
                
                u_tau_used = u_tau_used.clamp(min=0)
                
                intensity = u_tau_used.sum(dim=-1) # [batch_size]
                
                prob_jump = 1.0 - torch.exp(-delta_tau * intensity)
                mask_jump = torch.rand_like(x_t, dtype=torch.float32) < prob_jump

                
                if mask_jump.any():
                    active_u = u_tau_used[mask_jump]
                    
                    active_u_sum = active_u.sum(dim=-1, keepdim=True)
                    transition_probs = active_u / (active_u_sum + 1e-10)
                    
                    new_states = torch.distributions.Categorical(transition_probs.to(dtype=dtype_categorical)).sample()
                    x_t[mask_jump] = new_states

                u_tau_prev = u_tau_current.detach() 
                delta_tau_prev = delta_tau
                steps_counter += 1

                if return_intermediates:
                    res.append(x_t.clone())
                if verbose:
                    ctx.set_description(f"NFE: {steps_counter}")
                    ctx.update(1)

        if final_step_denoise and n_steps > 0:
            final_t = t_discretization[-1:]
            final_p1 = self.model(x=x_t, t=final_t.repeat(x_t.shape[0]), **model_extras)
            x_t = torch.distributions.Categorical(final_p1.to(dtype=dtype_categorical)).sample()
            
            steps_counter += 1
            if verbose: ctx.set_description(f"NFE: {steps_counter}")


        if return_intermediates:
            return torch.stack(res, dim=0)
        else:
            return x_t