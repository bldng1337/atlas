# This is taken from https://github.com/ostris/ai-toolkit/blob/main/toolkit/optimizers/adam8bit.py
import math
import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch import Tensor
from typing import Optional


def compute_scale_for_dtype(tensor, dtype):
    """
    Compute appropriate scale for the given tensor and target dtype.

    Args:
        tensor: Input tensor to be quantized
        dtype: Target dtype for quantization
    Returns:
        Appropriate scale factor for the quantization
    """
    if dtype == torch.int8:
        abs_max = torch.max(torch.abs(tensor))
        return abs_max / 127.0 if abs_max > 0 else 1.0
    elif dtype == torch.uint8:
        max_val = torch.max(tensor)
        min_val = torch.min(tensor)
        range_val = max_val - min_val
        return range_val / 255.0 if range_val > 0 else 1.0
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # For float8, we typically want to preserve the magnitude of the values
        # while fitting within the representable range of the format
        abs_max = torch.max(torch.abs(tensor))
        if dtype == torch.float8_e4m3fn:
            # e4m3fn has range [-448, 448] with no infinities
            max_representable = 448.0
        else:  # torch.float8_e5m2
            # e5m2 has range [-57344, 57344] with infinities
            max_representable = 57344.0

        return abs_max / max_representable if abs_max > 0 else 1.0
    else:
        raise ValueError(f"Unsupported dtype for quantization: {dtype}")

def quantize_tensor(tensor, dtype):
    """
    Quantize a floating-point tensor to the target dtype with appropriate scaling.

    Args:
        tensor: Input tensor (float)
        dtype: Target dtype for quantization
    Returns:
        quantized_data: Quantized tensor
        scale: Scale factor used
    """
    scale = compute_scale_for_dtype(tensor, dtype)

    if dtype == torch.int8:
        quantized_data = torch.clamp(torch.round(tensor / scale), -128, 127).to(dtype)
    elif dtype == torch.uint8:
        quantized_data = torch.clamp(torch.round(tensor / scale), 0, 255).to(dtype)
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # For float8, we scale and then cast directly to the target type
        # The casting operation will handle the appropriate rounding
        scaled_tensor = tensor / scale
        quantized_data = scaled_tensor.to(dtype)
    else:
        raise ValueError(f"Unsupported dtype for quantization: {dtype}")

    return quantized_data, scale


def update_parameter(target, result_float):
    """
    Updates a parameter tensor, handling both regular torch.Tensor and QBytesTensor cases
    with proper rescaling for quantized tensors.

    Args:
        target: The parameter to update (either torch.Tensor or QBytesTensor)
        result_float: The new values to assign (torch.Tensor)
    """
    # We dont need QBytesTensor
    # if isinstance(target, QBytesTensor):
    #     # Get the target dtype from the existing quantized tensor
    #     target_dtype = target._data.dtype

    #     # Handle device placement
    #     device = target._data.device
    #     result_float = result_float.to(device)

    #     # Compute new quantized values and scale
    #     quantized_data, new_scale = quantize_tensor(result_float, target_dtype)

    #     # Update the internal tensors with newly computed values
    #     target._data.copy_(quantized_data)
    #     target._scale.copy_(new_scale)
    # else:
    #     # Regular tensor update
    target.copy_(result_float)


def get_format_params(dtype: torch.dtype) -> tuple[int, int]:
    """
    Returns (mantissa_bits, total_bits) for each format.
    mantissa_bits excludes the implicit leading 1.
    """
    if dtype == torch.float32:
        return 23, 32
    elif dtype == torch.bfloat16:
        return 7, 16
    elif dtype == torch.float16:
        return 10, 16
    elif dtype == torch.float8_e4m3fn:
        return 3, 8
    elif dtype == torch.float8_e5m2:
        return 2, 8
    elif dtype == torch.int8:
        return 0, 8  # Int8 doesn't have mantissa bits
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def copy_stochastic_bf16(target: torch.Tensor, source: torch.Tensor):
    # adapted from https://github.com/Nerogar/OneTrainer/blob/411532e85f3cf2b52baa37597f9c145073d54511/modules/util/bf16_stochastic_rounding.py#L5
    # create a random 16 bit integer
    result = torch.randint_like(
        source,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
    )

    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))

    del result


def copy_stochastic(target: torch.Tensor, source: torch.Tensor, eps: Optional[float] = None) -> None:
    with torch.no_grad():
        # assert if target is on cpu, throw error
        assert target.device.type != 'cpu', "Target is on cpu!"
        assert source.device.type != 'cpu', "Source is on cpu!"

        if target.dtype == torch.float32:
            target.copy_(source)
            return
        if target.dtype == torch.bfloat16:
            copy_stochastic_bf16(target, source)
            return

        mantissa_bits, _ = get_format_params(target.dtype)
        round_factor = 2 ** (23 - mantissa_bits)

        # Add uniform noise for stochastic rounding
        noise = torch.rand_like(source, device=source.device) - 0.5
        rounded = torch.round(source * round_factor + noise)
        result_float = rounded / round_factor

        # Clamp for float8
        if target.dtype == torch.float8_e4m3fn:
            result_float.clamp_(-448.0, 448.0)
        elif target.dtype == torch.float8_e5m2:
            result_float.clamp_(-57344.0, 57344.0)

        update_parameter(target, result_float)


class Auto8bitTensor:
    def __init__(self, data: Tensor, *args, **kwargs):
        if isinstance(data, dict):  # Add constructor from state dict
            self._load_from_state_dict(data)
        else:
            abs_max = data.abs().max().item()
            scale = abs_max / 127.0 if abs_max > 0 else 1.0

            self.quantized = (data / scale).round().clamp(-127, 127).to(torch.int8)
            self.scale = scale
            self.orig_dtype = data.dtype

    def dequantize(self) -> Tensor:
        return self.quantized.to(dtype=torch.float32) * self.scale

    def to(self, *args, **kwargs):
        # Handle the dtype argument whether it's positional or keyword
        dtype = None
        if args and isinstance(args[0], torch.dtype):
            dtype = args[0]
            args = args[1:]
        elif 'dtype' in kwargs:
            dtype = kwargs['dtype']
            del kwargs['dtype']

        if dtype is not None:
            # First dequantize then convert to requested dtype
            return self.dequantize().to(dtype=dtype, *args, **kwargs)

        # If no dtype specified, just pass through to parent
        return self.dequantize().to(*args, **kwargs)

    def state_dict(self):
        """Returns a dictionary containing the current state of the tensor."""
        return {
            'quantized': self.quantized,
            'scale': self.scale,
            'orig_dtype': self.orig_dtype
        }

    def _load_from_state_dict(self, state_dict):
        """Loads the tensor state from a state dictionary."""
        self.quantized = state_dict['quantized']
        self.scale = state_dict['scale']
        self.orig_dtype = state_dict['orig_dtype']

    def __str__(self):
        return f"Auto8bitTensor({self.dequantize()})"


def stochastic_grad_accummulation(param):
    if hasattr(param, "_accum_grad"):
        grad_fp32 = param._accum_grad.clone().to(torch.float32)
        grad_fp32.add_(param.grad.to(torch.float32))
        copy_stochastic(param._accum_grad, grad_fp32)
        del grad_fp32
        del param.grad
    else:
        param._accum_grad = param.grad.clone()
        del param.grad


class Prodigy8bit(Optimizer):
    r"""
    Implements Adam with Prodigy step-sizes.
    Handles stochastic rounding for various precisions as well as stochastic gradient accumulation.
    Stores state in 8bit for memory savings.
    Leave LR set to 1 unless you encounter instability.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate adjustment parameter. Increases or decreases the Prodigy learning rate.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        beta3 (float):
            coefficients for computing the Prodidy stepsize using running averages.
            If set to None, uses the value of square root of beta2 (default: None).
        eps (float):
            Term added to the denominator outside of the root operation to improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        decouple (boolean):
            Use AdamW style decoupled weight decay
        use_bias_correction (boolean):
            Turn on Adam's bias correction. Off by default.
        safeguard_warmup (boolean):
            Remove lr from the denominator of D estimate to avoid issues during warm-up stage. Off by default.
        d0 (float):
            Initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
        d_coef (float):
            Coefficient in the expression for the estimate of d (default 1.0).
            Values such as 0.5 and 2.0 typically work as well.
            Changing this parameter is the preferred way to tune the method.
        growth_rate (float):
            prevent the D estimate from growing faster than this multiplicative rate.
            Default is inf, for unrestricted. Values like 1.02 give a kind of learning
            rate warmup effect.
        fsdp_in_use (bool):
            If you're using sharded parameters, this should be set to True. The optimizer
            will attempt to auto-detect this, but if you're using an implementation other
            than PyTorch's builtin version, the auto-detection won't work.
    """

    def __init__(self, params, lr=1.0,
                 betas=(0.9, 0.999), beta3=None,
                 eps=1e-8, weight_decay=0, decouple=True,
                 use_bias_correction=False, safeguard_warmup=False,
                 d0=1e-6, d_coef=1.0, growth_rate=float('inf'),
                 fsdp_in_use=False):
        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))

        if decouple and weight_decay > 0:
            print(f"Using decoupled weight decay")

        defaults = dict(lr=lr, betas=betas, beta3=beta3,
                        eps=eps, weight_decay=weight_decay,
                        d=d0, d0=d0, d_max=d0,
                        d_numerator=0.0, d_coef=d_coef,
                        k=0, growth_rate=growth_rate,
                        use_bias_correction=use_bias_correction,
                        decouple=decouple, safeguard_warmup=safeguard_warmup,
                        fsdp_in_use=fsdp_in_use)
        self.d0 = d0
        super(Prodigy8bit, self).__init__(params, defaults)

        self.is_stochastic_rounding_accumulation = False

        # setup stochastic grad accum hooks
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad and param.dtype != torch.float32:
                    self.is_stochastic_rounding_accumulation = True
                    param.register_post_accumulate_grad_hook(
                        stochastic_grad_accummulation
                    )

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True

    def step_hook(self):
        if not self.is_stochastic_rounding_accumulation:
            return
        # copy over stochastically rounded grads
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad and hasattr(param, "_accum_grad"):
                    param.grad = param._accum_grad
                    del param._accum_grad

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # call pre step
        self.step_hook()
        loss = None
        if closure is not None:
            loss = closure()

        d_denom = 0.0

        group = self.param_groups[0]
        use_bias_correction = group['use_bias_correction']
        beta1, beta2 = group['betas']
        beta3 = group['beta3']
        if beta3 is None:
            beta3 = math.sqrt(beta2)
        k = group['k']

        d = group['d']
        d_max = group['d_max']
        d_coef = group['d_coef']
        lr = max(group['lr'] for group in self.param_groups)

        if use_bias_correction:
            bias_correction = ((1 - beta2**(k+1))**0.5) / (1 - beta1**(k+1))
        else:
            bias_correction = 1

        dlr = d*lr*bias_correction

        growth_rate = group['growth_rate']
        decouple = group['decouple']
        fsdp_in_use = group['fsdp_in_use']

        d_numerator = group['d_numerator']
        d_numerator *= beta3

        for group in self.param_groups:
            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']
            group_lr = group['lr']
            d0 = group['d0']
            safeguard_warmup = group['safeguard_warmup']

            if group_lr not in [lr, 0.0]:
                raise RuntimeError(
                    f"Setting different lr values in different parameter groups is only supported for values of 0")

            for p in group['params']:
                if p.grad is None:
                    continue
                if hasattr(p, "_fsdp_flattened"):
                    fsdp_in_use = True

                grad = p.grad.data.to(torch.float32)
                p_fp32 = p.clone().to(torch.float32)

                # Apply weight decay (coupled variant)
                if decay != 0 and not decouple:
                    grad.add_(p_fp32.data, alpha=decay)

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    state['s'] = Auto8bitTensor(
                        torch.zeros_like(p_fp32.data).detach())
                    state['p0'] = Auto8bitTensor(p_fp32.detach().clone())
                    # Exponential moving average of gradient values
                    state['exp_avg'] = Auto8bitTensor(
                        torch.zeros_like(p_fp32.data).detach())
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = Auto8bitTensor(
                        torch.zeros_like(p_fp32.data).detach())

                exp_avg = state['exp_avg'].to(torch.float32)
                exp_avg_sq = state['exp_avg_sq'].to(torch.float32)

                s = state['s'].to(torch.float32)
                p0 = state['p0'].to(torch.float32)

                if group_lr > 0.0:
                    # we use d / d0 instead of just d to avoid getting values that are too small
                    d_numerator += (d / d0) * dlr * torch.dot(grad.flatten(),
                                                              (p0.data - p_fp32.data).flatten()).item()

                    # Adam EMA updates
                    exp_avg.mul_(beta1).add_(grad, alpha=d * (1-beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(
                        grad, grad, value=d * d * (1-beta2))

                    if safeguard_warmup:
                        s.mul_(beta3).add_(grad, alpha=((d / d0) * d))
                    else:
                        s.mul_(beta3).add_(grad, alpha=((d / d0) * dlr))
                    d_denom += s.abs().sum().item()

                # update state with stochastic rounding
                state['exp_avg'] = Auto8bitTensor(exp_avg)
                state['exp_avg_sq'] = Auto8bitTensor(exp_avg_sq)
                state['s'] = Auto8bitTensor(s)
                state['p0'] = Auto8bitTensor(p0)

        d_hat = d

        # if we have not done any progres, return
        # if we have any gradients available, will have d_denom > 0 (unless \|g\|=0)
        if d_denom == 0:
            return loss

        if lr > 0.0:
            if fsdp_in_use:
                dist_tensor = torch.zeros(2).cuda()
                dist_tensor[0] = d_numerator
                dist_tensor[1] = d_denom
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_d_numerator = dist_tensor[0]
                global_d_denom = dist_tensor[1]
            else:
                global_d_numerator = d_numerator
                global_d_denom = d_denom

            d_hat = d_coef * global_d_numerator / global_d_denom
            if d == group['d0']:
                d = max(d, d_hat)
            d_max = max(d_max, d_hat)
            d = min(d_max, d * growth_rate)

        for group in self.param_groups:
            group['d_numerator'] = global_d_numerator
            group['d_denom'] = global_d_denom
            group['d'] = d
            group['d_max'] = d_max
            group['d_hat'] = d_hat

            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.to(torch.float32)
                p_fp32 = p.clone().to(torch.float32)

                state = self.state[p]

                exp_avg = state['exp_avg'].to(torch.float32)
                exp_avg_sq = state['exp_avg_sq'].to(torch.float32)

                state['step'] += 1

                denom = exp_avg_sq.sqrt().add_(d * eps)

                # Apply weight decay (decoupled variant)
                if decay != 0 and decouple:
                    p_fp32.data.add_(p_fp32.data, alpha=-decay * dlr)

                # Take step
                p_fp32.data.addcdiv_(exp_avg, denom, value=-dlr)
                # apply stochastic rounding
                copy_stochastic(p.data, p_fp32.data)

            group['k'] = k + 1

        return loss
