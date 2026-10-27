"""Encodings for the "light" and "smallest" checkpoint modes.

"light" compresses only the optimizer state (below); the model weights stay at full precision.
"smallest" also stores the weights in fp16.

Model weights ("smallest" only): floating-point tensors are stored in fp16 and cast back to
their original dtype on load (integer buffers, fp16/bf16 tensors and values too large for
fp16 are kept as they are).

Optimizer state ("light" and "smallest"): floating-point tensors with at least
MIN_QUANT_NUMEL elements are stored with 8-bit blockwise quantization (one fp32 scale per
block of BLOCK values, pure PyTorch).
Adam-style state gets two refinements so a resumed run behaves:
- exp_avg_sq (and max_exp_avg_sq) is quantized as its square root, which shrinks its
  dynamic range;
- exp_avg is stored as its ratio to sqrt(exp_avg_sq) + eps, which is Adam's update
  direction and bounded. After a resume the update m / (sqrt(v) + eps) equals the saved
  ratio up to one quantization step, even where the second moment rounds to zero.
Small tensors and scalars (such as Adam's step counter) are stored exactly.
"""

import torch

BLOCK = 128              # values per quantization block
MIN_QUANT_NUMEL = 1024   # smaller optimizer tensors are stored exactly
FP16_MAX = 65504.0
_SQRT_KEYS = ("exp_avg_sq", "max_exp_avg_sq")
_TAG = "__efficio_q8__"


# ---------------------------------------------------------------- 8-bit blockwise
def quantize_blockwise(t, block=BLOCK):
    """8-bit symmetric quantization with one scale per block of `block` values.
    Each value is off by at most half a step: absmax(block) / 254."""
    flat = t.detach().reshape(-1).float()
    n = flat.numel()
    pad = (-n) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block)
    absmax = blocks.abs().amax(dim=1)
    scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
    q = torch.round(blocks / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return {"q": q.reshape(-1)[:n].clone(), "absmax": absmax.clone(), "block": block,
            "shape": tuple(t.shape), "dtype": _name(t.dtype)}


def dequantize_blockwise(packed):
    """Inverse of quantize_blockwise (float32 result in the original shape)."""
    q, absmax, block = packed["q"].float(), packed["absmax"].float(), packed["block"]
    n = q.numel()
    pad = (-n) % block
    if pad:
        q = torch.cat([q, q.new_zeros(pad)])
    scale = torch.where(absmax > 0, absmax / 127.0, torch.zeros_like(absmax))
    return (q.view(-1, block) * scale[:, None]).reshape(-1)[:n].reshape(packed["shape"])


def _name(dtype):
    return str(dtype).replace("torch.", "")


def _dtype(name):
    return getattr(torch, name)


# ---------------------------------------------------------------- model weights
def encode_model(state_dict):
    """fp16 copy of floating-point weights; returns (tensors, original dtypes)."""
    out, dtypes = {}, {}
    for name, t in state_dict.items():
        if (torch.is_tensor(t) and t.dtype in (torch.float32, torch.float64)
                and (t.numel() == 0 or t.abs().max().item() <= FP16_MAX)):
            out[name] = t.to(torch.float16)
            dtypes[name] = _name(t.dtype)
        else:
            out[name] = t
    return out, dtypes


def decode_model(tensors, dtypes):
    """Cast fp16 weights back to the dtypes they had when saved."""
    return {name: t.to(_dtype(dtypes[name])) if name in dtypes else t
            for name, t in tensors.items()}


# ---------------------------------------------------------------- optimizer state
def _quantizable(t):
    return torch.is_tensor(t) and t.is_floating_point() and t.numel() >= MIN_QUANT_NUMEL


def encode_optimizer(opt_state_dict):
    """Optimizer state_dict with large float state tensors quantized to 8 bits."""
    eps_of = {}
    for group in opt_state_dict.get("param_groups", []):
        for pid in group.get("params", []):
            eps_of[pid] = float(group.get("eps", 1e-8) or 1e-8)
    state = {}
    for pid, entries in opt_state_dict.get("state", {}).items():
        enc = {}
        for key, t in entries.items():
            if key in _SQRT_KEYS and _quantizable(t):
                enc[key] = {**quantize_blockwise(t.float().clamp_min(0).sqrt()), _TAG: "sqrt",
                            "dtype": _name(t.dtype)}
            elif _quantizable(t):
                enc[key] = {_TAG: "linear", **quantize_blockwise(t)}
            else:
                enc[key] = t
        m, v = entries.get("exp_avg"), entries.get("exp_avg_sq")
        if (_quantizable(m) and _quantizable(v) and m.shape == v.shape
                and enc["exp_avg_sq"].get(_TAG) == "sqrt"):
            # The ratio to the true root is Adam's own (bounded) update direction. Decoding
            # multiplies it by the decoded root, so after a resume m / (sqrt(v) + eps) is
            # the saved ratio, off by at most one quantization step.
            eps = eps_of.get(pid, 1e-8)
            ratio = m.float() / (v.float().clamp_min(0).sqrt() + eps)
            enc["exp_avg"] = {**quantize_blockwise(ratio), _TAG: "ratio", "eps": eps,
                              "dtype": _name(m.dtype)}
        state[pid] = enc
    return {**opt_state_dict, "state": state}


def decode_optimizer(opt_state_dict):
    """Inverse of encode_optimizer; tensors come back as float32 in their old dtype."""
    state = {}
    for pid, entries in opt_state_dict.get("state", {}).items():
        dec = {}
        for key, val in entries.items():                    # sqrt and linear first
            if isinstance(val, dict) and val.get(_TAG) == "sqrt":
                dec[key] = dequantize_blockwise(val).pow(2).to(_dtype(val["dtype"]))
            elif isinstance(val, dict) and val.get(_TAG) == "linear":
                dec[key] = dequantize_blockwise(val).to(_dtype(val["dtype"]))
            elif not (isinstance(val, dict) and val.get(_TAG) == "ratio"):
                dec[key] = val
        for key, val in entries.items():                    # then ratios, which need them
            if isinstance(val, dict) and val.get(_TAG) == "ratio":
                root = dec["exp_avg_sq"].float().sqrt()
                dec[key] = (dequantize_blockwise(val) * (root + val["eps"])).to(
                    _dtype(val["dtype"]))
        state[pid] = dec
    return {**opt_state_dict, "state": state}
