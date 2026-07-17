# DiReCT: Directionally-Restrained Constrained Training

Reference implementation of the **sample-selection** module described in

> *Towards Efficient LLMs Annealing with Principled Sample Selection*
> (ICML 2026)

Given a pre-trained model checkpoint (θ<sub>s</sub>, the state at the onset
of the annealing phase), a validation set and a candidate training pool,
`direct.select_samples` returns the indices of the **K** samples whose
gradients best align with the *flat* eigen-subspace of the validation
Hessian while respecting a budget on the *stiff* eigen-subspace.

## Installation

```bash
pip install -r requirements.txt
pip install -e .            # editable install of the `direct` package
```

## Quick start

```python
import torch
from direct import DiReCTConfig, select_samples

# 1. Bring your own model & data.
model = ...                                      # nn.Module (θ_s)
train_samples = [...]                            # list of arbitrary objects
val_samples   = [...]

# 2. Provide a scalar-loss callable.
def loss_fn(model, sample):
    x, y = sample
    return torch.nn.functional.cross_entropy(model(x), y)

# 3. Run the selector.
cfg = DiReCTConfig(K=1024, k_sketch=2048,
                   elbow_energy=0.945, tau_perp=0.1)
indices = select_samples(model, train_samples, val_samples,
                         loss_fn=loss_fn, cfg=cfg)
```

The return value is a `List[int]` of length `K` – the 0-based positions
into `train_samples` that DiReCT recommends including in the annealing
schedule.

## Pipeline

The implementation follows Algorithm 1 of the paper:

1. **Gradient sketching** – draw a Gaussian projection
   `R ∈ ℝ^{k × d}` with entries `N(0, 1/k)` and compute
   `z_x = R · ∇ℓ(x; θ_s)` for every validation and training sample.
   `R` is *never* materialised – it is regenerated on the fly in blocks
   from a deterministic seed (see [`direct/sketch.py`](direct/sketch.py)).
2. **Sketched Hessian** – form
   `H̃ = (1/M) Σ_x z_x z_xᵀ` and diagonalise
   (`torch.linalg.eigh`).
3. **Spectral elbow** – pick the smallest `k*` capturing
   `elbow_energy` (default `0.945`) of the total energy and split the
   spectrum into a stiff subspace `I_⊥` and a flat subspace `I_∥`
   (see [`direct/spectral.py`](direct/spectral.py)).
4. **Successive Convex Approximation** – solve the relaxed
   `max ‖G_∥ w‖²  s.t.  w ∈ [0,1]^N, 1ᵀw = K, aᵀw ≤ τ_⊥`
   with a projected-gradient-ascent inner loop and **Dykstra**
   projection onto the three constraints
   ([`direct/sca.py`](direct/sca.py),
   [`direct/projection.py`](direct/projection.py)).
5. **Top-K rounding** – return `argsort(-w*)[:K]`.

## GPT-2 debug example

The [`examples/gpt2_debug.py`](examples/gpt2_debug.py) script exercises
the entire pipeline on HuggingFace's `gpt2` model. To keep memory usage
manageable, the default parameter filter restricts autograd to the
`lm_head` and the last two block LayerNorms; use `--full-model` to run
against every trainable weight.

```bash
python examples/gpt2_debug.py \
    --model gpt2 \
    --num-train 64 --num-val 16 --k 8 --k-sketch 256
```

Expected output (abridged):

```
[example] loading model = gpt2
[example] N_train = 64, N_val = 16
[DiReCT] #params considered = 39,313,152, sketch dim k = 256
[DiReCT] val grad 16/16
[DiReCT] elbow k* = 15  |I_perp|=15  |I_parallel|=241  ...
[SCA] iter=00  obj=...   |dw|=...
...
[example] Selected indices: [3, 27, 41, 12, ...]
```

## Tests

```bash
pytest -v
```

Covers the Dykstra projection, spectral elbow, the Gaussian sketcher,
plus an end-to-end smoke test on a synthetic MLP.

## Layout

```
direct/
├── direct/
│   ├── __init__.py         # public API
│   ├── config.py           # DiReCTConfig
│   ├── gradients.py        # per-sample gradient helpers
│   ├── projection.py       # Dykstra projection
│   ├── sca.py              # Alg. 2: SCA outer loop
│   ├── selector.py         # Alg. 1: select_samples()
│   ├── sketch.py           # Gaussian random projection
│   └── spectral.py         # spectral elbow
├── examples/
│   └── gpt2_debug.py       # GPT-2 driver script
├── tests/
│   ├── test_projection.py
│   ├── test_selector.py
│   ├── test_sketch.py
│   └── test_spectral.py
├── requirements.txt
├── pyproject.toml
├── LICENSE                 # MIT
└── README.md               # you are here
```

## Citation

```bibtex
@inproceedings{xu2026direct,
  title     = {Towards Efficient LLMs Annealing with Principled Sample Selection},
  author    = {Xu, Yuanjian and Hao, Jianing and Zhang, Wanbo and Li, Zhong and Zhang, Guang},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
