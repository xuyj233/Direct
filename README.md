<div align="center">

# 🎯 DiReCT
### *Directionally-Restrained Constrained Training*

[![ICML 2026 Spotlight](https://img.shields.io/badge/ICML%202026-Spotlight-blueviolet?style=for-the-badge&logo=data:image/svg+xml;base64,)](https://icml.cc/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

🏆 **Accepted as a Spotlight paper at ICML 2026** 🏆

*Reference implementation of the **sample-selection** module described in*

> **Towards Efficient LLMs Annealing with Principled Sample Selection**
> *ICML 2026 (Spotlight 🏆)*

</div>

---

## ✨ Overview

Given a pre-trained model checkpoint (θ<sub>s</sub>, the state at the onset
of the annealing phase), a validation set and a candidate training pool,
`direct.select_samples` returns the indices of the **K** samples whose
gradients best align with the *flat* eigen-subspace of the validation
Hessian while respecting a budget on the *stiff* eigen-subspace.

### 🌟 Highlights

- ⚡ **Sketch-based** — Gaussian random projection avoids materialising the full Hessian.
- 🎯 **Principled selection** — flat / stiff subspace decomposition via spectral elbow.
- 🧩 **Model-agnostic** — plug in any `nn.Module` and a scalar-loss callable.
- 🔬 **Reproducible** — deterministic sketching from a single seed.

---

## 📦 Installation

```bash
pip install -r requirements.txt
pip install -e .            # editable install of the `direct` package
```

---

## 🚀 Quick Start

### Minimal example

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

The return value is a `List[int]` of length `K` — the 0-based positions
into `train_samples` that DiReCT recommends including in the annealing
schedule.

### Runnable demo on GPT-2

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

<details>
<summary><b>📋 Expected output (abridged)</b></summary>

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

</details>

---

## ⚙️ Key Configuration

| Argument | Default | Description |
| :--- | :---: | :--- |
| `K` | *required* | Number of training samples to select |
| `k_sketch` | `2048` | Dimension of the Gaussian sketch |
| `elbow_energy` | `0.945` | Cumulative-energy threshold for the spectral elbow |
| `tau_perp` | `0.1` | Relative budget on the stiff-subspace energy |
| `sca_max_iter` | `20` | Max outer SCA iterations |
| `inner_iters` | `50` | Inner PGA steps per SCA sub-problem |

See [`direct/config.py`](direct/config.py) for the full `DiReCTConfig` definition.

---

## 🧪 Tests

```bash
pytest -v
```

Covers the Dykstra projection, spectral elbow, the Gaussian sketcher,
plus an end-to-end smoke test on a synthetic MLP.

---

## 📁 Layout

```text
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

---

## 📖 Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{xu2026direct,
  title     = {Towards Efficient LLMs Annealing with Principled Sample Selection},
  author    = {Xu, Yuanjian and Hao, Jianing and Zhang, Wanbo and Li, Zhong and Zhang, Guang},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

---

## 📄 License

Released under the **MIT License** — see [LICENSE](LICENSE) for details.

<div align="center">

*Made with ❤️ for principled data selection.*

</div>
