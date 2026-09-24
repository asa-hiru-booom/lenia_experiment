# 🧬 Lenia Autonomous Discovery Lab: P4 Structural Skeleton

[![Deploy GitHub Pages](https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>/actions/workflows/pages.yml/badge.svg)](https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>/actions/workflows/pages.yml)
[![Live Interactive Report](https://img.shields.io/badge/GitHub%20Pages-Live%20Dossier-38bdf8?style=flat&logo=github)](https://<YOUR_GITHUB_USERNAME>.github.io/<YOUR_REPO_NAME>/)

An open-ended autonomous discovery platform using local Large Language Models (LLMs) via **Ollama** to synthesize mathematical laws for continuous cellular automata (**Lenia**).

The discovery engine formulates interaction kernels $K(r)$ and local reaction growth functions $G(u)$ using the **P4 (Structural Skeleton)** philosophy. It features asymmetric crescent wavefront seeding, dual-memory evolutionary genetics (champion preservation + short-term failure logs), an AST security sandbox, and zero-loss interrupt handling.

---

## 📊 Live Interactive Dossier

Explore real-time simulation playbacks, mathematical function morphing, and interactive Conway sandboxes directly in your browser:

👉 **[View the Live GitHub Pages Report](https://<YOUR_GITHUB_USERNAME>.github.io/<YOUR_REPO_NAME>/)**

---

## ⚡ What is `uv`?

[`uv`](https://github.com/astral-sh/uv) is an extremely fast Python package and project manager written in Rust by Astral. It serves as a seamless drop-in replacement for `pip`, `venv`, and `pip-tools`.

Thanks to [PEP 723 (Inline Script Metadata)](https://peps.python.org/pep-0723/), `run.py` declares its own dependencies at the top of the file. You do **not** need to create or activate a virtual environment manually—`uv` automatically provisions an isolated, cached environment on the fly.

---

## 🚀 Quickstart & Reproduction

### Prerequisites

1. Install `uv`:
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Install and launch [Ollama](https://ollama.ai/):
```bash
ollama pull qwen2.5-coder:14b
ollama serve
```

### Running the Discovery Loop

Clone the repository and run the engine:

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>

# Execute directly with uv (dependencies will be installed automatically)
uv run run.py
```

### Execution Controls & Interruption

- **Select Presets:** Choose `[1]` for a quick smoke test or `[2]` for deep continuous evolution.
- **Graceful Termination:** You can let the experiment run in the background. Press **`Ctrl+C` once at any time** to safely interrupt the execution. The engine will catch the signal, finish the current evaluation, compile all accumulated telemetry, and write `index.html` to disk.

---

## 🛠️ Repository Layout

```text
.
├── .github/workflows/pages.yml  # Automated deployment workflow for GitHub Pages
├── README.md                    # Project documentation and setup guide
├── index.html                   # Interactive benchmark report and customization guide
└── run.py                       # Self-contained discovery engine with inline metadata
```

---

## 📄 License

Distributed under the [MIT License](LICENSE).
