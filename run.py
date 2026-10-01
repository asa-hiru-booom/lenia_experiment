# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "ollama",
#     "pillow",
#     "scipy",
# ]
# ///

import ast
import base64
import datetime
import gc
import html
import io
import json
import math
import os
import signal
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ollama
from PIL import Image
from scipy.signal import fftconvolve

matplotlib.use("Agg")

# =====================================================================
# Global Interrupt Trap (Ctrl+C Graceful Shutdown)
# =====================================================================
INTERRUPT_REQUESTED = False


def sigint_handler(signum, frame):
    global INTERRUPT_REQUESTED
    if not INTERRUPT_REQUESTED:
        INTERRUPT_REQUESTED = True
        print("\n\n" + "!" * 78)
        print("⚠️  USER INTERRUPT DETECTED (Ctrl+C).")
        print("   Safely finishing current step and compiling report to 'index.html'...")
        print("!" * 78 + "\n")
    else:
        print("\nForce killing process immediately...")
        sys.exit(1)


signal.signal(signal.SIGINT, sigint_handler)


def format_elapsed(seconds: float) -> str:
    secs = int(seconds)
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}h {m:02d}m {s:02d}s"


# =====================================================================
# 1. P4 (Structural Skeleton) Philosophy & System Prompt
# =====================================================================
P4_PHILOSOPHY = {
    "title": "Structural Skeleton (Functional Composition)",
    "philosophy": (
        "Focus: Mathematical functional composition. Explore composite functions such as shifted Gaussians, "
        "rational algebraic forms, trigonometric modulations, or smooth polynomial cut-offs that exhibit sharp "
        "transitions between growth and decay."
    ),
}

SYSTEM_PROMPT = r"""You are a senior computational physicist discovering universal continuous cellular automata rules (Lenia).
Your explicit objective is to maximize Total Score S_total towards 100.0 by formulating `kernel_func(r)` and `growth_func(u)` that produce a coherent, self-propelling soliton.

【Strict Signatures & Grammar】
1. def kernel_func(r):
   - Takes EXACTLY ONE parameter: r (1D NumPy array in [0.0, 1.0]).
   - Returns a 1D NumPy array of interaction weights (MUST be >= 0.0 everywhere).
   - Signature MUST be `def kernel_func(r):`.
2. def growth_func(u):
   - Takes EXACTLY ONE parameter: u (1D NumPy array of local densities >= 0.0).
   - Returns a 1D NumPy array of state velocities in [-1.0, 1.0].
   - Signature MUST be `def growth_func(u):`.
   - Never reference or introduce external variables like 'displacement' in the code.

【Physical Axioms】
1. Rule of the Void: G(0) MUST be negative (G(0) <= -0.5, ideally ~ -1.0) to prevent spontaneous creation out of empty space.
2. Window of Life: G(u) should be positive only within an intermediate bounded density interval [u_min, u_max] and negative elsewhere.
3. The grid field has been initialized with an asymmetric crescent wavefront with a steep front gradient to initiate directional propulsion.
4. Mutation & Anti-Inflation Directive: Refer to the [CHAMPION BENCHMARK SEED] and [RECENT GENERATION TRAJECTORY]. Maintain the core propulsion mechanism, but tune parameters to prevent the trailing mass from expanding beyond initial levels.

【Evaluation Metric】
1. Viability Score S_viability (0.0 to 100.0): Evaluates survival duration and mass balance.
2. Mobility Score S_mobility (0.0 to 100.0): S_mobility = 5.0 + 95.0 * tanh(sqrt(displacement) / 1.5).
3. Total Score S_total = (S_viability * S_mobility) / 100.0.

【Allowed NumPy Functions】
np.exp, np.sqrt, np.square, np.clip, np.where, np.tanh, np.sin, np.cos, np.maximum, np.minimum, np.abs, np.sum, np.mean, np.log, np.sign, np.power, np.pi, np.zeros_like, np.ones_like.
CRITICAL: Do NOT use Python if/else on arrays. Never use slice assignments like `u[mask] = ...`.

【Output Format】
Return ONLY a valid JSON object:
{
  "thought": "- Bullet point 1: Analysis of the Champion Seed vs Recent Trajectory\n- Bullet point 2: Targeted functional mutation to suppress trailing mass and enhance translation",
  "kernel_code": "def kernel_func(r):\n    return ...",
  "growth_code": "def growth_func(u):\n    return ...",
  "latex_kernel": "K(r) = ...",
  "latex_growth": "G(u) = ..."
}
"""


# =====================================================================
# 2. AST Security Firewall
# =====================================================================
class SecurityError(Exception):
    pass


class ControlledSyntaxChecker(ast.NodeVisitor):
    ALLOWED_NODES = {
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.Assign,
        ast.Name,
        ast.Store,
        ast.Load,
        ast.Constant,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Attribute,
        ast.Subscript,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
        ast.Invert,
        ast.IfExp,
        ast.If,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
        ast.And,
        ast.Or,
        ast.Not,
        ast.BoolOp,
        ast.Pass,
        ast.List,
        ast.Tuple,
        ast.Slice,
    }

    ALLOWED_NP = {
        "exp",
        "sqrt",
        "square",
        "clip",
        "maximum",
        "minimum",
        "abs",
        "sin",
        "cos",
        "tanh",
        "power",
        "pi",
        "zeros_like",
        "ones_like",
        "where",
        "zeros",
        "ones",
        "sum",
        "mean",
        "log",
        "sign",
        "arctan",
        "arctan2",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "invert",
        "logical_and",
        "logical_or",
        "logical_not",
    }

    ALLOWED_BUILTINS = {"float", "int", "len", "range", "min", "max", "abs", "bool"}

    def generic_visit(self, node):
        if type(node) not in self.ALLOWED_NODES:
            raise SecurityError(f"Prohibited AST syntax: {type(node).__name__}.")
        super().generic_visit(node)

    def visit_Import(self, node):
        raise SecurityError("Import statement ('import') is prohibited.")

    def visit_ImportFrom(self, node):
        raise SecurityError("Import statement ('from ... import') is prohibited.")

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "np":
                func_name = node.func.attr
                if func_name not in self.ALLOWED_NP:
                    raise SecurityError(f"Unauthorized NumPy function: 'np.{func_name}'.")
                self.generic_visit(node)
                return
            else:
                raise SecurityError("Method calls on arbitrary objects are prohibited.")
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in self.ALLOWED_BUILTINS:
                raise SecurityError(f"Unauthorized builtin function call: '{func_name}()'.")
            self.generic_visit(node)
            return
        raise SecurityError(f"Complex or dynamic function call detected: {ast.dump(node.func)}")


def verify_code_safety(code_str: str):
    tree = ast.parse(code_str)
    ControlledSyntaxChecker().visit(tree)


# =====================================================================
# 3. Lenia Simulator (Crescent Wavefront Field)
# =====================================================================
class LeniaSimulator:

    def __init__(self, size=100, dt=0.1, max_steps=80, radius=13):
        self.size = size
        self.dt = dt
        self.max_steps = max_steps
        self.radius = radius
        self.mid = size // 2

    def build_kernel(self, kernel_func):
        y, x = np.ogrid[-self.radius : self.radius + 1, -self.radius : self.radius + 1]
        r = np.sqrt(x**2 + y**2) / self.radius
        mask = r <= 1.0

        K = np.zeros_like(r, dtype=np.float64)
        K[mask] = kernel_func(r[mask])
        K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
        K = np.maximum(K, 0.0)
        sum_k = np.sum(K)
        return K / sum_k if sum_k > 0 else K

    def init_state(self):
        y, x = np.ogrid[: self.size, : self.size]
        dx = x - self.mid
        dy = y - self.mid
        r = np.sqrt(dx**2 + dy**2) / 8.0

        base = np.exp(-0.5 * (r - 0.7) ** 2 / (0.35**2))
        theta = np.arctan2(dy, dx)
        polarity = 0.5 + 0.5 * np.cos(theta - np.pi / 4.0)

        A = base * (polarity**1.5)
        A = np.where(r > 1.8, 0.0, A)
        return np.clip(A, 0.0, 1.0)

    def run(self, kernel_func, growth_func):
        K = self.build_kernel(kernel_func)
        if np.all(K == 0):
            return {
                "status": "INVALID",
                "step": 0,
                "displacement": 0.0,
                "init_mass": 0.0,
                "final_mass": 0.0,
                "mass_trend": "Invalid Kernel (All zeros)",
                "frames": [],
                "reason": "Kernel evaluation resulted in all zeros or NaNs.", "telemetry": None,
            }

        A = self.init_state()
        init_mass = float(np.sum(A))

        y_idx, x_idx = np.indices(A.shape)
        init_com = np.array([np.sum(x_idx * A) / init_mass, np.sum(y_idx * A) / init_mass])

        frames = []
        for step in range(self.max_steps):
            if step % 2 == 0:
                frames.append(Image.fromarray((np.clip(A, 0.0, 1.0) * 255).astype(np.uint8)))

            U = fftconvolve(A, K, mode="same")
            G = growth_func(U)
            G = np.nan_to_num(G, nan=-1.0, posinf=1.0, neginf=-1.0)
            if step == 0:
                wave_mask = A > 0.02
                if np.any(wave_mask):
                    u_active, g_active = U[wave_mask], G[wave_mask]
                    telemetry = {
                        "u_min": float(np.min(u_active)), "u_mean": float(np.mean(u_active)), "u_max": float(np.max(u_active)),
                        "g_min": float(np.min(g_active)), "g_mean": float(np.mean(g_active)), "g_max": float(np.max(g_active)),
                        "pos_growth_ratio": float(np.mean(g_active > 0.0) * 100.0)
                    }
                else:
                    telemetry = {"u_min": 0.0, "u_mean": 0.0, "u_max": 0.0, "g_min": -1.0, "g_mean": -1.0, "g_max": -1.0, "pos_growth_ratio": 0.0}
            A = np.clip(A + self.dt * G, 0.0, 1.0)

            cur_m = float(np.sum(A))
            if cur_m < init_mass * 0.03:
                return {
                    "status": "DEAD",
                    "step": step,
                    "displacement": 0.0,
                    "init_mass": init_mass,
                    "final_mass": cur_m,
                    "mass_trend": f"Starvation: Mass collapsed from {init_mass:.1f} to {cur_m:.1f} at Step {step}.",
                    "frames": frames,
                    "reason": f"Extinction occurred at Step {step}.", "telemetry": telemetry,
                }

            if cur_m > (self.size**2) * 0.45:
                return {
                    "status": "EXPLODED",
                    "step": step,
                    "displacement": 0.0,
                    "init_mass": init_mass,
                    "final_mass": cur_m,
                    "mass_trend": f"Overcrowding Explosion: Mass inflated from {init_mass:.1f} to {cur_m:.1f} at Step {step}.",
                    "frames": frames,
                    "reason": f"System saturated grid at Step {step}.", "telemetry": telemetry,
                }

        tot_m = max(float(np.sum(A)), 1e-6)
        final_com = np.array([np.sum(x_idx * A) / tot_m, np.sum(y_idx * A) / tot_m])
        displacement = float(np.linalg.norm(final_com - init_com))

        if displacement < 1.0:
            status = "STAGNANT"
            reason = f"Survived all {self.max_steps} steps, but remained stationary (Displacement: {displacement:.2f}px < 1px threshold)."
        elif displacement < 3.0:
            status = "DRIFTING"
            reason = f"Survived all {self.max_steps} steps with weak drift (Displacement: {displacement:.2f}px)."
        else:
            status = "ALIVE"
            reason = f"Survived all {self.max_steps} steps with coherent translation (Displacement: {displacement:.2f}px)."

        return {
            "status": status,
            "step": self.max_steps,
            "displacement": displacement,
            "init_mass": init_mass,
            "final_mass": tot_m,
            "mass_trend": f"Sustained: Mass shifted from initial {init_mass:.1f} to {tot_m:.1f}.",
            "frames": frames,
            "reason": reason, "telemetry": telemetry,
        }


# =====================================================================
# 4. Host Scoring & Feedback Logic
# =====================================================================
def compute_meta_score_detailed(sim_result: dict, max_steps: int) -> dict:
    step_ratio = sim_result["step"] / max(max_steps, 1)

    init_m = sim_result["init_mass"]
    final_m = sim_result["final_mass"]
    mass_ratio = (final_m / init_m) if init_m > 0 else 0.0
    mass_balance = max(0.0, 1.0 - abs(mass_ratio - 1.0))

    s_viability = 100.0 * (step_ratio * (0.6 + 0.4 * mass_balance))

    disp = max(float(sim_result["displacement"]), 0.0)
    s_mobility = 5.0 + 95.0 * math.tanh(math.sqrt(disp) / 1.5)

    s_total = (s_viability * s_mobility) / 100.0

    return {
        "s_viability": s_viability,
        "s_mobility": s_mobility,
        "displacement": disp,
        "s_total": s_total,
        "step_ratio": step_ratio,
        "mass_balance": mass_balance,
    }


def format_generation_feedback(
    sim_result: dict,
    score_dict: dict,
    gen: int,
    max_steps: int,
    champion_record: dict | None,
    recent_history: list[str],
) -> str:
    v_score = score_dict["s_viability"]
    m_score = score_dict["s_mobility"]
    disp = score_dict["displacement"]
    total = score_dict["s_total"]
    status = sim_result["status"]

    if champion_record is None:
        champ_str = "None yet (Initial generation)."
    else:
        champ_str = (
            f"- Discovered at Gen {champion_record['gen']}: S_total = {champion_record['score']:.1f} / 100.0\n"
            f"- Metrics: Displacement = {champion_record['displacement']:.2f}px | S_viability = {champion_record['v_score']:.1f} | S_mobility = {champion_record['m_score']:.1f}\n"
            "- Code Architecture:\n"
            "--- START CHAMPION CODE ---\n"
            f"{champion_record['code']}\n"
            "--- END CHAMPION CODE ---\n"
            "* Directive: Retain the stable wavefront structure of this code while making targeted adjustments to constrain mass expansion."
        )

    recent_table = "\n".join(recent_history[-5:]) if recent_history else "No prior attempts."

    feedback = f"""[CHAMPION BENCHMARK SEED (All-Time Best in this Run)]
{champ_str}

[RECENT GENERATION TRAJECTORY (Last attempts)]
{recent_table}

[LATEST GENERATION OUTCOME (Gen {gen})]
- S_viability (Survival & Mass Balance): {v_score:.1f} / 100.0
  (Completed {sim_result['step']}/{max_steps} steps | Status: {status} | {sim_result['mass_trend']})
- S_mobility (Kinetic Translation): {m_score:.1f} / 100.0 (Displacement: {disp:.2f} px)
- Total Score: S_total = ({v_score:.1f} * {m_score:.1f}) / 100.0 = {total:.1f} / 100.0
Goal: Beat the Champion Seed's S_total by finding the ideal non-linear balance!"""
    return feedback


def sample_curves(kernel_func, growth_func, n_samples=100):
    r_pts = np.linspace(0, 1, n_samples)
    u_pts = np.linspace(0, 1, n_samples)
    try:
        raw_k = np.array(kernel_func(r_pts), dtype=float)
        k_vals = np.nan_to_num(raw_k, nan=0.0, posinf=0.0, neginf=0.0).tolist()
    except Exception:
        k_vals = [0.0] * n_samples
    try:
        raw_g = np.array(growth_func(u_pts), dtype=float)
        g_vals = np.nan_to_num(raw_g, nan=-1.0, posinf=1.0, neginf=-1.0).tolist()
    except Exception:
        g_vals = [-1.0] * n_samples
    return k_vals, g_vals


def plot_to_base64(kernel_func, growth_func, telemetry=None):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.4), dpi=90)
    r = np.linspace(0, 1, 150)
    try:
        axes[0].plot(r, kernel_func(r), color="#38bdf8", lw=2)
    except Exception:
        pass
    axes[0].set_title("Kernel K(r)", fontsize=10, color="#e2e8f0")
    axes[0].set_facecolor("#020617")
    axes[0].tick_params(colors="#94a3b8")
    axes[0].grid(True, alpha=0.15)

    u = np.linspace(0, 1, 150)
    try:
        if telemetry and telemetry["u_max"] > telemetry["u_min"]:
            axes[1].axvspan(telemetry["u_min"], telemetry["u_max"], color="#38bdf8", alpha=0.20, label=f"Observed U [{telemetry['u_min']:.2f}, {telemetry['u_max']:.2f}]")
            axes[1].axvline(telemetry["u_mean"], color="#38bdf8", ls=":", alpha=0.7, lw=1.5)
        axes[1].plot(u, growth_func(u), color="#34d399", lw=2)
        if telemetry:
            axes[1].legend(loc="upper right", fontsize=7, facecolor="#090d16", edgecolor="#334155", labelcolor="#cbd5e1")
        axes[1].axhline(0, color="#64748b", ls="--", alpha=0.5)
    except Exception:
        pass
    axes[1].set_title("Growth G(u)", fontsize=10, color="#e2e8f0")
    axes[1].set_facecolor("#020617")
    axes[1].tick_params(colors="#94a3b8")
    axes[1].grid(True, alpha=0.15)

    fig.patch.set_facecolor("#090d16")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def frames_to_gif_base64(frames):
    if not frames:
        return ""
    buf = io.BytesIO()
    colored = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
    colored[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=colored[1:],
        duration=50,
        loop=0,
    )
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_executive_review(client, review_model, matrix_results):
    print("🧠 Generating Executive Review via Ollama...")
    summary_lines = []
    for run_id, res in matrix_results.items():
        summary_lines.append(
            f"- [{run_id}]: BestScore={res['best_score']:.1f}, MaxMove={res['max_move']:.2f}px, FinalStatus={res['final_status']}"
        )
    prompt = f"""You are a computational physicist evaluating this Lenia P4 Structural Skeleton discovery run on {review_model}:
{chr(10).join(summary_lines)}

Write a concise 3-4 sentence evaluation in English covering:
1. Analysis of self-propulsion and trajectory stability under P4 Structural Skeleton.
2. The role of the champion seed and genetic memory in driving the highest score.
"""
    try:
        res = client.chat(
            model=review_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3},
        )
        return res["message"]["content"]
    except Exception:
        return f"Explored P4 Structural Skeleton discovery across {len(matrix_results)} regimes."


# =====================================================================
# 5. Execution Presets (Focused on 14B)
# =====================================================================
def select_execution_mode(client):
    target_model = "qwen2.5-coder:14b"
    try:
        models_data = client.list()
        local_models = [m.model for m in models_data.models]
        if not any("14b" in m for m in local_models):
            target_model = local_models[0] if local_models else "qwen2.5-coder:14b"
    except Exception:
        target_model = "qwen2.5-coder:14b"

    print("\n" + "=" * 78)
    print("🧬 LENIA P4 14B ENGINE LAB (DUAL-MEMORY + INTERRUPTIBLE)")
    print("=" * 78)
    print(f"📦 Dedicated Model: {target_model}")
    print("-" * 78)
    print("Select Execution Preset:")
    print(f"  [1] 🧪 Quick Test Run (2 Generations / ~1 min)")
    print(f"  [2] 🚀 Deep Continuous Evolution (Up to 100 Gens / 14B Dedicated)")
    print("      - Let it run as long as you like. Hit Ctrl+C anytime to write 'index.html'.")
    print("  [3] ⚙️ Custom Generation Count")
    print("-" * 78)

    choice = input("Enter choice [1/2/3] (Press Enter for Default: 2): ").strip()
    if not choice:
        choice = "2"

    if choice == "1":
        return {
            "mode_name": "P4 Test Run (Smoke Verification)",
            "models": [target_model],
            "gens_per_run": 2,
            "max_sim_steps": 40,
            "timeout_sec": 300,
        }
    elif choice == "2":
        return {
            "mode_name": f"P4 Continuous Evolution ({target_model})",
            "models": [target_model],
            "gens_per_run": 100,
            "max_sim_steps": 80,
            "timeout_sec": 86400,
        }
    else:
        gens_input = input("Generations to run (default: 50): ").strip()
        gens = int(gens_input) if gens_input.isdigit() else 50
        return {
            "mode_name": f"P4 Custom Evolution ({target_model})",
            "models": [target_model],
            "gens_per_run": gens,
            "max_sim_steps": 80,
            "timeout_sec": 86400,
        }


# =====================================================================
# 6. HTML Template (Includes Hacking & Customization Guide Tab)
# =====================================================================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Lenia Autonomous Discovery - P4 Structural Skeleton</title>

<!-- Chart.js -->
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<!-- KaTeX Styles & Scripts -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"/>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>

<style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #070a12; color: #e2e8f0; line-height: 1.6; padding: 2rem 1rem; margin: 0; }
    .container { max-width: 1160px; margin: 0 auto; }
    h1 { color: #38bdf8; text-align: center; margin-bottom: 0.2rem; font-size: 2.1rem; }
    
    .meta-panel {
        background: #111726;
        border-radius: 8px;
        padding: 1.2rem 1.6rem;
        margin: 1.2rem 0 1.8rem 0;
        border: 1px solid #1f2a40;
        font-size: 0.88rem;
    }
    .meta-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 0.6rem 1.4rem;
        margin-top: 0.5rem;
    }
    .meta-item strong { color: #38bdf8; }

    .main-tab-bar {
        display: flex;
        gap: 0.8rem;
        background: #090d16;
        padding: 0.4rem;
        border-radius: 10px;
        border: 1px solid #1f2a40;
        margin-bottom: 2rem;
        justify-content: center;
    }
    .main-tab-btn {
        background: #111726;
        border: 1px solid #1f2a40;
        color: #94a3b8;
        font-size: 1rem;
        padding: 0.65rem 1.8rem;
        border-radius: 8px;
        cursor: pointer;
        font-weight: bold;
        transition: all 0.2s ease;
    }
    .main-tab-btn:hover { color: #fff; background: #1a233a; }
    .main-tab-btn.active {
        background: #0284c7;
        color: #ffffff;
        border-color: #38bdf8;
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.35);
    }
    .main-tab-content { display: none; }
    .main-tab-content.active { display: block; }

    .synthesis-card {
        background: #111726;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 2rem;
        font-size: 0.95rem;
        color: #f1f5f9;
    }

    .matrix-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0 2rem 0;
        background: #111726;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #1f2a40;
    }
    .matrix-table th, .matrix-table td {
        padding: 0.9rem 0.6rem;
        text-align: center;
        border: 1px solid #1a233a;
    }
    .matrix-table th { background: #090d16; color: #38bdf8; font-size: 0.88rem; }
    .matrix-cell {
        cursor: pointer;
        transition: all 0.2s;
        border-radius: 6px;
        padding: 0.6rem 0.4rem;
    }
    .matrix-cell:hover { background: #1a233a; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.5); }

    .compare-container {
        background: #111726;
        border: 1px solid #1f2a40;
        border-radius: 12px;
        padding: 1.6rem;
        margin-bottom: 2rem;
    }
    .compare-selectors {
        display: flex;
        gap: 1.5rem;
        justify-content: center;
        margin-bottom: 1.4rem;
        flex-wrap: wrap;
    }
    .compare-select-group {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    select {
        background: #090d16;
        color: #e2e8f0;
        border: 1px solid #334155;
        padding: 0.45rem 0.8rem;
        border-radius: 6px;
        font-size: 0.88rem;
    }
    .compare-view {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
    }
    @media (max-width: 820px) {
        .compare-view { grid-template-columns: 1fr; }
    }
    .compare-panel {
        background: #090d16;
        border: 1px solid #1a233a;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
    }
    .compare-panel img { width: 100%; border-radius: 6px; background: #000; min-height: 180px; object-fit: contain; }

    .morph-wrapper { display: flex; flex-direction: column; align-items: center; background: #090d16; border-radius: 8px; padding: 1.2rem; margin: 1rem 0; border: 1px solid #1a233a; }
    .morph-canvas { background: #020617; border-radius: 6px; border: 1px solid #1a233a; }
    .morph-controls { display: flex; align-items: center; gap: 1rem; margin-top: 0.8rem; width: 100%; max-width: 600px; justify-content: center; }

    .gen-card {
        background: #090d16;
        border: 1px solid #1f2a40;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    .gen-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 0.8rem 0;
    }
    @media (max-width: 700px) {
        .gen-grid { grid-template-columns: 1fr; }
    }
    .gen-grid img { width: 100%; border-radius: 6px; background: #000; }

    .doc-section { background: #111726; border-radius: 10px; padding: 1.8rem; margin-bottom: 2rem; border: 1px solid #1f2a40; }
    .doc-section h2 { color: #38bdf8; font-size: 1.3rem; margin-top: 0; border-bottom: 1px solid #1f2a40; padding-bottom: 0.5rem; }
    .doc-section h3 { color: #f8fafc; font-size: 1.1rem; margin-top: 1.4rem; margin-bottom: 0.4rem; }
    .doc-section p, .doc-section li { font-size: 0.95rem; color: #cbd5e1; line-height: 1.7; }
    .formula-box { background: #090d16; padding: 0.8rem 1.2rem; border-radius: 6px; font-size: 1.05rem; margin: 0.8rem 0; text-align: center; border: 1px solid #1f2a40; }
    .analogy-box { background: rgba(56, 189, 248, 0.08); border-left: 4px solid #38bdf8; padding: 0.8rem 1.2rem; border-radius: 0 6px 6px 0; margin: 1rem 0; font-size: 0.92rem; }

    .code-box {
        background: #090d16;
        border: 1px solid #1f2a40;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.85rem;
        color: #38bdf8;
        overflow-x: auto;
        margin: 0.8rem 0;
    }

    details {
        background: #111726;
        border: 1px solid #1f2a40;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.2rem;
    }
    summary { font-weight: bold; color: #38bdf8; cursor: pointer; font-size: 1.05rem; }
</style>
</head>
<body>
<div class="container">
    <h1>🧬 Lenia Autonomous Discovery: P4 Structural Skeleton</h1>
    
    <div class="meta-panel">
        <div style="font-weight:bold; color:#f8fafc; margin-bottom:0.3rem;">Run Environment &amp; Hyperparameter Manifest</div>
        <div class="meta-grid">
            <div class="meta-item"><strong>Benchmark Mode:</strong> __MODE_NAME__</div>
            <div class="meta-item"><strong>Execution Period:</strong> __START_TIME__ → __END_TIME__</div>
            <div class="meta-item"><strong>Total Elapsed:</strong> __ELAPSED_SEC__ / __TOTAL_TIMEOUT__</div>
            <div class="meta-item"><strong>Generations per Run:</strong> Up to __GENS_PER_RUN__ (Interruptible via Ctrl+C)</div>
            <div class="meta-item"><strong>Arena:</strong> __GRID_SIZE__×__GRID_SIZE__ Torus | dt=__TIME_STEP_DT__ | Steps=__MAX_SIM_STEPS__</div>
            <div class="meta-item"><strong>Strategy Focus:</strong> P4 Structural Skeleton (Dual-Memory Active)</div>
        </div>
    </div>

    <div class="main-tab-bar">
        <button class="main-tab-btn active" id="btnMainReport" onclick="switchMainTab('report')">📊 P4 Discovery Report</button>
        <button class="main-tab-btn" id="btnMainCustom" onclick="switchMainTab('custom')">🛠️ Hacking &amp; Customization Guide</button>
        <button class="main-tab-btn" id="btnMainGuide" onclick="switchMainTab('guide')">📘 Mathematical Architecture</button>
    </div>

    <!-- TAB 1: P4 EXPERIMENT REPORT -->
    <div id="main-tab-report" class="main-tab-content active">

        <div class="synthesis-card">
            <strong style="color:#10b981; font-size:1.05rem;">🤖 Executive Comparative Synthesis:</strong>
            <p style="margin:0.5rem 0 0 0; line-height:1.7;">__EXECUTIVE_REVIEW__</p>
        </div>

        <div class="doc-section">
            <h2>1.0 Model Matrix Performance</h2>
            <table class="matrix-table">
                <thead>
                    <tr>
                        <th>Model Engine</th>
                        <th>Structural Skeleton (Functional Composition)</th>
                    </tr>
                </thead>
                <tbody>
                    __MATRIX_TABLE_ROWS__
                </tbody>
            </table>
        </div>

        <div class="compare-container">
            <h2 style="color:#38bdf8; font-size:1.3rem; margin-top:0;">2.0 Comparative Playback Inspector</h2>
            <div class="compare-selectors">
                <div class="compare-select-group">
                    <label style="color:#38bdf8; font-weight:bold;">Slot A:</label>
                    <select id="selectRunA"></select>
                    <select id="selectGenA"></select>
                </div>
                <div class="compare-select-group">
                    <label style="color:#34d399; font-weight:bold;">Slot B:</label>
                    <select id="selectRunB"></select>
                    <select id="selectGenB"></select>
                </div>
            </div>

            <div class="compare-view">
                <div class="compare-panel" id="panelA">
                    <h3 style="color:#38bdf8; margin:0 0 0.5rem 0;" id="titleA">Run A</h3>
                    <div style="font-size:0.85rem; color:#94a3b8; margin-bottom:0.5rem;" id="metaA">-</div>
                    <img id="gifA" src="" alt="Simulation A"/>
                    <img id="plotA" src="" alt="Functions A" style="margin-top:0.6rem;"/>
                    <div style="text-align:left; background:#111726; padding:0.8rem; border-radius:6px; margin-top:0.6rem; font-size:0.85rem;" id="thoughtA"></div>
                    <div style="text-align:left; background:#111726; padding:0.6rem; border-radius:6px; margin-top:0.4rem; font-size:0.82rem;" id="mathA"></div>
                </div>
                <div class="compare-panel" id="panelB">
                    <h3 style="color:#34d399; margin:0 0 0.5rem 0;" id="titleB">Run B</h3>
                    <div style="font-size:0.85rem; color:#94a3b8; margin-bottom:0.5rem;" id="metaB">-</div>
                    <img id="gifB" src="" alt="Simulation B"/>
                    <img id="plotB" src="" alt="Functions B" style="margin-top:0.6rem;"/>
                    <div style="text-align:left; background:#111726; padding:0.8rem; border-radius:6px; margin-top:0.6rem; font-size:0.85rem;" id="thoughtB"></div>
                    <div style="text-align:left; background:#111726; padding:0.6rem; border-radius:6px; margin-top:0.4rem; font-size:0.82rem;" id="mathB"></div>
                </div>
            </div>
        </div>

        <div class="doc-section">
            <h2>3.0 Score Trajectories (S_total Evolution)</h2>
            <div style="position:relative; height:340px; width:100%;">
                <canvas id="multiTimelineChart"></canvas>
            </div>
        </div>

        <div class="doc-section">
            <h2>4.0 Mathematical Function Morphing Animator</h2>
            <div class="morph-wrapper">
                <div style="margin-bottom:0.8rem;">
                    <label style="color:#38bdf8; font-weight:bold; font-size:0.9rem;">Target Engine: </label>
                    <select id="selectMorphRun"></select>
                </div>
                <canvas id="morphCanvas" class="morph-canvas" width="620" height="190"></canvas>
                <div class="morph-controls">
                    <button class="btn btn-primary" id="btnMorphPlay">▶ Play Morphing</button>
                    <input type="range" id="morphSlider" min="1" max="__ACTUAL_MAX_GEN__" value="1" style="flex-grow:1;">
                    <span style="font-size:0.85rem; color:#38bdf8; font-weight:bold; min-width:60px;" id="morphGenLabel">Gen 1</span>
                </div>
            </div>
        </div>

        <div class="doc-section">
            <h2>5.0 Generation Dossiers</h2>
            __DETAILED_RUNS_HTML__
        </div>

        <div class="doc-section">
            <h2>6.0 Cumulative Quarantine Log</h2>
            <pre class="code-box" style="color:#f43f5e;">__REFLECTIONS_LOG__</pre>
        </div>
    </div>

    <!-- TAB 2: HACKING & CUSTOMIZATION GUIDE -->
    <div id="main-tab-custom" class="main-tab-content">
        <div class="doc-section">
            <h2>🛠️ Experiment Hacking &amp; Parameter Tuning Guide</h2>
            <p>Every aspect of the exploration loop is fully open and modular. You can easily modify physical hyperparameters, simulation grids, fitness weightings, and prompt philosophies in <code>run.py</code>.</p>

            <h3>1. Modifying Simulation Hyperparameters</h3>
            <p>The core continuous automaton dynamics are governed by the <code>LeniaSimulator</code> initialization in <code>run.py</code>:</p>
            <pre class="code-box" style="color:#e2e8f0;"><code># Located inside run.py -> main() and LeniaSimulator.__init__()
sim = LeniaSimulator(
    size=100,        # Spatial grid resolution (100x100 torus)
    dt=0.1,          # Numerical integration timestep Δt (smaller = more stable, larger = faster)
    max_steps=80,    # Number of simulation steps per evaluation (80 steps ≈ 8 seconds of continuous time)
    radius=13        # Neighborhood kernel radius R (in grid cells)
)</code></pre>
            <ul>
                <li><strong>Increase <code>max_steps</code> (e.g., 120 - 160):</strong> Forces the LLM to search for long-term thermodynamic stability rather than short-lived transient solitons.</li>
                <li><strong>Adjust <code>dt</code> (e.g., 0.05):</strong> Resolves numerical stiffness when evaluating steep algebraic growth gradients.</li>
            </ul>

            <h3>2. Customizing Initial Fields (Breaking Symmetry)</h3>
            <p>The initial state generator is located in <code>LeniaSimulator.init_state()</code>. You can customize the shape to test different topological seeds:</p>
            <pre class="code-box" style="color:#e2e8f0;"><code>def init_state(self):
    y, x = np.ogrid[: self.size, : self.size]
    dx = x - self.mid
    dy = y - self.mid
    r = np.sqrt(dx**2 + dy**2) / 8.0

    # 1. Base localized Gaussian spot
    base = np.exp(-0.5 * (r - 0.7) ** 2 / (0.35**2))
    
    # 2. Directional polarity modulation (Crescent Wavefront)
    theta = np.arctan2(dy, dx)
    polarity = 0.5 + 0.5 * np.cos(theta - np.pi / 4.0)

    A = base * (polarity ** 1.5)
    return np.clip(np.where(r > 1.8, 0.0, A), 0.0, 1.0)</code></pre>

            <h3>3. Adjusting the Fitness Function &amp; Mobility Penalty</h3>
            <p>The multi-objective reward is evaluated in <code>compute_meta_score_detailed()</code>:</p>
            <pre class="code-box" style="color:#e2e8f0;"><code># S_viability: Penalizes mass explosion or starvation away from mass_ratio = 1.0
s_viability = 100.0 * (step_ratio * (0.6 + 0.4 * mass_balance))

# S_mobility: Hyperbolic tangent mapping of center-of-mass net displacement
s_mobility = 5.0 + 95.0 * math.tanh(math.sqrt(disp) / 1.5)

# Total Multiplicative Fitness
s_total = (s_viability * s_mobility) / 100.0</code></pre>
            <div class="analogy-box">
                <strong>Tuning Tip:</strong> If the model creates solitons that move fast but inflate in mass, sharpen the mass penalty in <code>mass_balance = max(0.0, 1.0 - 2.0 * abs(mass_ratio - 1.0))</code>.
            </div>

            <h3>4. Customizing the LLM System Prompt &amp; Strategies</h3>
            <p>The strategy guidelines and AST execution rules are stored in <code>P4_PHILOSOPHY</code> and <code>SYSTEM_PROMPT</code> at the top of <code>run.py</code>. You can introduce new physical concepts (e.g., fluid vortex dynamics, advection terms, or soliton collision resilience) directly into the prompt.</p>
        </div>
    </div>

    <!-- TAB 3: ARCHITECTURE GUIDE -->
    <div id="main-tab-guide" class="main-tab-content">
        <div class="doc-section">
            <h2>📘 Continuous Lenia Governing Equations</h2>
            <p>Continuous Lenia generalizes discrete cellular automata to a smooth continuous space and time continuum:</p>
            <div class="formula-box">
                $$A(x, t + \Delta t) = \operatorname{clip}\left( A(x, t) + \Delta t \cdot G( (K * A)(x, t) ), \; 0, \; 1 \right)$$
            </div>
            <p>Under Strategy P4 (Structural Skeleton), the kernel $K(r)$ and growth function $G(u)$ are constructed via functional composition to create sharp wavefront activation and trailing decay.</p>
        </div>
    </div>
</div>

<script>
window.addEventListener('DOMContentLoaded', () => {
    const MATRIX_DATA = __MATRIX_DATA_JSON__;

    function renderMathSafely(element) {
        if (window.renderMathInElement) {
            renderMathInElement(element, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false}
                ],
                throwOnError: false
            });
        }
    }

    window.switchMainTab = function(tabKey) {
        document.querySelectorAll('.main-tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.main-tab-content').forEach(c => c.classList.remove('active'));
        if (tabKey === 'report') {
            document.getElementById('btnMainReport').classList.add('active');
            document.getElementById('main-tab-report').classList.add('active');
        } else if (tabKey === 'custom') {
            document.getElementById('btnMainCustom').classList.add('active');
            document.getElementById('main-tab-custom').classList.add('active');
        } else {
            document.getElementById('btnMainGuide').classList.add('active');
            document.getElementById('main-tab-guide').classList.add('active');
            renderMathSafely(document.getElementById('main-tab-guide'));
        }
    };

    window.jumpToRun = function(runId) {
        const target = document.getElementById('details_' + runId);
        if (target) {
            target.open = true;
            target.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    };

    // Compare View
    const selRunA = document.getElementById('selectRunA');
    const selRunB = document.getElementById('selectRunB');
    const selMorph = document.getElementById('selectMorphRun');

    Object.keys(MATRIX_DATA).forEach(runId => {
        selRunA.appendChild(new Option(runId, runId));
        selRunB.appendChild(new Option(runId, runId));
        selMorph.appendChild(new Option(runId, runId));
    });

    if (selRunB.options.length > 1) selRunB.selectedIndex = 1;

    function updateGenOptions(runId, genSelectId) {
        const genSelect = document.getElementById(genSelectId);
        genSelect.innerHTML = '';
        const run = MATRIX_DATA[runId];
        if (run && run.generations) {
            run.generations.forEach(g => {
                genSelect.appendChild(new Option('Gen ' + g.gen + ' (' + g.status + ')', g.gen));
            });
        }
    }

    selRunA.onchange = () => { updateGenOptions(selRunA.value, 'selectGenA'); updateCompareView(); };
    selRunB.onchange = () => { updateGenOptions(selRunB.value, 'selectGenB'); updateCompareView(); };

    updateGenOptions(selRunA.value, 'selectGenA');
    updateGenOptions(selRunB.value, 'selectGenB');

    function updateCompareView() {
        const runAId = selRunA.value;
        const genAVal = parseInt(document.getElementById('selectGenA').value) || 1;
        const itemA = (MATRIX_DATA[runAId]?.generations || []).find(g => g.gen === genAVal);

        if (itemA) {
            document.getElementById('titleA').innerText = runAId + ' (Gen ' + itemA.gen + ')';
            document.getElementById('metaA').innerText = 'S_total: ' + itemA.score.toFixed(1) + ' | Move: ' + itemA.displacement.toFixed(2) + 'px | ' + itemA.status;
            document.getElementById('gifA').src = itemA.gif_b64 ? 'data:image/gif;base64,' + itemA.gif_b64 : '';
            document.getElementById('plotA').src = itemA.plot_b64 ? 'data:image/png;base64,' + itemA.plot_b64 : '';
            document.getElementById('thoughtA').innerHTML = '<strong>LLM Thought:</strong><br/>' + (itemA.thought || '').replace(/\n/g, '<br/>');
            document.getElementById('mathA').innerHTML = '<div style="color:#38bdf8;">$$' + itemA.latex_kernel + '$$</div><div style="color:#34d399; margin-top:4px;">$$' + itemA.latex_growth + '$$</div>';
            renderMathSafely(document.getElementById('mathA'));
        }

        const runBId = selRunB.value;
        const genBVal = parseInt(document.getElementById('selectGenB').value) || 1;
        const itemB = (MATRIX_DATA[runBId]?.generations || []).find(g => g.gen === genBVal);

        if (itemB) {
            document.getElementById('titleB').innerText = runBId + ' (Gen ' + itemB.gen + ')';
            document.getElementById('metaB').innerText = 'S_total: ' + itemB.score.toFixed(1) + ' | Move: ' + itemB.displacement.toFixed(2) + 'px | ' + itemB.status;
            document.getElementById('gifB').src = itemB.gif_b64 ? 'data:image/gif;base64,' + itemB.gif_b64 : '';
            document.getElementById('plotB').src = itemB.plot_b64 ? 'data:image/png;base64,' + itemB.plot_b64 : '';
            document.getElementById('thoughtB').innerHTML = '<strong>LLM Thought:</strong><br/>' + (itemB.thought || '').replace(/\n/g, '<br/>');
            document.getElementById('mathB').innerHTML = '<div style="color:#38bdf8;">$$' + itemB.latex_kernel + '$$</div><div style="color:#34d399; margin-top:4px;">$$' + itemB.latex_growth + '$$</div>';
            renderMathSafely(document.getElementById('mathB'));
        }
    }

    document.getElementById('selectGenA').onchange = updateCompareView;
    document.getElementById('selectGenB').onchange = updateCompareView;
    updateCompareView();

    // Chart.js Timeline Fix
    try {
        const colors = ['#38bdf8', '#34d399', '#f59e0b', '#f43f5e'];
        const datasets = [];
        let cIdx = 0;
        let maxGensCount = 1;
        Object.keys(MATRIX_DATA).forEach(runId => {
            const run = MATRIX_DATA[runId];
            const gCount = (run.generations || []).length;
            if (gCount > maxGensCount) maxGensCount = gCount;
            datasets.push({
                label: runId,
                data: (run.generations || []).map(g => g.score),
                borderColor: colors[cIdx % colors.length],
                borderWidth: 2,
                fill: false,
                tension: 0.2
            });
            cIdx++;
        });

        const ctx = document.getElementById('multiTimelineChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({length: maxGensCount}, (_, i) => 'Gen ' + (i + 1)),
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { color: '#1a233a' }, ticks: { color: '#94a3b8' } },
                    y: { grid: { color: '#1a233a' }, ticks: { color: '#38bdf8' }, title: { display: true, text: 'Total Score S_total', color: '#38bdf8' } }
                }
            }
        });
    } catch(e) { console.error("Chart rendering error:", e); }

    // Morph Canvas
    const morphCanvas = document.getElementById('morphCanvas');
    const mCtx = morphCanvas.getContext('2d');
    const morphSlider = document.getElementById('morphSlider');
    const morphGenLabel = document.getElementById('morphGenLabel');
    const btnMorphPlay = document.getElementById('btnMorphPlay');
    let morphTimer = null;
    let currentGenFloat = 1.0;

    function setupMorphRun() {
        currentGenFloat = 1.0;
        morphSlider.value = 1;
        morphGenLabel.innerText = "Gen 1";
        drawMorph(1.0);
    }

    function drawMorph(genFloat) {
        const runId = selMorph.value;
        const gens = MATRIX_DATA[runId]?.generations || [];
        if (gens.length === 0) return;

        const clamped = Math.max(1, Math.min(gens.length, genFloat));
        const idxA = Math.floor(clamped) - 1;
        const idxB = Math.min(idxA + 1, gens.length - 1);
        const alpha = clamped - Math.floor(clamped);

        const kA = gens[idxA].k_curve || [0];
        const kB = gens[idxB].k_curve || [0];
        const gA = gens[idxA].g_curve || [-1];
        const gB = gens[idxB].g_curve || [-1];

        mCtx.fillStyle = '#020617';
        mCtx.fillRect(0, 0, morphCanvas.width, morphCanvas.height);

        // Kernel Curve
        mCtx.strokeStyle = '#38bdf8';
        mCtx.lineWidth = 2;
        mCtx.beginPath();
        for (let i = 0; i < kA.length; i++) {
            const v = (kA[i] || 0) * (1 - alpha) + (kB[i] || 0) * alpha;
            const x = 30 + (i / (kA.length - 1)) * 250;
            const y = 165 - Math.min(Math.max(v, 0), 2.5) * 60;
            if (i === 0) mCtx.moveTo(x, y); else mCtx.lineTo(x, y);
        }
        mCtx.stroke();

        // Growth Curve
        mCtx.strokeStyle = '#34d399';
        mCtx.lineWidth = 2;
        mCtx.beginPath();
        for (let i = 0; i < gA.length; i++) {
            const v = (gA[i] || -1) * (1 - alpha) + (gB[i] || -1) * alpha;
            const x = 340 + (i / (gA.length - 1)) * 250;
            const y = 100 - v * 60;
            if (i === 0) mCtx.moveTo(x, y); else mCtx.lineTo(x, y);
        }
        mCtx.stroke();
    }

    morphSlider.oninput = function() {
        currentGenFloat = parseFloat(this.value);
        morphGenLabel.innerText = "Gen " + Math.round(currentGenFloat);
        drawMorph(currentGenFloat);
    };

    btnMorphPlay.onclick = function() {
        const runId = selMorph.value;
        const maxGen = (MATRIX_DATA[runId]?.generations || []).length || 30;
        if (morphTimer) {
            clearInterval(morphTimer);
            morphTimer = null;
            this.innerText = '▶ Play Morphing';
        } else {
            this.innerText = '⏸ Pause';
            morphTimer = setInterval(() => {
                currentGenFloat += 0.25;
                if (currentGenFloat > maxGen) currentGenFloat = 1.0;
                morphSlider.value = currentGenFloat;
                morphGenLabel.innerText = "Gen " + Math.round(currentGenFloat);
                drawMorph(currentGenFloat);
            }, 50);
        }
    };

    selMorph.onchange = setupMorphRun;
    setupMorphRun();

    // Initial Math Render
    renderMathSafely(document.body);
});
</script>
</body>
</html>
"""


# =====================================================================
# 7. Main Execution Loop (14B Dedicated Engine)
# =====================================================================
def main():
    global INTERRUPT_REQUESTED
    client = ollama.Client()

    cfg = select_execution_mode(client)

    sim = LeniaSimulator(size=100, dt=0.1, max_steps=cfg["max_sim_steps"], radius=13)

    start_time = time.time()
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    matrix_results = {}
    all_quarantine_logs = []

    models = cfg["models"]
    gens_per_run = cfg["gens_per_run"]
    timeout_sec = cfg["timeout_sec"]
    report_file = "index.html"

    total_runs = len(models)
    run_idx = 0
    actual_max_gen = 1

    print("=" * 78)
    print(f"🚀 LAUNCHING: {cfg['mode_name']}")
    print(f"   Target Strategy: P4 Structural Skeleton")
    print(f"   Dedicated Engine: {models[0]}")
    print(f"   Max Generations: {gens_per_run} | Sim Steps: {cfg['max_sim_steps']}")
    print("   💡 TIP: Hit Ctrl+C at any time to compile 'index.html' and exit safely.")
    print("=" * 78)

    for model_name in models:
        if INTERRUPT_REQUESTED or (time.time() - start_time >= timeout_sec):
            break

        run_idx += 1
        run_id = f"{model_name.replace(':', '_')}__P4_StructuralSkeleton"
        print("\n" + "-" * 78)
        print(f"▶️  [RUN {run_idx:02d}/{total_runs:02d}]: {run_id}")
        print(f"    Model Engine: {model_name}")
        print("-" * 78)

        run_reflections = []
        run_generations = []
        recent_history = []
        champion_record = None

        last_feedback = (
            "Initial Generation: The grid contains an asymmetric crescent wavefront with a steep front gradient.\n"
            "Formulate an initial kernel K(r) and growth function G(u) under P4 Structural Skeleton.\n"
            "Goal: Maximize Total Score S_total = (S_viability * S_mobility) / 100.0 towards 100.0."
        )

        for gen in range(1, gens_per_run + 1):
            if INTERRUPT_REQUESTED or (time.time() - start_time >= timeout_sec):
                print(f"\n🛑 Halting exploration loop at Gen {gen} on user request / timeout.")
                break

            if gen > actual_max_gen:
                actual_max_gen = gen

            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elapsed_str = format_elapsed(time.time() - start_time)
            overall_pct = (run_idx - 1 + (gen - 1) / gens_per_run) / total_runs * 100

            print(
                f"\n  [{now_str} | Elapsed: {elapsed_str} | Run {run_idx}/{total_runs} | Gen {gen:02d}/{gens_per_run:02d} ({overall_pct:.1f}%)]"
            )
            print("  Querying LLM with Dual-Memory...")

            code_accepted = False
            thought = ""
            full_code = ""
            latex_k = ""
            latex_g = ""
            k_fn = None
            g_fn = None
            retry = 0

            while not code_accepted and retry < 5 and not INTERRUPT_REQUESTED:
                reflections_context = "\n".join(run_reflections[-3:]) if run_reflections else "None."

                user_prompt = f"""[Strategy Guideline]
{P4_PHILOSOPHY['philosophy']}

{last_feedback}

[Regime Quarantine Memory]
{reflections_context}

[Instruction]
Formulate `kernel_func(r)` and `growth_func(u)` for Generation {gen}.
Both must accept EXACTLY ONE parameter (`r` or `u`).
1. `kernel_func(r)` MUST be non-negative everywhere (K(r) >= 0.0).
2. The Rule of the Void (G(0) <= -0.5) applies ONLY to `growth_func(u)`.
3. Optimize S_total based on the Champion Seed and Recent Trajectory!"""

                try:
                    res = client.chat(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        format="json",
                        options={"temperature": 0.75},
                    )
                    data = json.loads(res["message"]["content"])
                    thought_raw = data.get("thought", "- No thought provided")
                    thought = (
                        "\n".join(f"- {item.lstrip('- ')}" for item in thought_raw)
                        if isinstance(thought_raw, list)
                        else str(thought_raw)
                    )
                    k_code = data.get("kernel_code", "")
                    g_code = data.get("growth_code", "")
                    latex_k = data.get("latex_kernel", "K(r)")
                    latex_g = data.get("latex_growth", "G(u)")
                    full_code = f"{k_code}\n\n{g_code}"
                except Exception as e:
                    err_msg = f"JSON/API error: {e}"
                    run_reflections.append(err_msg)
                    all_quarantine_logs.append(f"[{run_id}|Gen {gen}] {err_msg}")
                    retry += 1
                    continue

                try:
                    verify_code_safety(full_code)
                except (SecurityError, SyntaxError) as err:
                    err_msg = f"Validation Error: {err}"
                    run_reflections.append(err_msg)
                    all_quarantine_logs.append(f"[{run_id}|Gen {gen}] {err_msg}")
                    retry += 1
                    continue

                exec_env = {"np": np}
                try:
                    exec(full_code, exec_env)
                    if "kernel_func" not in exec_env or "growth_func" not in exec_env:
                        raise KeyError("Both `kernel_func` and `growth_func` must be defined.")

                    k_fn = exec_env["kernel_func"]
                    g_fn = exec_env["growth_func"]

                    dummy_r = np.linspace(0.0, 1.0, 20)
                    dummy_u = np.linspace(0.0, 1.0, 20)
                    out_k = k_fn(dummy_r)
                    out_g = g_fn(dummy_u)

                    if not isinstance(out_k, (np.ndarray, float, int)) or not isinstance(
                        out_g, (np.ndarray, float, int)
                    ):
                        raise TypeError("Functions must return NumPy arrays or scalars.")

                    code_accepted = True
                except Exception as run_err:
                    err_msg = f"Dry-run Failed: {run_err}"
                    run_reflections.append(err_msg)
                    all_quarantine_logs.append(f"[{run_id}|Gen {gen}] {err_msg}")
                    retry += 1

            first_thought = thought.split("\n")[0] if thought else "None"
            print(f"    💡 Thought: {first_thought[:85]}...")

            if not code_accepted:
                print(f"    ❌ Gen {gen}: Marked INVALID.")
                res_sim = {
                    "status": "INVALID",
                    "step": 0,
                    "displacement": 0.0,
                    "init_mass": 0.0,
                    "final_mass": 0.0,
                    "mass_trend": "Dry-run/Syntax failure",
                    "frames": [],
                    "reason": "Failed dry-run validation.",
                }
                score_dict = {
                    "s_viability": 0.0,
                    "s_mobility": 5.0,
                    "displacement": 0.0,
                    "s_total": 0.0,
                }
                plot_b64 = ""
                gif_b64 = ""
                k_curve, g_curve = [0] * 100, [-1] * 100
            else:
                res_sim = sim.run(k_fn, g_fn)
                score_dict = compute_meta_score_detailed(res_sim, cfg["max_sim_steps"])
                plot_b64 = plot_to_base64(k_fn, g_fn)
                gif_b64 = frames_to_gif_base64(res_sim["frames"])
                k_curve, g_curve = sample_curves(k_fn, g_fn, n_samples=100)

            status_icon = "🟢" if res_sim["status"] == "ALIVE" else ("🟡" if "STAGNANT" in res_sim["status"] else "🔴")
            print(
                f"    {status_icon} Result: [{res_sim['status']}] | Move: {score_dict['displacement']:.2f}px | "
                f"S_total: {score_dict['s_total']:.1f} (V: {score_dict['s_viability']:.1f}, M: {score_dict['s_mobility']:.1f})"
            )
            print(f"       Dynamics: {res_sim['mass_trend']}")

            is_valid_champion = res_sim["status"] == "ALIVE" and score_dict["displacement"] >= 1.0
        if is_valid_champion and (champion_record is None or score_dict["s_total"] > champion_record["score"]):
                champion_record = {
                    "gen": gen,
                    "score": score_dict["s_total"],
                    "displacement": score_dict["displacement"],
                    "v_score": score_dict["s_viability"],
                    "m_score": score_dict["s_mobility"],
                    "code": full_code,
                    "latex_k": latex_k,
                    "latex_g": latex_g,
                }
                print(
                    f"    🏆 NEW BEST RECORD IN RUN! S_total: {score_dict['s_total']:.1f} | Move: {score_dict['displacement']:.2f}px"
                )

            summary_line = (
                f"Gen {gen:02d}: Move={score_dict['displacement']:.2f}px, "
                f"S_total={score_dict['s_total']:.1f} (V={score_dict['s_viability']:.1f}, M={score_dict['s_mobility']:.1f}), "
                f"Status={res_sim['status']}"
            )
            recent_history.append(summary_line)
            if len(recent_history) > 5:
                recent_history.pop(0)

            last_feedback = format_generation_feedback(
                res_sim,
                score_dict,
                gen,
                cfg["max_sim_steps"],
                champion_record,
                recent_history,
            )

            run_generations.append(
                {
                    "gen": gen,
                    "thought": thought,
                    "code": full_code,
                    "status": res_sim["status"],
                    "displacement": score_dict["displacement"],
                    "score": score_dict["s_total"],
                    "s_viability": score_dict["s_viability"],
                    "s_mobility": score_dict["s_mobility"],
                    "mass_trend": res_sim["mass_trend"],
                    "reason": res_sim["reason"],
                    "plot_b64": plot_b64,
                    "gif_b64": gif_b64,
                    "k_curve": k_curve,
                    "g_curve": g_curve,
                    "latex_kernel": latex_k,
                    "latex_growth": latex_g,
                }
            )

        best_score = max([g["score"] for g in run_generations], default=0.0)
        max_move = max([g["displacement"] for g in run_generations], default=0.0)
        alive_count = sum(1 for g in run_generations if g["status"] == "ALIVE")
        stagnant_count = sum(1 for g in run_generations if "STAGNANT" in g["status"])

        matrix_results[run_id] = {
            "model": model_name,
            "strategy": "P4_StructuralSkeleton",
            "title": P4_PHILOSOPHY["title"],
            "best_score": best_score,
            "max_move": max_move,
            "alive_count": alive_count,
            "stagnant_count": stagnant_count,
            "final_status": run_generations[-1]["status"] if run_generations else "NONE",
            "generations": run_generations,
        }
        gc.collect()

    end_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_elapsed = time.time() - start_time
    total_elapsed_str = format_elapsed(total_elapsed)
    total_timeout_str = format_elapsed(timeout_sec)

    executive_review = generate_executive_review(client, models[0], matrix_results)

    # Compile Final HTML (to index.html directly)
    print("\n📊 Assembling and exporting 'index.html'...")
    matrix_rows_html = ""
    for model_name in models:
        run_id = f"{model_name.replace(':', '_')}__P4_StructuralSkeleton"
        res = matrix_results.get(run_id)
        if res:
            color = (
                "#10b981"
                if res["max_move"] >= 3.0
                else ("#38bdf8" if res["max_move"] > 0.05 else ("#f59e0b" if res["stagnant_count"] > 0 else "#6b7280"))
            )
            matrix_rows_html += f"""
            <tr>
                <td style='font-weight:bold; color:#38bdf8;'>{model_name}</td>
                <td>
                    <div class="matrix-cell" onclick="jumpToRun('{run_id}')">
                        <div style="font-weight:bold; color:{color}; font-size:1.1rem;">{res['max_move']:.2f} px</div>
                        <div style="font-size:0.75rem; color:#94a3b8;">Score: {res['best_score']:.1f}</div>
                        <div style="font-size:0.72rem; background:#1a233a; padding:0.15rem 0.4rem; border-radius:4px; margin-top:0.3rem; color:#e2e8f0;">{res['final_status']}</div>
                    </div>
                </td>
            </tr>
            """

    detailed_runs_html = ""
    for run_id, res in matrix_results.items():
        cards_html = ""
        for g in res["generations"]:
            color = {
                "ALIVE": "#10b981",
                "DRIFTING": "#38bdf8",
                "STAGNANT": "#f59e0b",
                "DEAD": "#ef4444",
                "EXPLODED": "#dc2626",
            }.get(g["status"], "#6b7280")

            gif_tag = (
                f'<img src="data:image/gif;base64,{g["gif_b64"]}"/>'
                if g["gif_b64"]
                else '<div style="color:#64748b;padding:2rem;">No Simulation Video</div>'
            )
            plot_tag = (
                f'<img src="data:image/png;base64,{g["plot_b64"]}"/>'
                if g["plot_b64"]
                else '<div style="color:#64748b;padding:2rem;">No Function Plot</div>'
            )

            cards_html += f"""
            <div class="gen-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                    <strong style="color:#f8fafc; font-size:1.1rem;">Generation {g['gen']}</strong>
                    <span style="background:{color}; font-size:0.75rem; padding:0.2rem 0.6rem; border-radius:9999px; color:#fff; font-weight:bold;">{g['status']}</span>
                    <span style="color:#94a3b8; font-size:0.85rem;">Move: <strong>{g['displacement']:.2f}px</strong> | S_total: <strong>{g['score']:.1f}</strong> (V: {g.get('s_viability', 0):.1f}, M: {g.get('s_mobility', 0):.1f})</span>
                </div>

                <div style="background:#111726; padding:0.7rem 1rem; border-radius:6px; border-left:3px solid #38bdf8; font-size:0.85rem; margin-bottom:0.6rem;">
                    <strong style="color:#38bdf8;">Hypothesis:</strong>
                    <div style="color:#cbd5e1; margin-top:0.2rem; white-space:pre-wrap;">{html.escape(g['thought'])}</div>
                </div>

                <div style="background:#111726; padding:0.6rem 1rem; border-radius:6px; margin-bottom:0.6rem; border-left:3px solid #10b981;">
                    <div style="font-size:0.85rem; color:#38bdf8;">$${html.escape(g['latex_kernel'])}$$</div>
                    <div style="font-size:0.85rem; color:#34d399; margin-top:0.2rem;">$${html.escape(g['latex_growth'])}$$</div>
                </div>

                <div class="gen-grid">
                    <div>
                        <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:0.2rem;">Mathematical Functions</div>
                        {plot_tag}
                    </div>
                    <div>
                        <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:0.2rem;">Dynamic Simulation</div>
                        {gif_tag}
                    </div>
                </div>

                <div style="font-size:0.82rem; color:#94a3b8; margin-top:0.4rem;">
                    <strong>Dynamics:</strong> {html.escape(g['mass_trend'])} | <strong>Outcome:</strong> {html.escape(g['reason'])}
                </div>

                <details style="margin-top:0.5rem; background:#070a12; border:1px solid #1f2a40; padding:0.4rem 0.8rem;">
                    <summary style="font-size:0.75rem; color:#64748b;">View Generated Python Code</summary>
                    <pre class="code-box" style="margin:0.4rem 0 0 0; font-size:0.75rem;"><code>{html.escape(g['code'])}</code></pre>
                </details>
            </div>
            """

        detailed_runs_html += f"""
        <details id="details_{run_id}">
            <summary>{res['model']} — {res['title']} (Best Score: {res['best_score']:.1f} | Max Move: {res['max_move']:.2f}px | Final: {res['final_status']})</summary>
            <div style="padding-top:0.8rem;">
                {cards_html}
            </div>
        </details>
        """

    final_html = HTML_TEMPLATE
    replacements = {
        "__MODE_NAME__": cfg["mode_name"],
        "__START_TIME__": start_time_str,
        "__END_TIME__": end_time_str,
        "__ELAPSED_SEC__": total_elapsed_str,
        "__TOTAL_TIMEOUT__": total_timeout_str,
        "__GENS_PER_RUN__": str(gens_per_run),
        "__ACTUAL_MAX_GEN__": str(actual_max_gen),
        "__GRID_SIZE__": "100",
        "__MAX_SIM_STEPS__": str(cfg["max_sim_steps"]),
        "__TIME_STEP_DT__": "0.1",
        "__KERNEL_RADIUS__": "13",
        "__EXECUTIVE_REVIEW__": html.escape(executive_review),
        "__MATRIX_TABLE_ROWS__": matrix_rows_html,
        "__DETAILED_RUNS_HTML__": detailed_runs_html,
        "__REFLECTIONS_LOG__": html.escape(
            "\n".join(all_quarantine_logs) if all_quarantine_logs else "No violations logged."
        ),
        "__MATRIX_DATA_JSON__": json.dumps(matrix_results),
    }

    for k, v in replacements.items():
        final_html = final_html.replace(k, v)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(final_html)

    print("\n" + "=" * 78)
    print(f"🎉 P4 EVOLUTION RUN COMPLETED!")
    print(f"   Report Generated: {os.path.abspath(report_file)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
