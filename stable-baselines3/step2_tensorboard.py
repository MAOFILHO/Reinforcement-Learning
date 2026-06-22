"""
STEP 2 — Launch TensorBoard to visualize training
==================================================
After running step1, open TensorBoard to inspect live training curves.

Run:
    python step2_tensorboard.py

Then open: http://localhost:6006

What to screenshot for portfolio:
  - ep_rew_mean  (reward climbing toward 500)
  - train/entropy_loss
  - train/value_loss
  - Overview page after step4 (shows both PPO and A2C runs)
"""

import os
import subprocess
import sys

LOGS_DIR = os.path.abspath("logs")

# ── Verify TensorBoard is installed ───────────────────────────────────────────
try:
    import tensorboard
    print(f"✅ TensorBoard {tensorboard.__version__} is installed.")
except ImportError:
    print("❌ TensorBoard not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])
    print("✅ Installed.")

# ── Verify logs folder exists ─────────────────────────────────────────────────
if not os.path.isdir(LOGS_DIR):
    print(f"\n⚠️  No logs/ folder found at {LOGS_DIR}")
    print("   Run step1_cartpole_ppo.py first to generate training logs.")
    sys.exit(1)

runs = [d for d in os.listdir(LOGS_DIR) if os.path.isdir(os.path.join(LOGS_DIR, d))]
print(f"\n📂 Logs folder → {LOGS_DIR}")
print(f"   Found {len(runs)} run(s): {', '.join(runs)}")

# ── Launch TensorBoard ────────────────────────────────────────────────────────
print("\n🚀 Launching TensorBoard...")
print("   URL → http://localhost:6006")
print("   Press Ctrl+C to stop.\n")

subprocess.run(["tensorboard", "--logdir", LOGS_DIR])
