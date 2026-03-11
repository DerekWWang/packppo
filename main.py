"""SC-MAPPO entry point.

Delegates to train.main() which parses CLI args via tyro.

Run:
    uv run python main.py                    # Phase 1 training (all defaults)
    uv run python main.py --help             # show all options
    uv run python train.py --phase 2 ...    # direct access to trainer

For evaluation / rendering:
    uv run python evaluate.py --checkpoint runs/.../ckpt_final.pt
"""
from train import main

if __name__ == "__main__":
    main()
