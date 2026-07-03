from __future__ import annotations

import argparse

from deploy_mujoco_env import (
    DEFAULT_CONFIG,
    DEFAULT_FSM_CONFIG,
    DEFAULT_XML,
    run_deploy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified G1 MuJoCo deploy")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to deploy.yaml")
    parser.add_argument(
        "--fsm-config",
        type=str,
        default=str(DEFAULT_FSM_CONFIG),
        help="FSM config for mimic motion lookup",
    )
    parser.add_argument("--xml-path", type=str, default=DEFAULT_XML)
    parser.add_argument("--simulation-dt", type=float, default=0.001)
    parser.add_argument("--simulation-duration", type=float, default=600.0)
    parser.add_argument("--device", type=str, default="cpu", help="TorchScript device when exported/policy.pt is used")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_deploy(args.config, args)


if __name__ == "__main__":
    main()
