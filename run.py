import argparse
import json
import time
from argparse import Namespace
from pathlib import Path

import yaml

from train import train


MODEL_NAME = "SGD-DyG-M1-prior-conditioned-lora-PL-DisGLSL"


def load_config(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def coerce_value(raw):
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        if any(ch in raw for ch in (".", "e", "E")):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def resolve_config_path(project_dir, dataset_name, config_path):
    if config_path is not None:
        return config_path

    selected_dataset = dataset_name or "wiki_gl"
    return project_dir / "configs" / f"{selected_dataset}.yaml"


def parse_overrides(items):
    overrides = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Use key=value.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{item}'. Empty key.")
        overrides[key] = coerce_value(raw_value.strip())
    return overrides


def build_args(config, overrides):
    if not isinstance(config, dict):
        raise ValueError("Each dataset config must be a YAML mapping.")

    train_args = dict(config)
    unknown_keys = sorted(set(overrides) - set(train_args))
    if unknown_keys:
        raise ValueError(f"Unsupported override(s): {', '.join(unknown_keys)}")

    train_args.update(overrides)
    train_args["model_name"] = MODEL_NAME
    train_args["hidden_features"] = [train_args["hidden_feature"]] * train_args["layer"]
    return Namespace(**train_args)


def parse_args():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train M1 prior-conditioned LoRA PL-DisGLSL.")
    parser.add_argument("--config", type=Path, default=None, help="Path to a dataset YAML config file.")
    parser.add_argument("--output_dir", type=Path, default=project_dir / "logs")
    parser.add_argument("--dataset_name", type=str, default=None, help="Load configs/<dataset_name>.yaml.")
    parser.add_argument("--set", action="append", default=[], help="Override a training parameter, e.g. --set epochs=20.")
    return parser.parse_args()


def main():
    cli_args = parse_args()
    overrides = parse_overrides(cli_args.set)
    project_dir = Path(__file__).resolve().parent
    dataset_name = cli_args.dataset_name or overrides.get("dataset_name")
    config_path = resolve_config_path(project_dir, dataset_name, cli_args.config)
    config = load_config(config_path)
    args = build_args(config, overrides)

    if cli_args.dataset_name is not None and args.dataset_name != cli_args.dataset_name:
        raise ValueError(
            f"Config dataset_name is '{args.dataset_name}', but --dataset_name is '{cli_args.dataset_name}'."
        )

    output_dir = cli_args.output_dir / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    results = train(args)
    elapsed = time.perf_counter() - start

    summary = {
        "model_name": MODEL_NAME,
        "dataset_name": args.dataset_name,
        "config": str(config_path),
        "seconds": elapsed,
        "args": vars(args),
        "runs": results,
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Training finished in {elapsed:.2f}s.")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
