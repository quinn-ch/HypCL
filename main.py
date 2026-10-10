"""Train one of the 18 HypCL main experiments."""
import argparse
from datetime import datetime
from pathlib import Path

from utils.config import Experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        "--output-dir",
        type=Path,
        help="A new directory for train.log",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--download", action="store_true", help="Allow downloading CIFAR100"
    )
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    config = Experiment.load(args.config)
    output = args.output or Path("outputs") / (
        args.config.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    from trainer import train

    train(config, args.data_root, output, args.device, args.workers, args.download)


if __name__ == "__main__":
    main()
