"""Command-line interface for the quantization benchmark."""

from __future__ import annotations

import argparse
import logging
import sys

from model_quantization_lab.core import QuantizationBenchmark
from model_quantization_lab.models import BenchmarkConfig, QuantizationConfig, QuantizationMethod

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Model Quantization Lab -- compare quantization methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dim", type=int, default=256, help="Model hidden dimension"
    )
    parser.add_argument(
        "--num-layers", type=int, default=4, help="Number of transformer layers"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=1000, help="Vocabulary size"
    )
    parser.add_argument(
        "--seq-length", type=int, default=128, help="Sequence length"
    )
    parser.add_argument(
        "--eval-samples", type=int, default=64, help="Number of evaluation samples"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the quantization benchmark.

    Args:
        argv: Optional argument list.

    Returns:
        Exit code.
    """
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = BenchmarkConfig(
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        seq_length=args.seq_length,
        eval_samples=args.eval_samples,
        seed=args.seed,
    )

    benchmark = QuantizationBenchmark(config)
    benchmark.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
