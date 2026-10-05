"""Entry point for python -m mint.five.fig_atlas."""

import argparse
from mint.five.fig_atlas.compute import run_compute
from mint.five.fig_atlas.plot import run_plot


def main():
    parser = argparse.ArgumentParser(
        description="Figure: MINT Atlas of Pediatric ED Visits & CDR Optimization"
    )
    parser.add_argument(
        "--mode", choices=["compute", "plot", "name", "all"], default="all",
        help="compute = cluster + UMAP + PEWS (both splits); plot = generate panels; "
             "name = LLM cluster naming; all = compute then plot"
    )
    parser.add_argument(
        "--panel", choices=["a", "b", "d"], default=None,
        help="Plot only a specific panel"
    )
    args = parser.parse_args()

    if args.mode in ("compute", "all"):
        run_compute()

    if args.mode == "name":
        from mint.five.fig_atlas.naming import run_naming
        run_naming()

    if args.mode in ("plot", "all"):
        run_plot(panel=args.panel)


if __name__ == "__main__":
    main()
