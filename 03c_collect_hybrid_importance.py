#!/usr/bin/env python3
"""Same certified Stage03 artifact contract, hybrid Taylor + empirical gate Fisher."""
from pathlib import Path
import importlib.util
import sys

src = Path(__file__).with_name('03b_collect_group_fisher.py')
spec = importlib.util.spec_from_file_location('pruning_fisher_for_hybrid',src)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
if __name__ == '__main__':
    parser = mod.make_parser()
    args = parser.parse_args()
    args.importance = 'hybrid'  # force hybrid, never accidentally Fisher-only
    sys.exit(mod.run(args))
