"""Use the identical held15 point scorer with the sealed equal-Re intervention."""
import argparse
from pathlib import Path

from screen_random_real12000 import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'control-dir', 'manifest', 'real-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args(), rebalanced=True)
