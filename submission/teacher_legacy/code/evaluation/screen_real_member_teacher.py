"""Fixed teacher point gate; no native-UQ score or deployment claim."""
import argparse
from pathlib import Path

from screen_shared_point_readout import main


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('root','manifest','real-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args(),member_teacher=True)
