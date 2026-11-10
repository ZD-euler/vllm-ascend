#!/usr/bin/env python3
"""Create a model loader view using reference metadata and local weight symlinks.

Never reads weight contents. The base-model index/tokenizer identities must be
verified separately against the supplied reference metadata inventory.
"""
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    weights = args.weights.resolve(strict=True)
    reference = args.reference.resolve(strict=True)
    for name in ('config.json', 'quant_model_description.json'):
        if not (reference / name).is_file():
            raise ValueError('missing reference metadata: ' + name)
    args.output.mkdir(parents=True, exist_ok=False)
    for path in weights.iterdir():
        if path.name not in ('config.json', 'quant_model_description.json'):
            (args.output / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    for name in ('config.json', 'quant_model_description.json'):
        shutil.copyfile(reference / name, args.output / name)
    print('Created local loader view; no weight contents read or copied.')


if __name__ == '__main__':
    main()
