#!/usr/bin/env python3
"""Prepare matched raw/SR single-frame CenterPoint info files.

The source ``frame200_ori/infos_test_200.pkl`` contains the authoritative
annotations and radar metadata.  This script creates two *identical* train/
validation splits and changes only ``radar_path`` for the SR branch.  PCDs are
referenced in place; no point-cloud data is copied into the repository.
"""

from __future__ import annotations

import argparse
import copy
import pickle
from pathlib import Path


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Read PKLs written by NumPy 2 while running the project NumPy 1 ABI."""

    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = 'numpy.core' + module[len('numpy._core'):]
        return super().find_class(module, name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-info', type=Path, required=True)
    parser.add_argument('--sr-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sensor', default='RADAR_FRONT')
    parser.add_argument('--val-stride', type=int, default=5)
    parser.add_argument('--allow-missing-sr', action='store_true')
    return parser.parse_args()


def _sr_path(raw_path: str, source_root: Path, sr_root: Path) -> Path:
    raw = Path(raw_path)
    # Info paths are normally relative to frame200_ori.  Absolute paths are
    # also handled so this remains useful with regenerated PKLs.
    if raw.is_absolute():
        try:
            relative = raw.relative_to(source_root)
        except ValueError:
            relative = Path(*raw.parts[raw.parts.index('obs02'):]) if 'obs02' in raw.parts else Path(raw.name)
    else:
        relative = raw
    return (sr_root / relative).with_name(relative.stem + '_SR.pcd')


def main():
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    source_info = args.source_info.expanduser().resolve()
    sr_root = args.sr_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.val_stride < 2:
        raise ValueError('--val-stride must be >= 2')

    with source_info.open('rb') as stream:
        infos = _NumpyCompatUnpickler(stream).load()
    if not isinstance(infos, list) or not infos:
        raise ValueError('Expected a non-empty list in %s' % source_info)

    raw_infos = []
    sr_infos = []
    missing = []
    for index, original in enumerate(infos):
        info = copy.deepcopy(original)
        sensor_info = info.setdefault('radars', {}).get(args.sensor)
        if not sensor_info or not sensor_info.get('radar_path'):
            raise ValueError('Info %d has no %s radar_path' % (index, args.sensor))
        raw_path = sensor_info['radar_path']
        sr_path = _sr_path(raw_path, source_root, sr_root)
        if not sr_path.exists():
            missing.append((index, str(sr_path)))
        sr_info = copy.deepcopy(info)
        sr_info['radars'][args.sensor]['radar_path'] = str(sr_path)
        sr_info['point_cloud_variant'] = 'reconstructed_sr'
        raw_info = copy.deepcopy(info)
        raw_info['point_cloud_variant'] = 'raw'
        raw_infos.append(raw_info)
        sr_infos.append(sr_info)

    if missing and not args.allow_missing_sr:
        sample = '\n'.join('  %d %s' % item for item in missing[:10])
        raise FileNotFoundError(
            '%d SR PCDs are missing. First entries:\n%s' % (len(missing), sample)
        )
    if missing:
        keep = {i for i, _ in missing}
        raw_infos = [x for i, x in enumerate(raw_infos) if i not in keep]
        sr_infos = [x for i, x in enumerate(sr_infos) if i not in keep]

    # Every fifth frame is validation.  The index list is shared verbatim by
    # raw and SR branches, so data input is the only experimental variable.
    val_indices = set(range(0, len(raw_infos), args.val_stride))
    raw_train = [x for i, x in enumerate(raw_infos) if i not in val_indices]
    raw_val = [x for i, x in enumerate(raw_infos) if i in val_indices]
    sr_train = [x for i, x in enumerate(sr_infos) if i not in val_indices]
    sr_val = [x for i, x in enumerate(sr_infos) if i in val_indices]

    outputs = {
        'raw_train.pkl': raw_train,
        'raw_val.pkl': raw_val,
        'sr_train.pkl': sr_train,
        'sr_val.pkl': sr_val,
    }
    for name, payload in outputs.items():
        with (output_dir / name).open('wb') as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        'source_info': str(source_info),
        'source_root': str(source_root),
        'sr_root': str(sr_root),
        'sensor': args.sensor,
        'val_stride': args.val_stride,
        'num_samples': len(raw_infos),
        'num_train': len(raw_train),
        'num_val': len(raw_val),
        'missing_sr': len(missing),
        'outputs': {key: len(value) for key, value in outputs.items()},
    }
    with (output_dir / 'manifest.pkl').open('wb') as stream:
        pickle.dump(manifest, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(manifest)


if __name__ == '__main__':
    main()
