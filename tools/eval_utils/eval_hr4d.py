"""Evaluate an OpenPCDet result.pkl with the HR-4D protocol."""

import argparse
import json
import math
import pickle
from pathlib import Path

from pcdet.datasets.hr4d.hr4d_eval import evaluate_hr4d, format_evaluation_results


def parse_args():
    parser = argparse.ArgumentParser(description='HR-4D elliptical center-distance evaluation')
    parser.add_argument('--gt_infos', type=Path, required=True, help='Info pickle containing per-frame annos')
    parser.add_argument('--predictions', type=Path, required=True, help='OpenPCDet result.pkl')
    parser.add_argument('--classes', nargs='+', default=None, help='Evaluated classes; defaults to GT classes')
    parser.add_argument('--output_json', type=Path, default=None, help='Optional detailed JSON output')
    return parser.parse_args()


def load_pickle(path: Path):
    with path.open('rb') as file:
        return pickle.load(file)


def make_json_safe(value):
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    args = parse_args()
    gt_infos = load_pickle(args.gt_infos)
    pred_annos = load_pickle(args.predictions)
    gt_annos = [info['annos'] if 'annos' in info else info for info in gt_infos]

    details = evaluate_hr4d(gt_annos, pred_annos, class_names=args.classes)
    result_str, _ = format_evaluation_results(details)
    print(result_str)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open('w') as file:
            json.dump(make_json_safe(details), file, indent=2, allow_nan=False)
        print(f'Detailed metrics saved to {args.output_json}')


if __name__ == '__main__':
    main()
