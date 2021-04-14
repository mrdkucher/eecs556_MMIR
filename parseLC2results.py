
import argparse
import csv
import numpy as np
import os


def main(args):
    # Parse LC2 results from regLC2: create a table for all results with Cases as rows, and variants as columns
    lc2_dir = 'lc2_paired_mrus_brain'
    results_dirs = [case_dir for case_dir in os.listdir(lc2_dir) if 'Case' in case_dir and 'logs_reg' in case_dir]

    # Find all prefixes and suffixes to determine table size
    prefixes = []
    suffixes = []
    for case_dir in results_dirs:
        prefix, suffix = case_dir.split('logs_reg')
        if prefix not in prefixes:
            prefixes.append(prefix)
        if suffix not in suffixes:
            suffixes.append(suffix)

    rows = len(prefixes)
    cols = len(suffixes)
    mTREs = np.zeros((rows, 2 * cols))  # enable space for rigid and affine
    runtimes = np.zeros((rows, cols))

    for p, prefix in enumerate(prefixes):
        for s, suffix in enumerate(suffixes):
            case_dir = os.path.join(lc2_dir, prefix + 'logs_reg' + suffix)
            if os.path.exists(case_dir):
                with open(os.path.join(case_dir, "reg_results.txt"), 'r') as f:
                    lines = f.readlines()
                    line = 0
                    for line in lines:
                        if "seconds" in line:
                            runtimes[p, s] = float(line.split(' ')[-2])
                        if "landmark mTRE (rigid)" in line:
                            mTREs[p, 2 * s] = float(line.strip('\n').split(' ')[-1])
                        if "landmark mTRE (affine)" in line or "landmark mTRE:" in line:
                            mTREs[p, 2 * s + 1] = float(line.strip('\n').split(' ')[-1])

    # Save info as CSVs:
    with open(args.output + '_mTRE.csv', 'w') as f:
        writer = csv.writer(f)
        pretty_prefixes = [p.strip('_') for p in prefixes]
        pretty_suffixes = [s.strip('_') for s in suffixes]
        pretty_suffixes_double = []
        for ps in pretty_suffixes:
            pretty_suffixes_double.append(ps + '_rigid')
            pretty_suffixes_double.append(ps + '_affine')

        # Write headings
        writer.writerow(['Case #', *pretty_suffixes_double])
        for p, prefix in enumerate(pretty_prefixes):
            writer.writerow([prefix, *mTREs[p, :]])
    with open(args.output + '_runtimes.csv', 'w') as f:
        writer = csv.writer(f)
        pretty_prefixes = [p.strip('_') for p in prefixes]
        pretty_suffixes = sorted([s.strip('_') for s in suffixes])

        # Write headings
        writer.writerow(['Case #', *pretty_suffixes])
        for p, prefix in enumerate(pretty_prefixes):
            writer.writerow([prefix, *runtimes[p, :]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--output',
        help='output csv file name',
        dest='output',
        default='aggregate_results'
    )
    args = parser.parse_args()
    main(args)
