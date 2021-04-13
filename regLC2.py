import argparse
from types import SimpleNamespace
import os

import sys
sys.path.append("lc2_paired_mrus_brain")
from register import main as reg  # noqa: E402


def main(args, cases):
    # Run LC2 registration on entire dataset, or a specific pair
    train_dir = "RESECT/preprocessed/train"
    val_dir = "RESECT/preprocessed/valid"
    test_dir = "RESECT/preprocessed/test"

    mTREs = []
    for case in cases:
        # Create an object to store register options
        options = SimpleNamespace()
        case_dir = ""
        case_str = "Case" + str(case)
        case_filename = case_str + ".nii.gz"

        # find directory for case
        if any([case_filename == fname for fname in os.listdir(os.path.join(train_dir, "moving_images"))]):
            case_dir = train_dir
        elif any([case_filename == fname for fname in os.listdir(os.path.join(val_dir, "moving_images"))]):
            case_dir = val_dir
        else:
            case_dir = test_dir

        # Set registration options for lc2
        options.moving = os.path.join(case_dir, "moving_images", case_filename)
        options.fixed = os.path.join(case_dir, "fixed_images", case_filename)
        options.tag = os.path.join(case_dir, "landmarks", case_str + "-MRI-beforeUS.tag")
        # Use landmark mTRE for now
        options.moving_label = None  # os.path.join(case_dir, "moving_labels", case_filename)
        options.fixed_label = None  # os.path.join(case_dir, "fixed_labels", case_filename)
        options.image_size = args.image_size
        options.max_iter = args.max_iter
        options.v_bobyqa = True
        options.affine = args.affine
        options.seek_global_minimum = args.seek_global_minimum
        options.output = os.path.join("lc2_paired_mrus_brain", case_str + "_logs_reg")
        if args.postfix:
            options.output += '_' + args.postfix
        options.patch = args.patch
        options.patch_size = args.patch_size
        options.neighborhood = args.neighborhood

        # Perform image registration
        reg(options)

        # Save results to array:
        print(os.getcwd())
        with open(os.path.join(options.output, "reg_results.txt"), 'r') as f:
            lines = f.readlines()
            mTRE_found = False
            line = 0
            while not mTRE_found and line < len(lines):
                if "landmark mTRE" in lines[line]:
                    mTRE_found = True
                    mTREs.append(float(lines[line].strip('\n').split(' ')[-1]))
                line += 1

    print("Results:")
    for (c, m) in zip(cases, mTREs):
        print("Case {:d}: mTRE = {:f}".format(c, m))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--cases",
        help="Case number(s) to register from RESECT dataset (integer(s) from 1-27). -1 for all. e.g. -c 1 5 7",
        dest="cases",
        nargs="*",
        action="store",
        type=int,
        default=[-1]
    )
    parser.add_argument(
        "-s", "--image-size",
        help="3-entry tuple to resize image e.g. (256, 256, 288)",
        dest="image_size",
        nargs=3,
        action="store",
        type=int,
        default=[70, 65, 60]
    )
    parser.add_argument(
        '--no-patch',
        help='Use patch-based LC2',
        dest='patch',
        action='store_false',
    )
    parser.add_argument(
        '-p', '--patch-size',
        help='LC2 patch size. Default = 7',
        dest='patch_size',
        action='store',
        type=int,
        default=7
    )
    parser.add_argument(
        '-a', '--affine',
        help="Perform optional affine transformation",
        dest="affine",
        action="store_true",
        default=False
    )
    parser.add_argument(
        '-n', '--neighborhood',
        help='Use neighborhood intensity/gradient for LC2. Default = False',
        dest='neighborhood',
        action='store_true'
    )
    parser.add_argument(
        "--max-iter",
        help="number of iterations to run",
        dest="max_iter",
        action="store",
        type=int,
        default=10000
    )
    parser.add_argument(
        "-g", "--seek_global_minimum",
        help="enable seek global minimum option for bobyqa solver",
        dest="seek_global_minimum",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--postfix",
        help="Add a postfix to output directory",
        dest="postfix",
        action="store"
    )
    args = parser.parse_args()
    all_cases = [1, 2, 3, 4, 5, 6, 7, 8,
                 12, 13, 14, 15, 16, 17, 18, 19,
                 21, 23, 24, 25, 26, 27]
    cases = []
    if args.cases[0] == -1:
        print("Registering all")
        cases = all_cases
    else:
        for case in args.cases:
            if case in all_cases:
                cases.append(case)
            else:
                print("Case {:d} not found".format(case))
    print("Cases to register:", cases)
    main(args, cases)
