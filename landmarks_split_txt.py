import numpy as np
import argparse


def main(d_options):
    """
    Split a .tag file into landmarks
    Adapted from: https://gist.github.com/mattiaspaul/56a49fa792ef6f143e56699a06067712

    :param d_options: input options (dict)
        key: 'inputtag' - tag file for reading landmarks
        key: 'savetxt' - txt file prefix for saving landmarks
    """

    landmarks = []
    with open(d_options['inputtag']) as f:
        lines = f.readlines()
        for line in lines[5:]:  # skip first 5 lines
            line = line.strip("\n").strip(";").strip("\"").strip(" ")
            landmarks.append(np.fromstring(line, dtype=float, sep=' '))

    landmarks = np.asarray(landmarks)
    # print(landmarks)

    savetxt = d_options.get('savetxt', None)

    if savetxt is not None:
        with open(d_options['savetxt'] + "_mri.txt", "w") as text_file:
            for i in range(landmarks.shape[0]):
                text_file.write("%f %f %f %d \n" % (
                    landmarks[i, 0], landmarks[i, 1], landmarks[i, 2], i + 1))
        with open(d_options['savetxt'] + "_us.txt", "w") as text_file:
            for i in range(landmarks.shape[0]):
                text_file.write("%f %f %f %d \n" % (
                    landmarks[i, 3], landmarks[i, 4], landmarks[i, 5], i + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputtag", dest="inputtag",
                        help="input tag file from (MINC)", default=None, required=True)
    parser.add_argument("--savetxt", dest="savetxt",
                        help="output landmark file to (txt)", default=None, required=True)
    options = parser.parse_args()
    d_options = vars(options)
    main(d_options)
