import math
import sys

filename = None # name of binary output file to parse
n_rows = None   # number of rows represented in binary output file to parse
n_cols = None   # number of columns represented in binary output file to parse

def stream_bits(bytestr):
    bitstream = list()
    for byte in bytestr:
        bitstream.append(bool(byte & (1 << 0)))
        bitstream.append(bool(byte & (1 << 1)))
        bitstream.append(bool(byte & (1 << 2)))
        bitstream.append(bool(byte & (1 << 3)))
        bitstream.append(bool(byte & (1 << 4)))
        bitstream.append(bool(byte & (1 << 5)))
        bitstream.append(bool(byte & (1 << 6)))
        bitstream.append(bool(byte & (1 << 7)))
    return bitstream

def print_pattern(bytestr, n_rows, n_cols):
    charmap = {0: " ", 1: "#"}
    for i in range(n_rows):
        print("|", "|".join([charmap[c] for c in stream_bits(bytestr)[i*n_cols:(i+1)*n_cols]]), "|", sep="")

def parse_cmdline(argv):
    global filename, n_rows, n_cols
    argc = len(argv)
    help = \
        f"Usage: {argv[0]} FILEPATH SIZE\n\n" + \
        f"  -f, --file FILEPATH path to a Game of Life output file\n" + \
        f"  -s, --size NxM      number of rows x columns in the grid\n\n"
    i = 0
    while i < argc:
        if argv[i] in ("-f", "--file"):
            if argc > i + 1:
                filename = argv[i + 1]
                i += 2
            else:
                print(help)
                return 1
            continue
        if argv[i] in ("-s", "--size"):
            if argc > i + 1 and "x" in argv[i + 1]:
                n_rows, n_cols = list(map(int, argv[i + 1].split("x")))
                i += 2
            else:
                print(help)
                return 1
            continue
        else:
            i += 1
    if not all ([filename, n_rows, n_cols]):
        print(help)
        return 1
    return 0

def main(argv):
    parse_cmdline(argv)
    with open(filename, "rb") as f:
        for line in f:
            if len(line.strip()) != math.ceil(n_rows * n_cols / 8.0):
                raise IOError(f"line length ({len(line.strip())}) does not match input dims ({n_rows}, {n_cols})")
            print_pattern(line, n_rows, n_cols)
            print("-" * (2 * n_cols + 1))
        print("(END)")


if __name__ == "__main__":
    main(sys.argv)
