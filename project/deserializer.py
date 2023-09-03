import math
import sys

from tqdm import tqdm

filename = None # name of binary output file to parse
outfile = None  # name of output text file to write parsed output to
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

def print_pattern(f, bytestr, n_rows, n_cols):
    charmap = {0: "  ", 1: "{}"}
    for i in range(n_rows):
        bit_stream = stream_bits(bytestr)[i*n_cols:(i+1)*n_cols]
        outstr = "|" + "".join(charmap[c] for c in bit_stream) + "|"
        if f is not None:
            f.write(outstr + "\n")
        else:
            print(outstr)

def parse_cmdline(argv):
    global filename, outfile, n_rows, n_cols
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
        if argv[i] in ("-o", "--output"):
            if argc > i + 1:
                outfile = argv[i + 1]
                i += 2
            else:
                print(help)
                return 1
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
        raw_bytes = f.read()

    n_bytes = math.ceil(n_rows * n_cols / 8.0)
    n_lines = len(raw_bytes) / n_bytes
    if (int(n_lines) != n_lines):
        raise IOError(f"grid size of {n_rows}x{n_cols} does not evenly divide file size of {n_bytes} bytes")
    n_lines = int(n_lines)

    if outfile is not None:
        print(f"Parsing {n_lines} grid states of size {n_rows}x{n_cols} to {outfile}")
        with open(outfile, "w") as f:
            f.write("-" * (2 * n_cols + 2) + "\n")
            for i in tqdm(range(int(n_lines))):
                print_pattern(f, raw_bytes[i * n_bytes:(i + 1) * n_bytes], n_rows, n_cols)
                f.write("-" * (2 * n_cols + 2) + "\n")
            f.write("(END)")
    else:
        print(f"Parsing {n_lines} grid states of size {n_rows}x{n_cols} to stdout")
        print("-" * (2 * n_cols + 2))
        for i in tqdm(range(int(n_lines))):
            print_pattern(None, raw_bytes[i * n_bytes:(i + 1) * n_bytes], n_rows, n_cols)
            print("-" * (2 * n_cols + 2))
        f.write("(END)")


if __name__ == "__main__":
    main(sys.argv)
