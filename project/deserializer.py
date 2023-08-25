import math

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

def main(argv):
    if len(argv) != 4:
        print("Usage:", argv[0], "N_ROWS N_COLS FILE")
        return 1

    n_rows, n_cols, filename = argv[1:]
    n_rows, n_cols = map(int, [n_rows, n_cols])

    with open(filename, "rb") as f:
        for line in f:
            if len(line.strip()) != math.ceil(n_rows * n_cols / 8.0):
                raise IOError(f"line length ({len(line.strip())}) does not match input dims ({n_rows}, {n_cols})")
            print_pattern(line, n_rows, n_cols)
            print("-" * (2 * n_cols + 1))
        print("(END)")


if __name__ == "__main__":
    import sys
    main(sys.argv)
