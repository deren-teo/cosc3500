
def print_pattern(bytestr, n_rows, n_cols):
    charmap = {0: " ", 1: "#"}
    for i in range(n_rows):
        print("|", "|".join([charmap[c] for c in bytestr[i*n_cols:(i+1)*n_cols]]), "|", sep="")

def main(argv):
    if len(argv) != 4:
        print("Usage:", argv[0], "N_ROWS N_COLS FILE")
        return 1

    n_rows, n_cols, filename = argv[1:]
    n_rows, n_cols = map(int, [n_rows, n_cols])

    with open(filename, "rb") as f:
        for line in f:
            if len(line.strip()) != n_rows * n_cols:
                raise IOError(f"line length ({len(line.strip())}) does not match input dims ({n_rows}, {n_cols})")
            print_pattern(line, n_rows, n_cols)
            print("-" * (2 * n_cols + 1))
        print("(END)")


if __name__ == "__main__":
    import sys
    main(sys.argv)
