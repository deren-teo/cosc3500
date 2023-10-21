import argparse

def print_pattern(bytestr, n_rows, n_cols):
    charmap = {0: "  ", 1: "{}"}
    for i in range(n_rows):
        print("|" + "".join(charmap[c & 0x01] for c in bytestr[i*n_cols:(i+1)*n_cols]) + "|")

def main(args):
    # Read binary Game of Life output
    with open(args.filename, "rb") as f:
        raw_bytes = f.read()

    n_rows = args.gridrows
    n_cols = args.gridcols

    # Calculate number of generations represented in the binary
    n_bytes = len(raw_bytes)
    n_cells = n_rows * n_cols
    n_grids = n_bytes / n_cells
    if int(n_grids) != n_grids:
        raise IOError(f"grid size of {n_rows}x{n_cols} does not evenly divide file size of {n_bytes} bytes")
    n_grids = int(n_grids)

    # Parse the binary and print in graphic format
    byte_idx = 0
    print("-" * (2 * n_cols + 2))
    for _ in range(n_grids):
        next_idx = byte_idx + n_cells
        print_pattern(raw_bytes[byte_idx:next_idx], n_rows, n_cols)
        print("-" * (2 * n_cols + 2))
        byte_idx = next_idx
    print("(END)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Game of Life binary output file")
    parser.add_argument("gridrows", type=int, help="Simulated number of grid rows")
    parser.add_argument("gridcols", type=int, help="Simulated number of grid columns")
    args = parser.parse_args()
    main(args)
