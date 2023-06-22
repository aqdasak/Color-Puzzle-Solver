import numpy as np
from collections import Counter
from colored_water import ColoredWater
from tabular import flatten, tabulate


class Printer:
    def __init__(self, int_to_color_map: dict[int, str]) -> None:
        self.map = int_to_color_map
        self.max_color_len = len(max(int_to_color_map.values(), key=len))

    def print(self, step: int, color_val: int, from_: int, to: int) -> None:
        color = self.map[color_val]
        l = self.max_color_len - len(color)
        print(f'{step}:\t{color}:{" "*l}  {from_+1} -> {to+1}')


def print_info():
    print('All lines should have equal number of colors.')
    print(
        'Either a line can have only four characters or four group of characters separated by single space. Put 0 (zero) for empty place'
    )
    print('\nExamples:')
    print('✅ rgby')
    print('✅ r g b y')
    print('✅ red g b yel')
    print('✅ rgb0')
    print('✅ 0000')
    print('❌ red gby')
    print('❌ r  g b y')
    print('    ^^ 2 spaces')


def calculate_move(cur: np.ndarray, next: np.ndarray) -> tuple[int, int, int]:
    diff = next - cur
    from_row_col = np.where(diff < 0)
    to_row_col = np.where(diff > 0)
    val = cur[from_row_col][0]
    return val, (from_row_col[1][0], to_row_col[1][0])


def get_int_map(m: list) -> dict[str, int]:
    """
    '0' (str) should be mapped to 0 (int)
    """
    unique_chars = set(m)

    int_map: dict[str, int] = {}
    for i, char in enumerate(unique_chars, start=1):
        int_map[char] = i
    int_map['0'] = 0

    return int_map


def is_all_colors_freq_equal(m: list) -> None:
    """
    Ignores 0
    """
    m = m[:]
    while '0' in m:
        m.remove('0')
    counts = Counter(m)
    del counts['0']

    a={v:k for k,v in Counter(counts.values()).items()}
    most_probable_size=a[max(a)]

    flag = True
    for k, v in counts.items():
        if v != most_probable_size:
            flag = False
            print(f'{k} is present {v} times')
    return flag


def main():
    filename = input('Enter filename: ')
    with open(filename) as f:
        inp = f.read()
    matrix = tabulate(inp)
    flat_matrix = list(flatten(matrix))

    if not '0' in flat_matrix:
        print('No empty space available. Add empty tubes by entring 0')
        exit(1)

    if not is_all_colors_freq_equal(flat_matrix):
        print('Error: Inconsistant colors frequencies')
        exit(1)

    int_map = get_int_map(flat_matrix)
    matrix = [[int_map[i] for i in j] for j in matrix]

    try:
        matrix = np.array(matrix).T
    except ValueError:
        print('\nError: Inconsistant length in input\n')
        print_info()
        exit(1)

    solution = ColoredWater(matrix).solve(depthFirst=True)

    # Determining the move
    move = []
    for cur, next in zip(solution[:-1], solution[1:]):
        cur = cur.pos
        next = next.pos
        move.append(calculate_move(cur, next))

    # Removing the repeated same move as they are done in one go in the game
    prev = move[0]
    move_z = [move[0]]
    for cur in move[1:]:
        if cur != prev:
            move_z.append(cur)
        prev = cur

    rev_int_map = {v: k for k, v in int_map.items()}
    p = Printer(rev_int_map)
    print('\nSolution:')
    for i, (val, (from_, to)) in enumerate(move_z, start=1):
        p.print(i, val, from_, to)


if __name__ == '__main__':
    main()
