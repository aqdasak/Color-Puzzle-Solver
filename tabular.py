from typing import Iterable


def flatten(items):
    """Yield items from any nested iterable

    simple = [[1, 2, 3], [4, 5, 6], [7], [8, 9]]
    list(flatten(simple))
    # [1, 2, 3, 4, 5, 6, 7, 8, 9]

    complicated = [1, 2, [3, [4, [5, 6, {7, 8}], "steie"], {"qa": 12}]]              # numbers, strs, nested & mixed
    list(flatten(complicated))
    # [1, 2, 3, 4, 5, 6, 8, 7, 'steie', 'qa']
    """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def tabulate(string: str, delimiter=' ', *, comment_char='#') -> list[list[str]]:
    def split(line: str) -> list[str]:
        if delimiter in line:
            row = line.split(delimiter)

            # consecutive spaces leads to '' (empty string) in list
            while '' in row:
                row.remove('')
            return row

        return list(line)

    string = string.strip()

    # Creating 2D table
    table = string.split('\n')
    table = [i.strip() for i in table]
    table = [split(i) for i in table if len(i) > 0 and i[0] != comment_char]

    return table
