# Color Puzzle Solver

This program solves the color puzzle, which have test tubes containing colors of different colors. The goal of the game is to have single color in each test tube. You can only pour color on top of the same color but not on top of different color. While pouring top color moves completely whether it has more than one unit, if the space is available in the tube in which the color is poured. Some empty test tubes will be given to solve the puzzle

# Install from source
Poetry is required. For installation click [here](https://python-poetry.org/docs/#installation).

Download the source and install the dependencies by running:

``` 
git clone https://github.com/aqdasak/Color-Puzzle-Solver.git
cd Color-Puzzle-Solver
poetry install
```


# Usage

In the source folder containing pyproject.toml, run
```
poetry shell
```

Write your current color state as displayed in the game, in a file and then execute:
```
python .
```


# Attribution

Module `colored_water.py` is obtained from [stackoverflow](https://stackoverflow.com/a/69210205)
