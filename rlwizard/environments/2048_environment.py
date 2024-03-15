import numpy as np

# from rlwizard.environments import BaseEnvironment


class Board:
    def __init__(
        self,
        board_size: int = 4,
        tile_chance: float = 0.2,
    ) -> None:
        self.board_size = board_size
        self.tile_chance = tile_chance

        # Initialize board
        self.board = np.zeros((board_size, board_size), dtype=int)
        self._fill_tiles(n_tiles=2)

    def __repr__(self) -> str:
        return "Board():\n" + f"{self.board_size=}\n" + f"{self.board=}"

    def _fill_tiles(
        self,
        n_tiles: int = 1,
    ) -> None:
        # Find all empty tiles
        empty_tiles = np.argwhere(self.board == 0)

        # Randomly select n positions
        indices = np.random.choice(empty_tiles.shape[0], size=n_tiles, replace=False)
        selected_tiles = empty_tiles[indices]

        for tile in selected_tiles:
            rnd = np.random.random()
            self.board[tuple(tile)] = 2 if rnd > self.tile_chance else 4


if __name__ == "__main__":
    board = Board()
    print(board.board)
