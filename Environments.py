import chess
import numpy as np

PIECE_MAP = {
    'k': 0,
    'q': 1,
    'r': 2,
    'n': 3,
    'b': 4,
    'p': 5,
}


class Chess:
    def __init__(self, player_1, player_2, max_moves=50):
        self.board = chess.Board()
        self.pieces_map = PIECE_MAP
        self.max_moves = max_moves
        self.current_move = 0
        self.player_1 = player_1
        self.player_2 = player_2

    def step(self, move, player):
        self.board.push_san(move)
        reward, done = self.game_over(player)
        return self.encode_board(), reward, done

    def reset(self, player_1, player_2):
        self.player_1 = player_1
        self.player_2 = player_2
        self.board = chess.Board()
        self.current_move = 0
        return self.encode_board()

    def get_legal_moves(self):
        return [self.board.uci(move) for move in self.board.legal_moves]

    def game_over(self, player):
        outcome = self.board.outcome(claim_draw=True)
        self.current_move += 1
        if outcome is None and self.current_move < self.max_moves:
            return 0, False
        elif outcome is None and self.current_move >= self.max_moves:
            return 0, True
        elif outcome.termination == chess.Termination.CHECKMATE and player == self.player_1:
            return 1, True
        elif outcome.termination == chess.Termination.CHECKMATE and player == self.player_2:
            return -1, True
        else:
            return 0, True

    def encode_board(self):
        pieces = self.board.epd().split(' ', 1)[0]
        rows = pieces.split('/')
        np_board = np.zeros((8, 8, 12))
        for i, row, in enumerate(rows):
            j = 0
            for item in row:
                if item.isdigit():
                    j += int(item)
                else:
                    piece = self.pieces_map[item.lower()]
                    if item.islower():
                        piece += 6
                    np_board[i][j][piece] = 1
                    j += 1
        return np_board


if __name__ == '__main__':
    pass
