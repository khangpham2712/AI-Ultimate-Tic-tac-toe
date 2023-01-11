import numpy as np
import copy
from math import inf

'''
    @author: Le Hong Phong - 1813518
             Nguyen Hoang Khoa - 1812658
             Nguyen The Khang - 1812558
'''


def select_move(cur_state, remain_time):
    valid_moves = cur_state.get_valid_moves
    player = cur_state.player_to_move
    if cur_state.previous_move == None:
        return valid_moves[40]
    if len(valid_moves) != 0:
        if len(valid_moves ) >9:
            user_score, user_state, user_move = minimax(cur_state, player, 2)
        else:
            user_score, user_state, user_move = minimax(cur_state, player, 4)
        return user_move
    return None


def successors(cur_state):
    valid_moves = cur_state.get_valid_moves
    succ = []
    moves_list = []
    for one_moves in valid_moves:
        one_succ = copy.deepcopy(cur_state)
        one_succ.act_move(one_moves)
        succ.append(one_succ)
        moves_list.append(one_moves)
    return zip(succ, moves_list)


def opponent(p):
    if p == 1:
        return -1
    else:
        return 1


def evaluate_small_box(board, player):
    row_sum = np.sum(board, 1)
    col_sum = np.sum(board, 0)
    diag_sum_topleft = board.trace()
    diag_sum_topright = board[::-1].trace()
    score = 0
    # player is X
    if (player == 1):
        if (any(row_sum == 3) or any(col_sum == 3) or (diag_sum_topleft == 3) or (diag_sum_topright == 3)):
            score += 100
        if (any(row_sum == 2) or any(col_sum == 2) or (diag_sum_topleft == 2) or (diag_sum_topright == 2)):
            score += 10
        if (any(row_sum == 1) or any(col_sum == 1) or (diag_sum_topleft == 1) or (diag_sum_topright == 1)):
            score += 1
        if (any(row_sum == -3) or any(col_sum == -3) or (diag_sum_topleft == -3) or (diag_sum_topright == -3)):
            score += -100
        if (any(row_sum == -2) or any(col_sum == -2) or (diag_sum_topleft == -2) or (diag_sum_topright == -2)):
            score += -10
        if (any(row_sum == -1) or any(col_sum == -1) or (diag_sum_topleft == -1) or (diag_sum_topright == -1)):
            score += -1
    # player is O
    else:
        if (any(row_sum == 3) or any(col_sum == 3) or (diag_sum_topleft == 3) or (diag_sum_topright == 3)):
            score += -100
        if (any(row_sum == 2) or any(col_sum == 2) or (diag_sum_topleft == 2) or (diag_sum_topright == 2)):
            score += -10
        if (any(row_sum == 1) or any(col_sum == 1) or (diag_sum_topleft == 1) or (diag_sum_topright == 1)):
            score += 1
        if (any(row_sum == -3) or any(col_sum == -3) or (diag_sum_topleft == -3) or (diag_sum_topright == -3)):
            score += 100
        if (any(row_sum == -2) or any(col_sum == -2) or (diag_sum_topleft == -2) or (diag_sum_topright == -2)):
            score += 10
        if (any(row_sum == -1) or any(col_sum == -1) or (diag_sum_topleft == -1) or (diag_sum_topright == -1)):
            score += 1
    return score


def evaluate(cur_state, player):
    score = 0
    # calulator global_cells
    score += evaluate_small_box(cur_state.global_cells.reshape(3,
                                3), player) * 200
    # calulator all boxes
    for b in range(9):
        score += evaluate_small_box(cur_state.blocks[b], player)
    return score


def minimax(cur_state, player, depth):
    succ = successors(cur_state)
    best_move = (-inf, None, None)
    for s in succ:
        val = min_turn(s[0], opponent(player), depth-1,
                       -inf, inf)

        if val > best_move[0]:
            best_move = (val, s[0], s[0].previous_move)
    return best_move


def min_turn(cur_state, player, depth, alpha, beta):
    if depth <= 0:
        return evaluate(cur_state, opponent(player))
    succ = successors(cur_state)
    for s in succ:
        val = max_turn(s[0], opponent(player), depth-1,
                       alpha, beta)
        if val < beta:
            beta = val
        if alpha >= beta:
            break
    return beta


def max_turn(cur_state, player, depth, alpha, beta):
    global box_won
    if depth <= 0:
        return evaluate(cur_state, player)
    succ = successors(cur_state)
    for s in succ:
        val = min_turn(s[0], opponent(player), depth-1,
                       alpha, beta)
        if alpha < val:
            alpha = val
        if alpha >= beta:
            break
    return alpha
