import numpy as np
from state import State, UltimateTTT_Move


def alpha_beta_cutoff_search(state, d=4, cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning."""

    player = state.player_to_move

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -np.inf
        for a in state.get_valid_moves:
            if len(state.get_valid_moves) != 0:
                temp_state = State(state)
                temp_state.act_move(a)
                v = max(v, min_value(temp_state, alpha, beta, depth + 1))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            else: 
                break
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = np.inf
        for a in state.get_valid_moves:
            if len(state.get_valid_moves) != 0:
                temp_state = State(state)
                temp_state.act_move(a)
                v = min(v, max_value(temp_state, alpha, beta, depth + 1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            else: 
                break
        return v

    # Body of alpha_beta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    
    cutoff_test = (cutoff_test or (lambda temp_state, depth: depth > d or temp_state.get_valid_moves))
    best_score = -np.inf
    beta = np.inf
    best_action = None
    for a in state.get_valid_moves:
        if len(state.get_valid_moves) != 0:
            temp_state = State(state)
            temp_state.act_move(a)
            if player == -1:
                v = min_value(temp_state, best_score, beta, 1)
            else:
                v = max_value(temp_state, best_score, beta, 1)
            if v > best_score:
                best_score = v
                best_action = a
        else: 
            break
    return best_action


def eval_func(state: State):
    player = state.player_to_move
    if state.game_over == 1: 
        return 1000
    elif state.game_over == -1: 
        return -1000

    # calculate each block score
    block_score = []
    for i in range(9):
        if state.game_result(state.blocks[i]) == 1:
            block_score.append(24)
        elif state.game_result(state.blocks[i]) == -1:
            block_score.append(-24)
        elif state.game_result(state.blocks[i]) == 0:
            block_score.append(0)
        else:
            player_cell = 0
            opponent_cell = 0
            for j in range(9):
                if state.blocks[i].reshape(9)[j] == 1:
                    if j == 4: player_cell += 4
                    elif j % 2 == 0: player_cell += 3
                    else: player_cell += 2
                if state.blocks[i].reshape(9)[j] == -1:
                    if j == 4: opponent_cell += 4
                    elif j % 2 == 0: opponent_cell += 3
                    else: opponent_cell += 2

            block_score.append(player_cell - opponent_cell)

    # calculate global score
    result = 0
    for i in range(9):
        if i == 4:
            result += block_score[i] * 4
        elif i % 2 == 0:
            result += block_score[i] * 3
        else:
            result += block_score[i] * 2
    
    if player == -1:
        return result
    else:
        return - result


def select_move(cur_state, remain_time):
    if remain_time > 5:
        return alpha_beta_cutoff_search(cur_state, 10, None, eval_func)
    else:
        valid_moves = cur_state.get_valid_moves
        if len(valid_moves) != 0:
            return np.random.choice(valid_moves)
    return None


