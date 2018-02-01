#include "player.hpp"
#include <cmath>
#include <cstdlib>
#include <limits>
#include <initializer_list>

namespace TICTACTOE3D
{

Player::Player() {
  otherPlayer[CELL_X] = CELL_O;
  otherPlayer[CELL_O] = CELL_X;

  }

GameState Player::play(const GameState &pState,const Deadline &pDue)
{
    //std::cerr << "Processing " << pState.toMessage() << std::endl;
    maxPlayer = pState.getNextPlayer();
    opponent = otherPlayer[maxPlayer];
    std::vector<GameState> lNextStates;
    pState.findPossibleMoves(lNextStates);


    if (lNextStates.size() == 0) return GameState(pState, Move());
    if (lNextStates.size() == 1) return lNextStates[0];

    int move;
    int best = std::numeric_limits<int>::min();
    for (int idx = 0; idx<lNextStates.size(); ++idx) {
      int v = alphaBeta(lNextStates[idx], maxPlayer, std::numeric_limits<int>::min(), std::numeric_limits<int>::max(), 2);
      if (v > best) {
        move = idx;
        best = v;
      }
    }
    /*
     * Here you should write your clever algorithms to get the best next move, ie the best
     * next state. This skeleton returns a random move instead.
     */

    return lNextStates[move];
}


int Player::alphaBeta(const GameState &pState, int player, int alpha, int beta, int depth) {

  if (depth == 0 || pState.isEOG()) {
    return heuristic(pState);
  }
  else {
    std::vector<GameState> children;
    pState.findPossibleMoves(children);
    if (player == maxPlayer){
      int v = std::numeric_limits<int>::min();
      for (int i=0; i<children.size(); ++i) {
        v = fmax(v, alphaBeta(children[i], opponent, alpha, beta, depth-1));
        alpha = fmax(alpha, v);
        if (beta <= alpha) break;
      }
      return v;
    }
    else { //player != maxPlayer
      int v = std::numeric_limits<int>::max();
      for (int i=0; i<children.size(); ++i) {
        v = fmin(v, alphaBeta(children[i], maxPlayer, alpha, beta, depth-1));
        beta = fmin(beta, v);
        if (beta <= alpha) break;
      }
      return v;
    }
  }
}

int Player::heuristic(const GameState &pState) {
  int score = 0;

  if (pState.isEOG()) {
    if ((pState.isXWin() && maxPlayer == CELL_X)|| (pState.isOWin() && maxPlayer == CELL_O)) {
      return 1000;
    }
    else if ((pState.isXWin() && maxPlayer == CELL_O)|| (pState.isOWin() && maxPlayer == CELL_X)) {
      return -1000;
    }
    else {
      return 0;
    }
  }


  for (size_t winPathIdx = 0; winPathIdx < 76; winPathIdx++) {
    int winPathScore = 0;
    int opponentWinPathScore = 0;
    for (size_t posIdx = 0; posIdx < 4; posIdx++) {
      int pos = possibleWinPaths[winPathIdx][posIdx];
      uint8_t contentsOfCell = pState.at(pos);
      if (contentsOfCell == maxPlayer) {
        winPathScore++;
      } else if (contentsOfCell == opponent) {
        opponentWinPathScore++;
      }
    }
    if ((winPathScore > 0 && opponentWinPathScore > 0) || (winPathScore == 0 && opponentWinPathScore == 0)){
      score += 0;
    }
    else if (winPathScore > 0 && opponentWinPathScore == 0){
      score += std::pow(10, winPathScore-1);
    }
    else if (opponentWinPathScore > 0 && winPathScore == 0) {
      score -= std::pow(10, opponentWinPathScore-1);
    }
  }
  return score;

}

/*namespace TICTACTOE3D*/ }
