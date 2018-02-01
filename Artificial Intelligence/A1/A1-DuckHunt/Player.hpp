#ifndef _DUCKS_PLAYER_HPP_
#define _DUCKS_PLAYER_HPP_

#include "Deadline.hpp"
#include "GameState.hpp"
#include "Action.hpp"
#include "Bird.hpp"
#include "Constants.hpp"
#include "HMM.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <math.h>

namespace ducks
{

class Player
{
public:
    /**
     * Constructor
     * There is no data in the beginning, so not much should be done here.
     */
    Player();

    /**
     * Shoot!
     *
     * This is the function where you start your work.
     *
     * You will receive a variable pState, which contains information about all
     * birds, both dead and alive. Each birds contains all past actions.
     *
     * The state also contains the scores for all players and the number of
     * time steps elapsed since the last time this function was called.
     *
     * @param pState the GameState object with observations etc
     * @param pDue time before which we must have returned
     * @return the prediction of a bird we want to shoot at, or cDontShoot to pass
     */
    Action shoot(const GameState &pState, const Deadline &pDue);

    /**
     * Guess the species!
     * This function will be called at the end of each round, to give you
     * a chance to identify the species of the birds for extra points.
     *
     * Fill the vector with guesses for the all birds.
     * Use SPECIES_UNKNOWN to avoid guessing.
     *
     * @param pState the GameState object with observations etc
     * @param pDue time before which we must have returned
     * @return a vector with guesses for all the birds
     */
    std::vector<ESpecies> guess(const GameState &pState, const Deadline &pDue);

    /**
     * If you hit the bird you were trying to shoot, you will be notified
     * through this function.
     *
     * @param pState the GameState object with observations etc
     * @param pBird the bird you hit
     * @param pDue time before which we must have returned
     */
    void hit(const GameState &pState, int pBird, const Deadline &pDue);

    /**
     * If you made any guesses, you will find out the true species of those
     * birds through this function.
     *
     * @param pState the GameState object with observations etc
     * @param pSpecies the vector with species
     * @param pDue time before which we must have returned
     */
    void reveal(const GameState &pState, const std::vector<ESpecies> &pSpecies, const Deadline &pDue);
  private:
    std::vector<HMM> HMMs; //Vector for keeping the HMMs for the current round of birds before they are sorted according to species
    std::vector<bool> birdSpeciesFound;
    std::vector<std::vector<HMM> > models; //For keeping all trained HMMs, indexed by the different species
    std::vector<HMM> speciesModels; //For keeping one combined model for each species
    double shootThresh = 0.69; //The shooting threshhold for probability for a species given an obsSeq
    bool storkFound;


    void combineHMMs();
};

} /*namespace ducks*/

#endif
