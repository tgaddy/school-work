#include "Player.hpp"
#include <utility>

namespace ducks
{

Player::Player()
{
  speciesModels = std::vector<HMM>(static_cast<int>(COUNT_SPECIES));
  birdSpeciesFound = std::vector<bool> (static_cast<int>(COUNT_SPECIES),false);
  storkFound = false;
}

Action Player::shoot(const GameState &pState, const Deadline &pDue)
{
    //return cDontShoot;
    //During the first timestep for each round we reinitialize the HMMs vector
    int timestep = pState.getBird(0).getSeqLength();
    //if (timestep == 1) {
    HMMs = std::vector<HMM>(pState.getNumBirds());
    for (size_t i = 0; i < pState.getNumBirds(); i++) {
      HMM temp = HMM();
      HMMs[i] = temp;
      //}
    }

    int startShootingAtTimestep = 99-pState.getNumBirds();//(pState.getNumPlayers() > 1 ? 25 : 99 - pState.getNumBirds());
    //80 - pState.getRound()*5 - (pState.getNumPlayers()-1)*20;
    int firstShootingRound = 1;
    if (pState.getRound() < firstShootingRound) {
      return cDontShoot;
    } else if (pState.getRound() >= firstShootingRound && timestep > startShootingAtTimestep) {
      std::vector<std::pair<int,double> > birdMovementProbs(pState.getNumBirds());
      int timePerTraining = pDue.remainingMs()/pState.getNumBirds();
      //Get most probable next movement for each bird
      for (size_t birdNum = 0; birdNum < pState.getNumBirds(); birdNum++) {

        HMMs[birdNum].trainModelOnObsSeq(pState.getBird(birdNum), Deadline(timePerTraining));
        if (pState.getBird(birdNum).isDead()) {
          birdMovementProbs[birdNum] = std::pair<int,double>(-1,0.0);
          continue;
        }

        int mostProbSpecies = -1;
        double maxProb = std::numeric_limits<double>::lowest();

        for (size_t i = 0; i < speciesModels.size(); i++) {
          if (!birdSpeciesFound[i]) continue;
          double probOfObsSeq = speciesModels[i].getProbOfObsSeq(pState.getBird(birdNum));
          if (probOfObsSeq > maxProb) {
            maxProb = probOfObsSeq;
            mostProbSpecies = i;
          }
        }

        if (mostProbSpecies == 5) {
          return cDontShoot;
        }

        std::pair<int,double> movementPair(-1,0.0);
        //Checking if there was any species that were probable enough
        //if (mostProbSpecies != -1) {
          movementPair = HMMs[birdNum].predictNextMovement(pState.getBird(birdNum), shootThresh);
        //}
        birdMovementProbs[birdNum] = movementPair;
      }

      //Determine which bird to shoot
      Action shootAction = cDontShoot;
      double maxProb = shootThresh;
      for (size_t i = 0; i < birdMovementProbs.size(); i++) {
        if (birdMovementProbs[i].first != -1 && birdMovementProbs[i].second > maxProb) {
          maxProb = birdMovementProbs[i].second;
          shootAction = Action(i, static_cast<EMovement>(birdMovementProbs[i].first));
        }
      }
      return shootAction;
    }

    // This line choose not to shoot
    return cDontShoot;
}

std::vector<ESpecies> Player::guess(const GameState &pState, const Deadline &pDue)
{
    /*
    shootThresh *= 0.95;
    if (shootThresh < 0.5)
      shootThresh = 0.5;
    */
    HMMs = std::vector<HMM>(pState.getNumBirds());
    for (size_t i = 0; i < pState.getNumBirds(); i++) {
      HMM temp = HMM();
      HMMs[i] = temp;
    }

    std::vector<ESpecies> lGuesses(pState.getNumBirds(), SPECIES_UNKNOWN);
    int timePerTraining = pDue.remainingMs()/pState.getNumBirds();

    for (int i = 0; i < pState.getNumBirds(); ++i) {
      HMMs[i].trainModelOnObsSeq(pState.getBird(i),Deadline(timePerTraining));
    }

    if (pState.getRound() == 0) {
        for (size_t i = 0; i < lGuesses.size(); i++) {
          lGuesses[i] = SPECIES_PIGEON;
        }
    } else {
      std::vector<std::pair<int,double> > birdProbs;

      for (size_t i = 0; i < pState.getNumBirds(); i++) {
        std::pair<int,double> birdProb = std::pair<int,double> (-1,std::numeric_limits<double>::lowest());

        for (size_t j = 0; j < speciesModels.size(); j++) {
          if (!birdSpeciesFound[j]) continue;
          int numFiniteProbabilities = 0;
          double speciesProb = speciesModels[j].getProbOfObsSeq(pState.getBird(i));
          if (std::isfinite(speciesProb) && speciesProb > birdProb.second) {
            birdProb.first = j;
            birdProb.second = speciesProb;
          }
        }
        birdProbs.push_back(birdProb);
        lGuesses[i] = static_cast<ESpecies>(birdProb.first);
      }

      for (int i=0; i<lGuesses.size(); ++i) {
        if (birdProbs[i].second < -900) {
          if (pState.getRound() > 4)
            lGuesses[i] = SPECIES_UNKNOWN;
          else
            lGuesses[i] = SPECIES_PIGEON;
        }
      }
    }

    return lGuesses;
}

void Player::hit(const GameState &pState, int pBird, const Deadline &pDue)
{
    std::cerr << "HIT BIRD!!!" << std::endl;
}

void Player::reveal(const GameState &pState, const std::vector<ESpecies> &pSpecies, const Deadline &pDue)
{
    if (pState.getRound() == 0) {
      std::vector<std::vector<HMM> > tempModels(COUNT_SPECIES);

      for (size_t i = 0; i < pSpecies.size(); i++) {
        if (pSpecies[i] == -1) continue;

        birdSpeciesFound[static_cast<int>(pSpecies[i])] = true;
        tempModels[pSpecies[i]].push_back(HMMs[i]);
      }
      models = tempModels;

    } else {
      for (size_t i = 0; i < pSpecies.size(); i++) {
        if (pSpecies[i] == -1) continue;
        birdSpeciesFound[static_cast<int>(pSpecies[i])] = true;
        models[pSpecies[i]].push_back(HMMs[i]);
      }
    }
    if (birdSpeciesFound[5]) storkFound = true;

    combineHMMs();
}

void Player::combineHMMs() {
  for (int i=0 ; i<models.size(); ++i) {
    if (!birdSpeciesFound[i]) continue;

    if (models[i].size() > 1) {
      HMM cmbModel = HMM(models[i]);
      speciesModels[i] = cmbModel;

    } else if (models[i].size() == 1) {
      speciesModels[i] = models[i][0];
    }
  }
}


} /*namespace ducks*/
