#ifndef __HMM_HPP_INCLUDED__
#define __HMM_HPP_INCLUDED__

#include "Constants.hpp"
#include "Deadline.hpp"
#include "Bird.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <climits>

namespace ducks {

class HMM {
public:
  HMM();
  HMM(int states, int obs);
  HMM(std::vector<HMM> HMMs);
  void printMat(std::vector<std::vector<double>> &mat);
  std::vector<std::vector<double> > matMul(std::vector<std::vector<double> > &matA, std::vector<std::vector<double> > &matB);
  int runStuff();
  void setNumStates(int nStates);
  void setNumObs(int nObs);
  void updateObsSeq(EMovement observation);
  std::pair<int,double> predictNextMovement(const Bird &bird, double shootThresh);
  void trainModelOnObsSeq(const Bird &bird, const Deadline &pDue);
  void trainModelOnObsSeq(std::vector<int> obsSequence, const Deadline &pDue);
  double getProbOfObsSeq(const Bird &bird);
  double getProbOfObsSeq(std::vector<int> newObsSeq);
  std::vector<int> getObsSeq();

private:
  std::vector<std::vector<double> > A;
  std::vector<std::vector<double> > B;
  std::vector<std::vector<double> > Pi;
  std::vector<std::vector<double> > alpha;
  std::vector<std::vector<double> > beta;
  std::vector<std::vector<std::vector<double> > > diGamma;
  std::vector<std::vector<double> > gamma;
  std::vector<int> obsSeq;

  int numStates = 5;
  int numObs = 9;

  std::vector<int> extractObsSeq(const Bird &bird);
  double getProb(std::vector<int> newObsSeq);
  void trainModel(const Deadline &pDue);
  std::vector<std::vector<double> > initializeRandomMatrix(int rows, int cols);
  void initializeRandomizedModel();
  std::vector<int> generateViterbiPath(std::vector<std::vector<double> > &matA, std::vector<std::vector<double> > &matB, std::vector<std::vector<double> > &matPi, std::vector<int> &obsSeq);
  std::vector<std::vector<double> > calculateAlpha(std::vector<std::vector<double> > &matA, std::vector<std::vector<double> > &matB, std::vector<std::vector<double> > &matPi, std::vector<int> &obsSeq, std::vector<double> &scaleC);
  std::vector<std::vector<double> > calculateBeta(std::vector<std::vector<double> > &matA, std::vector<std::vector<double> > &matB, std::vector<std::vector<double> > &matPi, std::vector<int> &obsSeq, std::vector<double> &scaleC);
  std::vector<std::vector<std::vector<double> > > CalculateDiGamma(std::vector<std::vector<double> > &matA, std::vector<std::vector<double> > &matB, std::vector<std::vector<double> > &alpha, std::vector<std::vector<double> > &beta, std::vector<int> &obsSeq);
  std::vector<std::vector<double> > calculateGamma(std::vector<std::vector<std::vector<double> > > &diGamma, int obsSeqSize, int numStates, std::vector<std::vector<double> > &alpha);
  void updateA(std::vector<std::vector<double> > &matA, std::vector<std::vector<std::vector<double> > > &diGamma, std::vector<std::vector<double> > &gamma);
  void updateB(std::vector<std::vector<double> > &matB, std::vector<std::vector<double> > &gamma, std::vector<int> &obsSeq);
  void updatePi(std::vector<std::vector<double> > &Pi, std::vector<std::vector<double> > &gamma);


};

}

#endif
