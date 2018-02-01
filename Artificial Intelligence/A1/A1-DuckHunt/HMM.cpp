//============================================================================
// Name        : HMM3.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "HMM.hpp"
#include <fstream>

namespace ducks {

HMM::HMM() {
	initializeRandomizedModel();
}

HMM::HMM(int states, int obs) {
	numStates = states;
	numObs = obs;
	initializeRandomizedModel();
}


HMM::HMM(std::vector<HMM> HMMs) {
	A = std::vector<std::vector<double> > (numStates);
	B = std::vector<std::vector<double> > (numStates);
	Pi = std::vector<std::vector<double> > (1);

	for (size_t i = 0; i < numStates; i++) {
		//Calculate A
		for (size_t j = 0; j < numStates; j++) {
			double numA = 0.0, denomA = 0.0;
			for (size_t k = 0; k < HMMs.size(); ++k) {
				double diGamSum = 0.0, gamSum = 0.0;

				for (size_t t = 0; t < HMMs[k].obsSeq.size()-1; t++) {
					diGamSum += HMMs[k].diGamma[t][i][j];
					gamSum += HMMs[k].gamma[t][i];
				}

				numA += diGamSum;
				denomA += gamSum;
			}

			A[i].push_back(numA/denomA);
		}

		//Calculate B
		for (size_t j = 0; j < numObs; j++) {
			double numB = 0.0, denomB = 0.0;
			for (size_t k = 0; k < HMMs.size(); ++k) {
				double gamSum = 0.0, indicatorSum = 0.0;
				for (size_t t = 0; t < HMMs[k].obsSeq.size(); t++) {
					gamSum += HMMs[k].gamma[t][i];
					indicatorSum += (HMMs[k].getObsSeq()[t] == j ? HMMs[k].gamma[t][i] : 0);
				}
				numB += indicatorSum;
				denomB += gamSum;
			}
			B[i].push_back(numB/denomB);
		}

		//Calculate Pi
		double numPi = 0.0;
		for (size_t k = 0; k < HMMs.size(); k++) {
			numPi += HMMs[k].gamma[0][i];
		}
		Pi[0].push_back(numPi/HMMs.size());
	}
}
/*
HMM::HMM(std::vector<HMM> HMMs) {
	A = std::vector<std::vector<double> > (numStates);
	B = std::vector<std::vector<double> > (numStates);
	Pi = std::vector<std::vector<double> > (1);

	for (size_t i = 0; i < numStates; i++) {
		//Calculate A
		for (size_t j = 0; j < numStates; j++) {
			double numA = 0.0;
			for (size_t k = 0; k < HMMs.size(); ++k) {
				numA += HMMs[k].A[i][j];
			}

			A[i].push_back(numA/HMMs.size());
		}

		//Calculate B
		for (size_t j = 0; j < numObs; j++) {
			double numB = 0.0;
			for (size_t k = 0; k < HMMs.size(); ++k) {
				numB += HMMs[k].B[i][j];
			}
			B[i].push_back(numB/HMMs.size());
		}

		//Calculate Pi
		double numPi = 0.0;
		for (size_t k = 0; k < HMMs.size(); k++) {
			numPi += HMMs[k].Pi[0][i];
		}
		Pi[0].push_back(numPi/HMMs.size());
	}
}
*/
void HMM::printMat(std::vector<std::vector<double>> &mat) {
	for (int row = 0; row < mat.size(); ++row) {
		for (int col = 0; col < mat[0].size();++col) {
			std::cout << mat[row][col] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

//element-wise multiplication of matrices
std::vector<std::vector<double> > HMM::matMul(std::vector<std::vector<double> > &matA, std::vector<std::vector<double> > &matB) {
	if (matA[0].size() != matB.size())
		throw std::invalid_argument( "received invalid matrix dimensions" );

	std::vector<std::vector<double> > result(matA.size());

	for (int rowA = 0; rowA < matA.size(); ++rowA) {
		std::vector<double> partA = matA[rowA];

		for (int colB = 0; colB < matB[0].size(); ++colB) {

			std::vector<double> partB;
			for (int rowB = 0; rowB < matB.size(); ++rowB) {
				partB.push_back(matB[rowB][colB]);
			}

			//Doing element-wise multiplication of partA and partB
			std::vector<double> indexParts(partA.size());
			std::transform( partA.begin(), partA.end(),
			                partB.begin(), indexParts.begin(),
			                std::multiplies<double>() );

			double sum  = 0.0;
			for (double elem : indexParts) {
				sum += elem;
			}
			result[rowA].push_back(sum);
		}
	}

	return result;
}

std::vector<int> HMM::generateViterbiPath(std::vector<std::vector<double> > &matA, std::vector<std::vector<double> > &matB, std::vector<std::vector<double> > &matPi, std::vector<int> &obsSeq) {
	std::vector<std::vector<double> > delta(obsSeq.size());
	std::vector<std::vector<int> > deltaIndex(obsSeq.size());
	std::vector<int> bestPath;

	//Calc delta0s
	for (int i=0; i< matPi[0].size(); ++i) {
		delta[0].push_back(matPi[0][i] * matB[i][obsSeq[0]]);
		deltaIndex[0].push_back(0);
	}


	for (int t=1; t<obsSeq.size(); ++t) {
		for (int i=0; i < matPi[0].size(); ++i) {
			double max= -0.1;
			int index = -1;
			double currDelta = 0.0;

			for (int j=0; j<matPi[0].size(); ++j) {
				currDelta = delta[t-1][j]*matA[j][i]*matB[i][obsSeq[t]];
				if (currDelta > max) {
					max = currDelta;
					index = j;
				}
			}

			delta[t].push_back(max);
			deltaIndex[t].push_back(index);

		}
	}

	double maxDeltaT = -0.1;
	int lastIndex = -1;
	for (int i = 0; i < delta[obsSeq.size()-1].size(); ++i) {
		if (delta[obsSeq.size()-1][i] > maxDeltaT) {
			maxDeltaT = delta[obsSeq.size()-1][i];
			lastIndex = i;
		}
	}

	bestPath.push_back(lastIndex);

	for (int t = obsSeq.size()-1; t>0; --t) {
		bestPath.push_back(deltaIndex[t][lastIndex]);
		lastIndex = deltaIndex[t][lastIndex];
	}

	reverse(bestPath.begin(), bestPath.end());
	//printMat(delta);
	//cout << endl;
	//printMat(deltaIndex);
	return bestPath;
}

std::vector<std::vector<double> > HMM::calculateAlpha(std::vector<std::vector<double> > &matA, std::vector<std::vector<double> > &matB, std::vector<std::vector<double> > &matPi, std::vector<int> &obsSeq, std::vector<double> &scaleC){
	std::vector<std::vector<double> > alpha(obsSeq.size());

	//Calc alpha0s
	scaleC[0] = 0.0;
	for (int i=0; i < matPi[0].size(); ++i) {
		alpha[0].push_back(matPi[0][i] * matB[i][obsSeq[0]]);
		scaleC[0] += alpha[0][i];
	}
	scaleC[0] = 1/scaleC[0];

	for (int i=0; i < matPi[0].size(); ++i) {
		alpha[0][i] *= scaleC[0];
	}

	for (int t=1; t<obsSeq.size(); ++t) {
		scaleC[t] = 0.0;
		for (int i=0; i < matPi[0].size(); ++i) {
			double sum=0.0;
			for (int j=0; j<matPi[0].size(); ++j) {
				sum += alpha[t-1][j] * matA[j][i];
			}
			alpha[t].push_back(sum*matB[i][obsSeq[t]]);
			scaleC[t] += alpha[t][i];
		}
		scaleC[t] = 1/scaleC[t];
		for (int i=0; i < matPi[0].size(); ++i) {
			alpha[t][i] *= scaleC[t];
		}
	}

	for (int i =0; i<alpha.size(); ++i) {
		for (int j=0; j<alpha[0].size(); ++j) {
			if (std::isnan(alpha[i][j])) alpha[i][j] = 0;
		}
	}
	return alpha;
}


std::vector<std::vector<double> > HMM::calculateBeta(std::vector<std::vector<double> > &matA, std::vector<std::vector<double> > &matB, std::vector<std::vector<double> > &matPi, std::vector<int> &obsSeq, std::vector<double> &scaleC) {
  std::vector<std::vector<double> > beta(obsSeq.size());
	for (int i=0; i< matPi[0].size(); ++i) {
		beta[obsSeq.size()-1].push_back(scaleC[obsSeq.size()-1]);
	}

	for (int t = obsSeq.size()-2; t >= 0; --t) {
		for (int i=0; i < matPi[0].size(); ++i) {
			double betaSum=0.0;

			for (int j=0; j<matPi[0].size(); ++j) {
				betaSum += matA[i][j] * matB[j][obsSeq[t+1]] * beta[t+1][j];
			}
			betaSum *= scaleC[t];
			beta[t].push_back(betaSum);

		}
	}

	for (int i =0; i<beta.size(); ++i) {
		for (int j=0; j<beta[0].size(); ++j) {
			if (std::isnan(beta[i][j])) beta[i][j] = 0;
		}
	}
  return beta;
}

std::vector<std::vector<std::vector<double> > > HMM::CalculateDiGamma(std::vector<std::vector<double> > &matA, std::vector<std::vector<double> > &matB, std::vector<std::vector<double> > &alpha, std::vector<std::vector<double> > &beta, std::vector<int> &obsSeq) {

	std::vector<std::vector<std::vector<double> > > diGamma(obsSeq.size()-1);
	for (int i = 0; i < diGamma.size(); ++i) {
		diGamma[i] = std::vector<std::vector<double> >(matA.size());
	}

	for (int t = 0; t < obsSeq.size()-1; ++t) {
		double denom = 0;
		for (int i = 0; i < matA.size(); ++i) {
			for (int j = 0; j < matA.size(); ++j) {
				denom += alpha[t][i]*matA[i][j]*matB[j][obsSeq[t+1]]*beta[t+1][j];
			}
		}

		for (int i=0; i< matA.size(); ++i) {
			for (int j = 0; j < matA.size(); ++j) {
				diGamma[t][i].push_back((alpha[t][i] * matA[i][j] * matB[j][obsSeq[t+1]] * beta[t+1][j]) / denom);
			}
		}
	}
	for (int i = 0; i < diGamma.size(); ++i) {
		for (int j=0; j< diGamma[0].size(); ++j) {
			for (int k = 0; k < diGamma[0][0].size(); ++k) {
				if (std::isnan(diGamma[i][j][k])) diGamma[i][j][k] = 0;
			}
		}
	}

	return diGamma;
}

std::vector<std::vector<double> > HMM::calculateGamma(std::vector<std::vector<std::vector<double> > > &diGamma, int obsSeqSize, int numStates, std::vector<std::vector<double> > &alpha) {
	std::vector<std::vector<double> > gamma(obsSeqSize);
	for (int t = 0; t < obsSeqSize-1; ++t) {
		for (int i = 0; i < numStates; ++i) {
			double gamSum = 0.0;
			for (int j = 0; j < numStates; ++j) {
				gamSum += diGamma[t][i][j];
			}
			gamma[t].push_back(gamSum);
		}
	}

	double denom = 0;
	for (int i=0; i < numStates; ++i) {
		denom += alpha[obsSeqSize-1][i];
	}
	for (int i=0; i < numStates; ++i) {
		gamma[obsSeqSize-1].push_back(alpha[obsSeqSize-1][i]/denom);
	}
	for (int i =0; i<gamma.size(); ++i) {
		for (int j=0; j<gamma[0].size(); ++j) {
			if (std::isnan(gamma[i][j])) gamma[i][j] = 0;
		}
	}
	return gamma;
}

void HMM::updateA(std::vector<std::vector<double> > &matA, std::vector<std::vector<std::vector<double> > > &diGamma, std::vector<std::vector<double> > &gamma) {
	for (int i = 0; i < matA.size(); ++i) {
		for (int j = 0; j < matA.size(); ++j) {
			double diGamSum = 0.0;
			double gamSum = 0.0;
			for (int t = 0; t < gamma.size()-1; ++t) {
				diGamSum += diGamma[t][i][j];
				gamSum += gamma[t][i];
			}
			matA[i][j] = (std::isnan(diGamSum/gamSum) ? 0 : diGamSum/gamSum);
		}
	}
}

void HMM::updateB(std::vector<std::vector<double> > &matB, std::vector<std::vector<double> > &gamma, std::vector<int> &obsSeq) {
	for (int j = 0; j < gamma[0].size(); ++j) {
		for (int k = 0; k < matB[0].size(); ++k) {
			double indicatorSum = 0.0;
			double gamSum = 0.0;
			for (int t = 0; t < obsSeq.size(); ++t) {
				gamSum += gamma[t][j];
				indicatorSum += (obsSeq[t] == k ? gamma[t][j] : 0);
			}
			matB[j][k] = (std::isnan(indicatorSum/gamSum) ? 0 : indicatorSum/gamSum);
		}
	}
}

void HMM::updatePi(std::vector<std::vector<double> > &Pi, std::vector<std::vector<double> > &gamma) {
	for (int i=0; i < Pi[0].size(); ++i) {
		Pi[0][i] = gamma[0][i];
	}
}


std::vector<std::vector<double> > HMM::initializeRandomMatrix(int rows, int cols) {
	srand(time(NULL));
	std::vector<std::vector<double> > mat(rows);

	for (size_t i = 0; i < rows; i++) {
		double rowSum = 0.0;
		for (size_t j = 0; j < cols; j++) {
			double number = double(rand())/RAND_MAX;
			mat[i].push_back(number);
			rowSum += number;
		}

		for (size_t j = 0; j < rows; j++) {
			mat[i][j] /= rowSum;
		}
	}
	return mat;
}

void HMM::initializeRandomizedModel() {
	A = initializeRandomMatrix(numStates,numStates);
	B = initializeRandomMatrix(numStates,numObs);
	Pi = initializeRandomMatrix(1,numStates);
}

std::pair<int,double> HMM::predictNextMovement(const Bird &bird, double shootThresh) {
	std::vector<int> newObsSeq = extractObsSeq(bird);
	std::vector<int> viterbiPath = generateViterbiPath(A,B,Pi,newObsSeq);
	std::vector<std::vector<double> > probsOfStates;
	std::vector<double> temp(numStates);

	for (size_t i = 0; i < numStates; i++) {
		temp[i] = (i == viterbiPath[viterbiPath.size()-1] ? 1 : 0);
	}

	probsOfStates.push_back(temp);

	std::vector<std::vector<double> > intermediate = matMul(probsOfStates,A);
	std::vector<std::vector<double> > probsOfObs = matMul(intermediate,B);

	double maxProb = shootThresh;
	int mostProbMovement = -1;
	for (size_t i = 0; i < probsOfObs[0].size(); i++) {
		if (probsOfObs[0][i] > maxProb) {
			maxProb = probsOfObs[0][i];
			mostProbMovement = i;
		}
	}
	return std::pair<int,double>(mostProbMovement, maxProb);
}

void HMM::updateObsSeq(EMovement observation) {
	obsSeq.push_back(observation);
}

//Training the model using the Baum-Welch algorithm
void HMM::trainModel(const Deadline &pDue) {
	std::vector<double> c(obsSeq.size());
	alpha = calculateAlpha(A, B, Pi, obsSeq, c);
	beta = calculateBeta(A, B, Pi, obsSeq, c);
	diGamma = CalculateDiGamma(A, B, alpha, beta, obsSeq);
	gamma = calculateGamma(diGamma,obsSeq.size(), A.size(), alpha);

	double oldProb = std::numeric_limits<double>::lowest();

	while(pDue.remainingMs() > 5) {
		double logProb = 0.0;

		updateA(A, diGamma, gamma);
		updateB(B, gamma, obsSeq);
		updatePi(Pi, gamma);

		alpha = calculateAlpha(A, B, Pi, obsSeq, c);
		beta = calculateBeta(A, B, Pi, obsSeq, c);
		diGamma = CalculateDiGamma(A, B, alpha, beta, obsSeq);
		gamma = calculateGamma(diGamma,obsSeq.size(), A.size(), alpha);

		for (int t=0; t<obsSeq.size(); ++t) {
			logProb += log(c[t]);
		}
		logProb = -logProb;

		//Stopping the training if the probability of obsSeq stops improving
		if (logProb/oldProb >= 0.99) break;
		oldProb = logProb;
	}
}

void HMM::trainModelOnObsSeq(const Bird &bird, const Deadline &pDue) {
	obsSeq = extractObsSeq(bird);
	trainModel(pDue);
}

void HMM::trainModelOnObsSeq(std::vector<int> obsSequence, const Deadline &pDue) {
	obsSeq = obsSequence;
	trainModel(pDue);

}

void HMM::setNumStates(int nStates) {
	numStates = nStates;
}

void HMM::setNumObs(int nObs) {
	numObs = nObs;
}

double HMM::getProbOfObsSeq(const Bird &bird) {
	std::vector<int> newObsSeq = extractObsSeq(bird);
	return getProb(newObsSeq);
}

std::vector<int> HMM::extractObsSeq(const Bird &bird) {
	std::vector<int> newObsSeq;
	for (int i=0; i<bird.getSeqLength(); i++) {
		if (bird.getObservation(i) == MOVE_DEAD) break;
		newObsSeq.push_back(bird.getObservation(i));
	}
	return newObsSeq;
}

double HMM::getProbOfObsSeq(std::vector<int> newObsSeq) {
	return getProb(newObsSeq);
}

double HMM::getProb(std::vector<int> newObsSeq) {
	std::vector<double> c(newObsSeq.size());
	std::vector<std::vector<double> > newAlpha = calculateAlpha(A,B,Pi,newObsSeq,c);

	double logProb = 0.0;

	for (int t=0; t<c.size(); ++t) {
		logProb += log(c[t]);
	}
	logProb = -logProb;

	return logProb;
}


std::vector<int> HMM::getObsSeq() {
	return obsSeq;
}

}
