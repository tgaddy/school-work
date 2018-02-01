import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.metrics.pairwise import pairwise_kernels
import pickle
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pylab as plt
from sklearn.datasets import *


# --------- Kernel Functions ---------------------------
def linear_spline(a,b, **kwargs):
    return 1 + a*b + a*b*min(a,b) -\
           0.5*(a+b)*min(a,b)**2 +\
           (1/3)*(min(a,b)**3)

def gaussian_kernel(a,b,**kwargs):
    if "gamma" in kwargs.keys():
        return np.exp((- np.linalg.norm(a-b) * kwargs["gamma"]))
    else:
        raise Exception("Please specify gamma")

def get_kernel(X, kernel, Y=None, **kwargs):
    # Manual kernel calculation
    # Should return same value as scikitlearn for same kernels
    # TODO: Vectorize
    if callable(kernel):
        if Y is None:
            K = np.zeros(shape=(len(X),len(X)))
            for j in range(K.shape[0]):
                for i in range(K.shape[1]):
                    K[j,i] = kernel(X[i], X[j], **kwargs)
        else:
            K = np.zeros(shape=(len(X),len(Y)))
            for j in range(K.shape[0]):
                for i in range(K.shape[1]):
                    K[j,i] = kernel(Y[i], X[j], **kwargs)

    # If using scikitlearn
    else:
        # If Y is None => just pairwise kernel for X
        if len(X.shape) < 2:
            X = X.reshape(-1,1)
        K = pairwise_kernels(X=X, Y=Y, metric=kernel, **kwargs)


    return K

def initialize_variable(shape, random_offset):
    """
    Initializes vector with zero values or randomly offset
    using values drawn from random distribution.
    Used for alphas and W.
    """
    var = np.zeros(shape = shape)
    if random_offset:
        offset = np.random.normal(loc=0, scale=1, size=len(var))**2
        var += (offset/offset.sum()) # Keep them small
    return var

def get_yn(W, D, **kwargs):
    """
    New function to get y_n for RVM classification.
    This should be the right one (also much faster).
    If correct though.
    """
    # Same as params["W"].T.dot(params["D"][i,:])
    # For each i in N
    return D.dot(W)


# -------------- Parameter functions ---------------------------
def get_D(X, K):
    # Switched order
    return np.concatenate([K,
                           np.ones(shape = (len(X),1))],
                           axis=1)

def get_Sigma(D, A, **kwargs):
    if "B" in kwargs.keys():
        # Classification
        B = kwargs["B"]
        S = np.linalg.inv((D.T.dot(B).dot(D)) + A)
    elif "sigmasq" in kwargs.keys():
        # Regression
        sigmasq = kwargs["sigmasq"]
        S = np.linalg.inv(sigmasq**(-1)*D.T.dot(D) + A)
    return S

def get_mu(sigmasq, Sigma, D, Y):
    return sigmasq**(-1)*Sigma.dot(D.T).dot(Y)

def get_gamma(alpha, Sigma):
    return 1 - alpha * np.diag(Sigma)

def get_alpha(gamma, mu):
    # np.where to deal with case where mu == 0
    # There should be no such case after pruning
    a = np.where(mu==0.0, np.NaN, mu)
    b = gamma/(a**2)
    return np.where(np.isnan(b), 0, b)

def get_sigmasq(Y, D, mu, N, gamma):
    a = np.linalg.norm(Y - D.dot(mu))
    return a/(N - np.sum(gamma))

def add_label(x, label):
    labels = np.zeros(shape=(len(x),1)) + label
    return np.concatenate([x, labels], axis=1)

# --------------------- Wrapper for functions --------------------------------
def initiate(X, Y, kernel, random_offset, alpha=None,
             sigmasq=None, W=None, **kwargs):
    # Create kernel
    K = get_kernel(X, kernel, **kwargs)
    # Design matrix
    D = get_D(X,K)
    N = len(Y)

    # Alphas
    if alpha is None:
        alpha = initialize_variable(shape=(D.shape[1]),
                                    random_offset=random_offset)
    # Weights
    if W:
        # Initiate them as zeros
        w = initialize_variable(shape=(D.shape[1]),
                                    random_offset=False)
    # Initial A matrix
    A = np.diag(alpha)
    # Relevant vectors
    rv = np.array(range(len(alpha)))
    # Boolean mask for relevant vectors
    active = np.ones(len(alpha), dtype=int)

    # If regression
    if sigmasq:
        params = {"A":A, "D":D, "alpha":alpha,
              "Y":Y, "rv":rv, "N":N, "X":X, "sigmasq":sigmasq,
              "active":active, "K":K}
    # If classification
    elif W:
        params = {"A":A, "D":D, "alpha":alpha,
              "Y":Y, "rv":rv, "N":N, "X":X, "W":w,
              "active":active, "K":K}

    return params

def update_r(params, max_iter, verbose, threshold, cc, **kwargs):
    results = [params]
    iters = 0
    while True:
        op = results[-1]

        # Update variables
        # Procedure from Tipping's paper to my understanding

        Sigma = get_Sigma(D=op["D"], A=op["A"], sigmasq=op["sigmasq"])
        mu = get_mu(sigmasq=op["sigmasq"], Sigma=Sigma, D=op["D"],
                    Y=op["Y"])
        gamma = get_gamma(alpha=op["alpha"], Sigma=Sigma)
        if "gamma" in op.keys():
            # If not first iteration
            alpha = get_alpha(gamma=op["gamma"], mu=mu)
            sigmasq = get_sigmasq(Y=op["Y"], D=op["D"], mu=mu, N=op["N"],
                              #gamma=op["gamma"]
                              gamma=gamma
                              )
        else:
            # If first iteration, use current gamma values
            alpha = get_alpha(gamma=gamma, mu=mu)
            sigmasq = get_sigmasq(Y=op["Y"], D=op["D"], mu=mu, N=op["N"],
                                  gamma=gamma)

        change_alpha = sum(alpha) - sum(op["alpha"])
        # Prune irrelevant vectors
        mask = gamma >= threshold
        mu = mu[mask]
        gamma = gamma[mask]

        alpha = alpha[mask]
        A = np.diag(alpha)
        temp = Sigma[mask]
        Sigma = temp[:,mask]
        D = op["D"][:,mask]
        rv = op["rv"][mask]

        # Add updated parameters to list
        res = {"A":A, "D":D, "gamma":gamma, "alpha":alpha,
                        "Sigma":Sigma,
                        "sigmasq":sigmasq, "Y":op["Y"], "mu":mu, "N":op["N"],
                        "rv":rv, "X":op["X"], "change_alpha":change_alpha}

        # TODO: Calulate objective correctly
        #objective = get_objective(res)
        #res["objective"] = n if abs(n)!=np.inf else np.NaN

        results.append(res)

        if verbose:
            print ("Pruning {0} element(s)".format(len(mask)-mask.sum()))
            print ("Alpha [{0:.2f}, {1:.2f}]".format(min(alpha),
                                                     max(alpha)))
            print ("Gamma [{0:.2f}, {1:.2f}]".format(min(gamma),
                                                     max(gamma)))
            print ("Sigma [{0:.2f}, {1:.2f}]".format(min(np.diag(Sigma)),
                                                     max(np.diag(Sigma))))
            print ("sigma square = {0:.6f}".format(op["sigmasq"]))
            print ("\n----   -----   ----\n")
        iters += 1

        # Condition to exit the loop
        # Convergence criterion
        delta = np.amax(np.absolute(alpha - op["alpha"][mask]))
        if ((delta < cc) and (iters > 1)) | (iters >= max_iter):

            print ("Number of iterations: ", iters)
            print ("Initial variance = {0:.4f}; Converged variance: {1:.4f}"\
                   .format(results[0]["sigmasq"], results[-1]["sigmasq"]))
            print("Number of relevant vectors: ", len(results[-1]["rv"]))

            return results

            break


def log_posterior(w, A, D, Y):
    '''
    Define the log posterior.
    Inputs:
       parameters
       estimated outputs (yn)
    Outputs:
       Approximation to the posterior distribution
       Jacobian

    Y are true targets, yn are estimates
    Define the log-posterior (negative of book so that we can minimize
    instead of maximize)
    '''
    yn = get_yn(D=D, W=w)
    # Logit the yn values now
    yn = expit(yn)
    log_p = -1 * (np.sum(np.log(yn[Y == 1]),0) + np.sum(np.log(1-yn[Y==0]),0))
    log_p = log_p + 0.5*w.T.dot(A.dot(w))
    # Define the jacobian
    jacobian = A.dot(w) - D.T.dot(Y-yn)

    return log_p, jacobian

def hessian(w, A, D, Y):
    '''
    Defines the hessian of the log-posterior distribution.
    Inputs:
       Parameters
       B matrix (NxN diagonal with elements b_i = y_i*(1-y_i))
    Outputs:
       Hessian of the log-posterior distribution
    '''
    yn = get_yn(D=D, W=w)
    # Logit the yn values now
    yn = expit(yn)
    # Shape N x N
    B = np.diag(yn*(1-yn))

    return A + D.T.dot(B.dot(D))

def update_c(params, max_iter, threshold=1e9, cc=0.0001, **kwargs):
    """
    Update algo for classification
    """
    results = [params]
    iters = 0
    alpha_old = params["alpha"]
    while True:
        op = results[-1]

        # Estimate output based on current params
        yn = get_yn(D=op["D"], W=op["W"])
        # Logit the yn values now
        yn = expit(yn)
        betas = yn*(1-yn)
        B = np.diag(betas)

        # Approximate the posterior
        posterior = minimize(
            fun = log_posterior,
            hess = hessian,
            x0 = op["W"],
            args = (op["A"], op["D"], op["Y"]),
            method = 'Newton-CG',
            jac = True,
            options = {'maxiter': 50}
        )

        w = posterior.x
        #print("Alphas", len(alpha_old))

        Sigma = np.linalg.inv(hessian(w, op["A"], op["D"], op["Y"]))
        gamma = get_gamma(op["alpha"], Sigma)
        alpha = get_alpha(gamma, w)

        mask = alpha <= threshold
        rv = op["rv"][mask]
        alpha = alpha[mask]
        alpha_old = alpha_old[mask]
        D = op["D"][:,mask]
        A = np.diag(alpha)
        w = w[mask]

        # Add updated parameters to list (why append)
        results.append({"A":A, "D":D, "gamma":gamma, "alpha":alpha,
                        "Sigma":Sigma, "B":B, "X":op["X"], "betas":betas,
                        "W":w, "Y":op["Y"], "N":op["N"],"rv":rv, "YN":yn})

        #print("Alphas", len(alpha))

        delta = np.amax(np.absolute(alpha - alpha_old))
        if delta < cc and iters > 1:
            return results
            break
        alpha_old = alpha

        iters += 1
        # Condition to exit the loop
        if iters >= max_iter:

            print ("Number of relevant vectors: ", len(rv))
            print ("Number of iterations: ",iters)

            return results
            break

def get_class(X, params, kernel, **kwargs):
    """
    Calculate posterior for class labels for classification.
    """
    # Relevant vectors
    X_ = params["X"]
    if params["rv"][-1] == len(params["X"]):
        #in case we have pruned out the bias, we add it again
        X_ = np.concatenate((X_, np.ones(shape=(1, np.shape(X_)[1]))), axis=0)
    RV = X_[params["rv"]]

    if len(X.shape)<2:
        # Single sample
        X = X.reshape(1, -1)
    D = get_kernel(X=X, Y=RV, kernel=kernel, **kwargs)

    return expit(D.dot(params["W"].reshape(-1,1)))

def predict(Xstar, params, kernel, **kwargs):
    """
    Get prediction mean and std of posterior for regression.
    """
    Kstar = np.zeros(shape=(len(Xstar), len(params["rv"])))
    ystar = np.zeros(shape=(len(Xstar)))
    X = params["X"].reshape(-1,)
    x_ = np.concatenate([X, np.ones(1)], axis=0)
    # For each test data point calculate phi(x)
    if not callable(kernel):
        for i,a in enumerate(Xstar):
            Kstar[i,:] = pairwise_kernels(a, x_[params["rv"]].reshape(-1,1),
                                          metric = kernel, **kwargs)
    else:
        for i,a in enumerate(Xstar):
            for j,b in enumerate(params["rv"]):
                Kstar[i,j] = kernel(a, x_[b], **kwargs)
    # Calculate ystar for each test data point
    for i,v in enumerate(Kstar):
        ystar[i] = params["mu"].T.dot(v)
    # Calculate sigma square for each test data point
    sigmasqstar = np.zeros(shape=(len(Xstar)))
    for i,v in enumerate(Kstar):
        sigmasqstar[i] = params["sigmasq"] + v.T.dot(params["Sigma"]).dot(v)
    return ystar, sigmasqstar, Kstar

def get_objective(params):
    """
    Calculate likelihood objective function.
    """
    I = -np.log(np.linalg.det(params["Sigma"])) - params["N"]\
        *np.log(params["sigmasq"]**(-1))\
        -np.log(np.linalg.det(params["A"]))
    II = params["sigmasq"]**(-1)*params["Y"].T.dot(params["Y"] - \
         params["D"].dot(params["mu"]))
    return -0.5*(I + II)

def plot_regression(params, Y_true, Xstar, text, kernel, **kwargs):
    # Last entry is bias term
    mask = params["rv"][:-1] if params["rv"][-1]\
           ==len(params["X"]) else params["rv"]
    # Predict
    yhat, confint, Kstar = predict(Xstar, params, kernel, **kwargs)
    plt.subplots(figsize=(10,5))
    plt.scatter(params["X"], params["Y"], label="Data", c="black", s=2);
    plt.plot(params["X"], Y_true, c="black", linestyle="-",label="True Function");
    plt.plot(Xstar, yhat, c="black", linestyle="--", label="Prediction");
    plt.fill_between(Xstar, yhat-np.sqrt(confint), yhat+np.sqrt(confint),
                         alpha=0.6, color="salmon", label="Variance")
    plt.scatter(params["X"][mask], params["Y"][mask], label="Relevant Vectors",
                facecolors='none', edgecolors='darkred');
    plt.title("Relevant Vector Machine",{'size':'20'});
    plt.xlabel(r"$X$");
    plt.legend();
    plt.figtext(0.28, -0.05, text, {'size':'15'})
    plt.ylabel(r"$\frac{sin(X)}{X}$", rotation=360, labelpad=20);

def get_grid(X):
    # Get extent of grid
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # Create cartesian coordinates for grid
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                         np.arange(x2_min, x2_max, 0.1))
    # Create list of coordinates for which to evaluate the predictive distribution
    Z = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, Z

def plot_classification(params, kernel, text, **kwargs):

    xx, yy, Z = get_grid(params["X"])

    # TODO: Figure out how to vectorize
    predictions = np.array(list(map(lambda s: get_class(s, params=params,
                                                        kernel=kernel,
                                                        **kwargs),
                                Z)))

    zz = predictions.reshape(xx.shape)
    # SVM like decision boundary
    decision = np.where(zz >= 0.5, 1, 0)

    rv = params["rv"][:-1] if params["rv"][-1]==len(params["X"]) else params["rv"]

    # Visualize
    fig, ax = plt.subplots(figsize=(10,5))
    m = np.array(list(map(lambda s: "x" if s == 1 else "o", params["Y"])))
    unique_markers = np.unique(params["Y"])
    for um in unique_markers:
        mask = params["Y"] == um
        plt.scatter(params["X"][:,0][mask], params["X"][:,1][mask],
                    marker = m[mask][0], c="black");
    plt.scatter(params["X"][rv][:,0], params["X"][rv][:,1], s=120, facecolors='none',
                edgecolors='darkred', label="Relevant Vectors")
    plt.contourf(xx, yy, decision, alpha=0.2, cmap="RdBu")
    plt.contourf(xx, yy, zz, alpha=0.3, cmap="RdBu")
    plt.xlabel(r"$X_1$");
    plt.ylabel(r"$X_2$", rotation=360, labelpad=20);
    plt.legend();
    plt.title("Relevance Vector Machine Classification", {"size":20});
    plt.figtext(0.39, -0.05, text, {'size':'15'});


def load_dataset(name):
    if name == "Classification_data.pickle":
        data = pickle.load(open(name, "rb"))
        X = data["X"]
        Y = data["Y"]
    elif name == "RiplaySynthetic250" or name == "RiplaySynthetic1000":
        data = np.loadtxt(name)
        X = data[:, [0, 1]]
        Y = data[:, 2]
    elif name == "BreastCancerTrain":
        data = load_breast_cancer()
        X = data.data[:199]
        Y = data.target[:199]
    elif name == "PimaDiabetesTrain" or name == "PimaDiabetesTest":
        data = np.loadtxt(name)
        X = data[:, 0:6]
        Y = data[:, 7]
    elif name == "BreastCancerTest":
        data = load_breast_cancer()
        X = data.data[200:]
        Y = data.target[200:]
    elif name == "BananaTrain":
        data = np.loadtxt('BANANA')
        X = data[:399, [0, 1]]
        Y = data[:399, 2]
    elif name == "BananaTest":
        data = np.loadtxt('BANANA')
        X = data[501:1000, [0, 1]]
        Y = data[501:1000, 2]
    elif name == "WaveformTrain":
        data = np.loadtxt('WAVEFORM')
        X = data[:399, 0:20]
        Y = data[:399, 21]
    elif name == "WaveformTest":
        data = np.loadtxt('WAVEFORM')
        X = data[501:1000, 0:20]
        Y = data[501:1000, 21]
    elif name == "BostonTrain":
        boston = load_boston()
        X = boston["data"][:299]
        Y = boston["target"][:299]
    elif name == "BostonTest":
        boston = load_boston()
        X = boston["data"][300:]
        Y = boston["target"][300:]
    else:
        print("WRONG FILE NAME")
        exit(1)

    return (X, Y)




def test_dataset(train_dataset_name, test_dataset_name, fp, k, g):
    testdatapoints, testclasses = load_dataset(test_dataset_name)
    predictions = np.array(list(map(lambda s: get_class(s, fp, k, gamma=g), testdatapoints)))
    predictions = predictions.flatten()
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
    errors = np.sum(np.abs(predictions - testclasses))
    errorrate = (errors * 100) / np.size(testclasses)
    print("For the training dataset", train_dataset_name, "the relevance vectors were", np.size(fp["rv"]))
    print("For the test dataset", test_dataset_name, ", the error rate is", errorrate, "%")
    print("The errors were", int(errors), "in a total of test datapoints", np.size(testclasses))
