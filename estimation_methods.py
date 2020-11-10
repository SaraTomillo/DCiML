from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
from utils.methods.DRE import DensityRatioEstimator
import numpy as np
from utils.methods.KMM import iwe_kernel_mean_matching
from utils.utils import T


def kliep(X_train, X_test):
    kliep = DensityRatioEstimator()
    kliep.fit(X_train, X_test)  # keyword arguments are X_train and X_test
    importance = kliep.predict(X_train)
    return importance

def kliep_model(P_train, P_test):
    aux = np.asarray(P_train)
    if len(aux.shape) == 1:
        P_trainT = T(P_train)
        P_testT = T(P_test)
    else:
        P_trainT = P_train
        P_testT = P_test

    kliep = DensityRatioEstimator()

    kliep.fit(P_trainT, P_testT)  # keyword arguments are X_train and X_test
    importance = kliep.predict(P_trainT)
    return importance

def kliep_mixed(P_train, P_test):
    # Necesario añadir un if
    aux = np.asarray(P_train)
    if len(aux.shape) == 1:
        P_trainT = T(P_train)
        P_testT = T(P_test)
    else:
        P_trainT = P_train
        P_testT = P_test

    kliep = DensityRatioEstimator()

    P_testT = T(P_test)
    kliep.fit(P_trainT, P_testT)  # keyword arguments are X_train and X_test
    importance = kliep.predict(P_trainT)
    return importance

def kernel_density(X_train, X_test, bandwidth, kernel):
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    # Train's DF
    kde.fit(X_train)
    DF_trainV = kde.score_samples(X_train)
#    DF_train = list(map(lambda x: np.exp(x), DF_trainV))

    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    # Test's DF
    kde.fit(X_test)
    DF_testV = kde.score_samples(X_train)
#    DF_test = list(map(lambda x: np.exp(x), DF_testV))

    importance = np.zeros(len(X_train))
    for i in range(len(X_train)):
#       importance[i] = DF_test[i] / DF_train[i]
        importance[i] =np.exp(DF_testV[i]-DF_trainV[i])

    # Adjust for mean(importance)=1
    importance_sum = np.sum(importance)
    if importance_sum == 0:
        for i in range(len(X_train)):
            importance[i] = 1
        importance_sum = np.sum(importance)

    Mean1Coef = len(X_train) / importance_sum
    for i in range(len(X_train)):
        importance[i] = importance[i] * Mean1Coef
    return importance

def kernel_density_model(P_train, P_test, bandwidth, kernel):
    aux = np.asarray(P_train)
    if len(aux.shape) == 1:
        P_trainV = list(map(lambda x: [x], P_train))
        P_testV = list(map(lambda x: [x], P_test))
    else:
        P_trainV = P_train
        P_testV = P_test

    # kde = KernelDensity(bandwidth=np.std(P_train), kernel='epanechnikov')
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)

    # Train's DF
    kde.fit(P_trainV)
    DF_trainV = kde.score_samples(P_trainV)
    #DF_train = list(map(lambda x: np.exp(x), DF_trainV))

    # kde = KernelDensity(bandwidth=np.std(P_test), kernel='epanechnikov')
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    # Test's DF
    kde.fit(P_testV)
    DF_testV = kde.score_samples(P_trainV)
    #DF_test = list(map(lambda x: np.exp(x), DF_testV))

    importance = np.zeros(len(P_train))
    for i in range(len(P_train)):
        #importance[i] = DF_test[i] / DF_train[i]
        importance[i] = np.exp(DF_testV[i] - DF_trainV[i])

    # Adjust for mean(importance)=1
    importance_sum = np.sum(importance)
    if importance_sum == 0:
        for i in range(len(P_train)):
            importance[i] = 1
        importance_sum = np.sum(importance)

    Mean1Coef = len(P_train) / importance_sum
    for i in range(len(P_train)):
        importance[i] = importance[i] * Mean1Coef
    return importance


def kernel_density_mixed(P_train, P_test, bandwidth, kernel):
    aux = np.asarray(P_train)
    if len(aux.shape) == 1:
        P_trainV = list(map(lambda x: [x], P_train))
        P_testV = list(map(lambda x: [x], P_test))
    else:
        P_trainV = P_train
        P_testV = P_test

    # kde = KernelDensity(bandwidth=np.std(P_train), kernel='epanechnikov')
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)

    # Train's DF
    kde.fit(P_trainV)
    DF_trainV = kde.score_samples(P_trainV)
    #DF_train = list(map(lambda x: np.exp(x), DF_trainV))

    # kde = KernelDensity(bandwidth=np.std(P_test), kernel='epanechnikov')
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    # Test's DF
    aux = []
    for element in P_testV:
        aux.append([element])
    P_testV = np.array(aux)

    kde.fit(P_testV)
    DF_testV = kde.score_samples(P_trainV)
    #DF_test = list(map(lambda x: np.exp(x), DF_testV))

    importance = np.zeros(len(P_train))
    for i in range(len(P_train)):
        #importance[i] = DF_test[i] / DF_train[i]
        importance[i] = np.exp(DF_testV[i] - DF_trainV[i])

    # Adjust for mean(importance)=1
    importance_sum = np.sum(importance)
    if importance_sum == 0:
        for i in range(len(P_train)):
            importance[i] = 1
        importance_sum = np.sum(importance)

    Mean1Coef = len(P_train) / importance_sum
    for i in range(len(P_train)):
        importance[i] = importance[i] * Mean1Coef
    return importance

def log_regression(X_train, X_test, random):
    X = np.concatenate((X_train, X_test))
    y = np.zeros(len(X))
    y[:len(X_train)] = 1
    Y = y
    log_reg = LogisticRegression(random_state=random)
    log_reg.fit(X, Y)
    Prob_train = log_reg.predict_proba(X_train)[:, 1]  # clases [0,1]
    Prob_test = log_reg.predict_proba(X_train)[:, 0]  # 1 - P_train
    importance = Prob_train / Prob_test

    # Adjust for mean(importance)=1
    importance_sum = np.sum(importance)
    if importance_sum == 0:
        for i in range(len(X_train)):
            importance[i] = 1
        importance_sum = np.sum(importance)

    Mean1Coef = len(X_train) / importance_sum
    for i in range(len(X_train)):
        importance[i] = importance[i] * Mean1Coef

    return importance


def log_regression_model(P_train, P_test, random):
    X = []
    for element in P_train:
        X.append([element])
    for element in P_test:
        X.append([element])
    y = np.zeros(len(X))
    y[:len(P_train)] = 1
    Y = y
    log_reg = LogisticRegression(random_state=random)
    log_reg.fit(X, Y)

    ptrain = []
    for element in P_train:
        ptrain.append([element])

    Prob_train = log_reg.predict_proba(ptrain)[:, 1]  # clases [0,1]
    Prob_test = log_reg.predict_proba(ptrain)[:, 0]  # 1 - P_train
    importance = Prob_train / Prob_test

    # Adjust for mean(importance)=1
    importance_sum = np.sum(importance)
    if importance_sum == 0:
        for i in range(len(P_train)):
            importance[i] = 1
        importance_sum = np.sum(importance)

    Mean1Coef = len(P_train) / importance_sum
    for i in range(len(P_train)):
        importance[i] = importance[i] * Mean1Coef

    return importance



def log_regression_mixed(Y_train, P_test, random):
    X = []
    for element in Y_train:
        X.append([element])
    for element in P_test:
        X.append([element])
    y = np.zeros(len(X))
    y[:len(Y_train)] = 1
    Y = y
    log_reg = LogisticRegression(random_state=random)
    log_reg.fit(X, Y)

    Prob_train = log_reg.predict_proba(Y_train)[:, 1]  # clases [0,1]
    Prob_test = log_reg.predict_proba(Y_train)[:, 0]  # 1 - P_train
    importance = Prob_train / Prob_test

    # Adjust for mean(importance)=1
    importance_sum = np.sum(importance)
    if importance_sum == 0:
        for i in range(len(Y_train)):
            importance[i] = 1
        importance_sum = np.sum(importance)

    Mean1Coef = len(Y_train) / importance_sum
    for i in range(len(Y_train)):
        importance[i] = importance[i] * Mean1Coef

    return importance


def log_regression_model_classification(P_train, P_test, random):
    X = np.concatenate((P_train, P_test))
    y = np.zeros(len(X))
    y[:len(P_train)] = 1
    Y = y
    log_reg = LogisticRegression(random_state=random)
    log_reg.fit(X, Y)
    Prob_train = log_reg.predict_proba(P_train)[:, 1]  # clases [0,1]
    Prob_test = log_reg.predict_proba(P_train)[:, 0]  # 1 - P_train

    importance = Prob_train / Prob_test

    # Adjust for mean(importance)=1
    importance_sum = np.sum(importance)
    if importance_sum == 0:
        for i in range(len(P_train)):
            importance[i] = 1
        importance_sum = np.sum(importance)

    Mean1Coef = len(P_train) / importance_sum
    for i in range(len(P_train)):
        importance[i] = importance[i] * Mean1Coef

    return importance


def kmm(X_train, X_test):
    importance = iwe_kernel_mean_matching(X_train, X_test)
    return importance


def kmm_model(P_train, P_test):
    train = P_train

    train = []
    for element in P_train:
        train.append([element])
    train = np.array(train)

    test = []
    for element in P_test:
        test.append([element])
    test = np.array(test)

    importance = iwe_kernel_mean_matching(train, test)
    return importance

def kmm_mixed(Y_train, P_test):
    train = Y_train
    test = []
    for element in P_test:
        test.append([element])
    test = np.array(test)

    importance = iwe_kernel_mean_matching(train, test)
    return importance


def kmm_model_classification(P_train, P_test):
    importance = iwe_kernel_mean_matching(P_train, P_test)
    return importance


