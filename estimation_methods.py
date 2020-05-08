from sklearn.neighbors import KernelDensity
from DRE import DensityRatioEstimator
import numpy as np

# Aux function to transpose 1 dim
def T(V):
    VT = np.zeros([len(V), 1])
    for i in range(len(V)):
        VT[i] = V[i]
    return VT

def kliep(X_train, X_test):
    kliep = DensityRatioEstimator()
    kliep.fit(X_train, X_test)  # keyword arguments are X_train and X_test
    importance = kliep.predict(X_train)
    return importance

def kliep_model(P_train, P_test):
    # Necesario añadir un if
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

def kernel_density(X_train, X_test, bandwidth, kernel):
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    # Train's DF
    kde.fit(X_train)
    DF_trainV = kde.score_samples(X_train)
    DF_train = list(map(lambda x: np.exp(x), DF_trainV))

    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    # Test's DF
    kde.fit(X_test)
    DF_testV = kde.score_samples(X_train)
    DF_test = list(map(lambda x: np.exp(x), DF_testV))

    DF_train = DF_train
    DF_test = DF_test

    importance = np.zeros(len(X_train))
    for i in range(len(X_train)):
        importance[i] = DF_test[i] / DF_train[i]

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
    DF_train = list(map(lambda x: np.exp(x), DF_trainV))

    # kde = KernelDensity(bandwidth=np.std(P_test), kernel='epanechnikov')
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    # Test's DF
    kde.fit(P_testV)
    DF_testV = kde.score_samples(P_trainV)
    DF_test = list(map(lambda x: np.exp(x), DF_testV))

    importance = np.zeros(len(P_train))
    for i in range(len(P_train)):
        importance[i] = DF_test[i] / DF_train[i]

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