import argparse
import traceback

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVR

import estimation_methods
from utils import IO

from utils.utils import minmax, reduce, npShuffle

def error(predictions, y):
    return abs(predictions - y)

bandwith = 1
kernel = 'linear'

def regression(X_train, Y_train, X_test, Y_test, seed):
    random = np.random.RandomState(seed)
    model = SVR(kernel='linear', C=1, coef0=0)

    minmax_X = minmax(X_train)
    X_train = reduce(X_train, minmax_X)
    X_test = reduce(X_test, minmax_X)

    minmax_Y = minmax(Y_train)
    Y_train = reduce(Y_train, minmax_Y)
    Y_test = reduce(Y_test, minmax_Y)

    model.fit(X_train, Y_train)

    Pred_Train = model.predict(X_train)
    Pred_Eval = model.predict(X_test)

    X_train=npShuffle(X_train,seed)
    Y_train=npShuffle(Y_train,seed)

    eval = error(Pred_Eval, Y_test)
    CV = cross_val_predict(model, X_train, Y_train, cv=10)
    CVError = error(CV, Y_train)

    importanceModel = LogisticRegression(random_state=random, C=1, coef0=0)
    importanceModel.fit(X_train, Y_train)

    P_train = importanceModel.predict(X_train)
    P_test = importanceModel.predict(X_test)


    train = []
    train.append(Y_train)
    train.append(P_train)

    test = []
    test.append(Y_test)
    test.append(P_test)

    try:
        importances_logReg = estimation_methods.log_regression(X_train, X_test, random)
        LR_err = CVError * importances_logReg
    except Exception:
        print("Fallo LR")
        LR_err = []
        traceback.print_exc()

    try:
        importances_MLogReg = estimation_methods.log_regression_model(P_train, P_test, random)
        PLR_err = CVError * importances_MLogReg
    except Exception:
        print("Fallo PLR")
        PLR_err = []
        traceback.print_exc()

    try:
        importances_logReg = estimation_methods.log_regression(Y_train, Y_test, random)
        CLR_err = CVError * importances_logReg
    except Exception:
        print("Fallo CLR")
        CLR_err = []
        traceback.print_exc()

    try:
        importances_MLogReg = estimation_methods.log_regression_mixed(Y_train, P_test, random)
        MLR_err = CVError * importances_MLogReg
    except Exception:
        print("Fallo MLR")
        MLR_err = []
        traceback.print_exc()

    try:
        importances_logReg = estimation_methods.log_regression(train, test, random)
        BLR_err = CVError * importances_logReg
    except Exception:
        print("Fallo BLR")
        BLR_err = []
        traceback.print_exc()

    try:
        importances_KMM = estimation_methods.kmm(X_train, X_test)
        KMM_err = CVError * importances_KMM
    except Exception:
        print("Fallo KMM")
        KMM_err = []
        traceback.print_exc()

    try:
        importances_MKMM = estimation_methods.kmm_model(P_train, P_test)
        PKMM_err = CVError * importances_MKMM
    except Exception:
        print("Fallo PKMM")
        PKMM_err = []
        traceback.print_exc()

    try:
        importances_KMM = estimation_methods.kmm(Y_train, Y_test)
        CKMM_err = CVError * importances_KMM
    except Exception:
        print("Fallo CKMM")
        CKMM_err = []
        traceback.print_exc()

    try:
        importances_MKMM = estimation_methods.kmm_mixed(Y_train, P_test)
        MKMM_err = CVError * importances_MKMM
    except Exception:
        print("Fallo MKMM")
        MKMM_err = []
        traceback.print_exc()

    try:
        importances_KMM = estimation_methods.kmm(train, test)
        BKMM_err = CVError * importances_KMM
    except Exception:
        print("Fallo BKMM")
        BKMM_err = []
        traceback.print_exc()

    try:
        importances_kernel_density = estimation_methods.kernel_density(X_train, X_test, bandwith, kernel)
        KDE_err = CVError * importances_kernel_density
    except Exception:
        print("Fallo KDE")
        KDE_err = []
        traceback.print_exc()

    try:
        importances_kernel_density_model = estimation_methods.kernel_density_model(P_train, P_test, bandwith, kernel)
        PKDE_err = CVError * importances_kernel_density_model
    except Exception:
        print("Fallo PKDE")
        PKDE_err = []
        traceback.print_exc()

    try:
        importances_kernel_density = estimation_methods.kernel_density(Y_train, Y_test, bandwith, kernel)
        CKDE_err = CVError * importances_kernel_density
    except Exception:
        print("Fallo CKDE")
        CKDE_err = []
        traceback.print_exc()

    try:
        importances_kernel_density_model = estimation_methods.kernel_density_mixed(Y_train, P_test, bandwith, kernel)
        MKDE_err = CVError * importances_kernel_density_model
    except Exception:
        print("Fallo MKDE")
        MKDE_err = []
        traceback.print_exc()

    try:
        importances_kernel_density = estimation_methods.kernel_density(train, test, bandwith, kernel)
        BKDE_err = CVError * importances_kernel_density
    except Exception:
        print("Fallo BKDE")
        BKDE_err = []
        traceback.print_exc()

    try:
        importances_kliep = estimation_methods.kliep(X_train, X_test)
        KLIEP_err = CVError * importances_kliep
    except Exception:
        print("Fallo KLIEP")
        KLIEP_err = []
        traceback.print_exc()

    try:
        importances_kliep_model = estimation_methods.kliep_model(P_train, P_test)
        PKLIEP_err = CVError * importances_kliep_model
    except Exception:
        print("Fallo PKLIEP")
        PKLIEP_err = []
        traceback.print_exc()

    try:
        importances_kliep = estimation_methods.kliep(Y_train, Y_test)
        CKLIEP_err = CVError * importances_kliep
    except Exception:
        print("Fallo CKLIEP")
        CKLIEP_err = []
        traceback.print_exc()

    try:
        importances_kliep_model = estimation_methods.kliep_mixed(Y_train, P_test)
        MKLIEP_err = CVError * importances_kliep_model
    except Exception:
        print("Fallo MKLIEP")
        MKLIEP_err = []
        traceback.print_exc()

    try:
        importances_kliep = estimation_methods.kliep(train, test)
        BKLIEP_err = CVError * importances_kliep
    except Exception:
        print("Fallo BKLIEP")
        BKLIEP_err = []
        traceback.print_exc()

    return [np.average(eval), np.average(CVError),
            np.average(LR_err), np.average(PLR_err), np.average(CLR_err), np.average(MLR_err), np.average(BLR_err),
            np.average(KMM_err), np.average(PKMM_err), np.average(CKMM_err), np.average(MKMM_err), np.average(BKMM_err),
            np.average(KDE_err), np.average(PKDE_err), np.average(CKDE_err), np.average(MKDE_err), np.average(BKDE_err),
            np.average(KLIEP_err), np.average(PKLIEP_err), np.average(CKLIEP_err), np.average(MKLIEP_err),
            np.average(BKLIEP_err)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename_train", help="the csv file containing the train data")
    parser.add_argument("filename_test", help="the csv file containing the test data")
    parser.add_argument("dataset_name", help="the name of the dataset")
    parser.add_argument("execution_id", help="the execution id for the experiment")
    parser.add_argument("seed", help="the seed")
    args = parser.parse_args()

    filename_train = str(args.filename_train)
    filename_test = str(args.filename_test)
    dataset_name = str(args.dataset_name)
    execution_id = str(args.execution_id)
    seed = str(args.seed)

    X_train, Y_train, X_test, Y_test = IO.readDataset(filename_train, filename_test)
    results = regression(X_train, Y_train, X_test, Y_test, seed)
    IO.writeToCSV(results, "regression-LR", dataset_name, execution_id, seed)


if __name__== "__main__":
  main()