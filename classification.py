import argparse
import traceback

import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

import estimation_methods
from utils import IO

import tracemalloc
from utils.utils import minmax, reduce, npShuffle


def error(predictions, y):
    return np.sign(np.abs(predictions-y))


bandwith = 1
kernel = 'rbf'

def classification(X_train, Y_train, X_test, Y_test, seed):
    tracemalloc.start()
    model = SVC(kernel='linear', C=1, coef0=0)

    minmax_X = minmax(X_train)
    X_train = reduce(X_train, minmax_X)
    X_test = reduce(X_test, minmax_X)

    model.fit(X_train, Y_train)
    Pred_Eval = model.predict(X_test)

    eval = error(Pred_Eval, Y_test)

    X_train=npShuffle(X_train,seed)
    Y_train=npShuffle(Y_train,seed)

    CV = cross_val_predict(model, X_train, Y_train, cv=10)#, method='decision_function')
    CVError = error(CV, Y_train)

    importanceModel = SVC(kernel='linear', C=1, class_weight='balanced', coef0=0)
    importanceModel.fit(X_train, Y_train)

    P_train = importanceModel.decision_function(X_train)
    P_test = importanceModel.decision_function(X_test)

    try:
        importances_logReg = estimation_methods.log_regression(Y_train, Y_test)
        LR_err = CVError * importances_logReg
    except Exception:
        print("Error LR")
        LR_err = []
        traceback.print_exc()

    try:
        importances_MLogReg = estimation_methods.log_regression_model_classification(P_train, P_test)
        PLR_err = CVError * importances_MLogReg
    except Exception:
        print("Error PLR")
        PLR_err = []
        traceback.print_exc()

    try:
        importances_KMM = estimation_methods.kmm(Y_train, Y_test)
        KMM_err = CVError * importances_KMM
    except Exception:
        print("Error KMM")
        KMM_err = []
        traceback.print_exc()

    try:
        importances_MKMM = estimation_methods.kmm_model_classification(P_train, P_test)
        PKMM_err = CVError * importances_MKMM
    except Exception:
        print("Error PKMM")
        PKMM_err = []
        traceback.print_exc()

    try:
        importances_kernel_density = estimation_methods.kernel_density(Y_train, Y_test, bandwith, kernel)
        KDE_err = CVError * importances_kernel_density
    except Exception:
        print("Error KDE")
        KDE_err = []
        traceback.print_exc()

    try:
        importances_kernel_density_model = estimation_methods.kernel_density_model(P_train, P_test, bandwith, kernel)
        PKDE_err = CVError * importances_kernel_density_model
    except Exception:
        print("Error PKDE")
        PKDE_err = []
        traceback.print_exc()

    try:
        importances_KLIEP = estimation_methods.KLIEP(Y_train, Y_test)
        KLIEP_err = CVError * importances_KLIEP
    except Exception:
        print("Error KLIEP")
        KLIEP_err = []
        traceback.print_exc()

    try:
        importances_KLIEP_model = estimation_methods.KLIEP_model(P_train, P_test)
        PKLIEP_err = CVError * importances_KLIEP_model
    except Exception:
        print("Error PKLIEP")
        PKLIEP_err = []
        traceback.print_exc()

    return [np.average(eval), np.average(CVError),
            np.average(LR_err), np.average(PLR_err),
            np.average(KMM_err), np.average(PKMM_err),
            np.average(KDE_err), np.average(PKDE_err),
            np.average(KLIEP_err), np.average(PKLIEP_err)]

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
    results = classification(X_train, Y_train, X_test, Y_test, seed)
    IO.writeToCSV(results, dataset_name, execution_id, seed)

if __name__== "__main__":
  main()