import argparse
import traceback
from random import Random

import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

import estimation_methods
import IO

import tracemalloc
from utils import minmax, reduce, display_top, npShuffle


def error(predictions, y):
    return np.sign(np.abs(predictions-y))


bandwith = 1
kernel = 'epanechnikov'

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
        importances_logReg = estimation_methods.log_regression(X_train, X_test)
        logReg_err = CVError * importances_logReg
    except Exception:
        print("Fallo LogReg")
        logReg_err = []
        traceback.print_exc()

    try:
        importances_MLogReg = estimation_methods.log_regression_model(P_train, P_test)
        MLogReg_err = CVError * importances_MLogReg
    except Exception:
        print("Fallo MLogReg")
        MLogReg_err = []
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
        MKMM_err = CVError * importances_MKMM
    except Exception:
        print("Fallo MKMM")
        MKMM_err = []
        traceback.print_exc()

    try:
        importances_kernel_density = estimation_methods.kernel_density(X_train, X_test, bandwith, kernel)
        kernel_density_err = CVError * importances_kernel_density
    except Exception:
        print("Fallo kernel density")
        kernel_density_err = []
        traceback.print_exc()

    try:
        importances_kernel_density_model = estimation_methods.kernel_density_model(P_train, P_test, bandwith, kernel)
        kernel_density_model_err = CVError * importances_kernel_density_model
    except Exception:
        print("Fallo kernel density con modelo")
        kernel_density_model_err = []
        traceback.print_exc()

    try:
        importances_kliep = estimation_methods.kliep(X_train, X_test)
        kliep_err = CVError * importances_kliep
    except Exception:
        print("Fallo kliep")
        kliep_err = []
        traceback.print_exc()

    try:
        importances_kliep_model = estimation_methods.kliep_model(P_train, P_test)
        kliep_model_err = CVError * importances_kliep_model
    except Exception:
        print("Fallo kliep con modelo")
        kliep_model_err = []
        traceback.print_exc()

    logReg_err = []
    KMM_err = []
    kernel_density_err = []
    kliep_err = []
    return [np.average(eval), np.average(CVError),
            np.average(logReg_err), np.average(MLogReg_err),
            np.average(KMM_err), np.average(MKMM_err),
            np.average(kernel_density_err), np.average(kernel_density_model_err),
            np.average(kliep_err), np.average(kliep_model_err)]

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