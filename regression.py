import argparse
import traceback

import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR

import estimation_methods
import IO


def error(predictions, y):
    return abs(predictions - y)

bandwith = 1
kernel = 'epanechnikov'

def regression(X_train, Y_train, X_test, Y_test):
    model = SVR(kernel='linear', C=1, coef0=0)

    model.fit(X_train, Y_train)

    Pred_Train = model.predict(X_train)
    Pred_Eval = model.predict(X_test)

    eval = error(Pred_Eval, Y_test)
    CV = cross_val_predict(model, X_train, Y_train, cv=10)
    CVError = error(CV, Y_train)

    importanceModel = SVR(kernel='linear', C=1, coef0=0)
    importanceModel.fit(X_train, Y_train)

    P_train = importanceModel.predict(X_train)
    P_test = importanceModel.predict(X_test)

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

    #return [eval, CVError, kliep_err, kliep_model_err, kernel_density_err, kernel_density_model_err]
    return [np.average(eval), np.average(CVError), np.average(kliep_err), np.average(kliep_model_err), np.average(kernel_density_err), np.average(kernel_density_model_err)]

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
    results = regression(X_train, Y_train, X_test, Y_test)
    IO.writeToCSV(results, dataset_name, execution_id,seed)


if __name__== "__main__":
  main()