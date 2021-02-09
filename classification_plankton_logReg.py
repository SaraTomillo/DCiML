import argparse
import traceback

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

from utils import IO
import estimation_methods
from utils.methods import ensemble_KMM
import tracemalloc
from utils.utils import minmax, reduce, npShuffle


def error(predictions, y):
    return np.sign(np.abs(predictions-y))


bandwith = 1
kernel = 'linear'

def classification(X_train, Y_train, X_test, Y_test, seed):
    random = np.random.RandomState(int(seed))
    tracemalloc.start()
    model = SVC(kernel='linear', C=1)

    minmax_X = minmax(X_train)
    X_train = reduce(X_train, minmax_X)
    X_test = reduce(X_test, minmax_X)

    model.fit(X_train, Y_train)
    Pred_Eval = model.predict(X_test)

    eval_err = error(Pred_Eval, Y_test)

    X_train=npShuffle(X_train,seed)
    Y_train=npShuffle(Y_train,seed)

    CV = cross_val_predict(model, X_train, Y_train, cv=10)#, method='decision_function')
    CV_err = error(CV, Y_train)

    importanceModel = LogisticRegression(random_state=random, C=1, class_weight='balanced')
    importanceModel.fit(X_train, Y_train)

    P_train = importanceModel.decision_function(X_train)
    P_test = importanceModel.decision_function(X_test)

    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    train = []
    for row in range(len(X_train)):
        aux = []
        for element in X_train[row]:
            aux.append(element)
        aux.append(P_train[row])
        train.append(aux)

    test = []
    for row in range(len(X_test)):
        aux = []
        for element in X_test[row]:
            aux.append(element)
        aux.append(P_test[row])
        test.append(aux)

    train = Y_train.reshape(-1, 1)
    test = Y_test.reshape(-1, 1)


    try:
        importances_LR_C = estimation_methods.LR(X_train, X_test, random)
        LR_C_err = CV_err * importances_LR_C
    except Exception:
        print("Error LR-C")
        LR_C_err = []
        traceback.print_exc()

    try:
        importances_LR_P = estimation_methods.LR_P_classification(P_train, P_test, random)
        LR_P_err = CV_err * importances_LR_P
    except Exception:
        print("Error LR-P")
        LR_P_err = []
        traceback.print_exc()
    if len(LR_P_err) == 0:
        try:
            importances_LR_P = estimation_methods.LR_P(P_train, P_test, random)
            LR_P_err = CV_err * importances_LR_P
        except Exception:
            print("Error LR-CP")
            LR_P_err = []
            traceback.print_exc()

    try:
        importances_LR_CP = estimation_methods.LR(train, test, random)
        LR_CP_err = CV_err * importances_LR_CP
    except Exception:
        print("Error LR-CP")
        LR_CP_err = []
        traceback.print_exc()

    try:
        importances_KMM_C = ensemble_KMM.EKMM_train(X_train, X_test)
        KMM_C_err = CV_err * importances_KMM_C
    except Exception:
        print("Error KMM-C")
        KMM_C_err = []
        traceback.print_exc()

    try:
        importances_KMM_P = ensemble_KMM.EKMM_train(P_train, P_test)
        KMM_P_err = CV_err * importances_KMM_P
    except Exception:
        print("Error KMM-P")
        KMM_P_err = []
        traceback.print_exc()
    if len(KMM_P_err) == 0:
        try:
            importances_KMM_P = ensemble_KMM.EKMM_train(P_train, P_test)
            KMM_P_err = CV_err * importances_KMM_P
        except Exception:
            print("Error KMM-P")
            KMM_P_err = []
            traceback.print_exc()
    try:
        importances_KMM_CP = ensemble_KMM.EKMM_train(train, test)
        KMM_CP_err = CV_err * importances_KMM_CP
    except Exception:
        print("Error KMM-CP")
        KMM_CP_err = []
        traceback.print_exc()


    try:
        importances_KDE_C = estimation_methods.KDE(X_train, X_test, bandwith, kernel)
        KDE_C_err = CV_err * importances_KDE_C
    except Exception:
        print("Error KDE-C")
        KDE_C_err = []
        traceback.print_exc()

    try:
        importances_KDE_P = estimation_methods.KDE_P(P_train, P_test, bandwith, kernel)
        KDE_P_err = CV_err * importances_KDE_P
    except Exception:
        print("Error KDE-P")
        KDE_P_err = []
        traceback.print_exc()

    try:
        importances_KDE_CP = estimation_methods.KDE(train, test, bandwith, kernel)
        KDE_CP_err = CV_err * importances_KDE_CP
    except Exception:
        print("Error KDE-CP")
        KDE_CP_err = []
        traceback.print_exc()

    return [np.average(eval), np.average(CV_err),
            np.average(LR_C_err), np.average(LR_P_err), np.average(LR_CP_err),
            np.average(KMM_C_err), np.average(KMM_P_err), np.average(KMM_CP_err),
            np.average(KDE_C_err), np.average(KDE_P_err), np.average(KDE_CP_err)]

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
    IO.writeToCSV(results, "plankton-LR", dataset_name, execution_id, seed)

if __name__== "__main__":
  main()