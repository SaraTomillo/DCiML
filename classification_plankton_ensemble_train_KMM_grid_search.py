import argparse
import traceback

import numpy as np
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.svm import SVC

from utils import IO
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

    svc = SVC()
    params = {
        'C': [10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3)],
        'kernel': ['linear'],
        'class_weight': ['balanced'],
        'coef0': [0]
    }
    grid_SVC = GridSearchCV(estimator=svc,
                           param_grid=params,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=None)
    grid_SVC.fit(X_train, Y_train)
    importanceModel = grid_SVC.best_estimator_

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
        importances_KMM = ensemble_KMM.ensemble_KMM_train(X_train, X_test)
        KMM_err = CVError * importances_KMM
    except Exception:
        print("Error KMM")
        KMM_err = []
        traceback.print_exc()

    try:
        importances_PKMM = ensemble_KMM.ensemble_KMM_train(P_train, P_test)
        PKMM_err = CVError * importances_PKMM
    except Exception:
        print("Error PKMM")
        PKMM_err = []
        traceback.print_exc()
    if len(PKMM_err) == 0:
        try:
            importances_PKMM = ensemble_KMM.ensemble_KMM_train(P_train, P_test)
            PKMM_err = CVError * importances_PKMM
        except Exception:
            print("Error PKMM")
            PKMM_err = []
            traceback.print_exc()
    try:
        importances_BKMM = ensemble_KMM.ensemble_KMM_train(train, test)
        BKMM_err = CVError * importances_BKMM
    except Exception:
        print("Error BKMM")
        BKMM_err = []
        traceback.print_exc()

    return [np.average(eval), np.average(CVError),
            np.average(KMM_err), np.average(PKMM_err), np.average(BKMM_err)]


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
    IO.writeToCSV(results, "plankton", dataset_name, execution_id, seed)

if __name__== "__main__":
  main()