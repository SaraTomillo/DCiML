#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:34:02 2019

@author: quevedo
"""

import scipy.stats as ss
import numpy as np


class FriedmanNemenyi:
    """
    Calculates a statistical stes Friedman and the the rank test Nemenyi

     Params:
            minimize : If the best value is the smallest then = True
    """

    def _calculateRanks(self, ET):
        ranks = []
        for E in ET:
            irank = np.argsort(np.argsort(self.order * np.array(E)))  # en lugar de np.argsort(E)
            rank = np.zeros(len(irank))
            for i in range(len(E)):
                count = []
                index = []
                for j in range(len(E)):
                    if (E[j] == E[i]):
                        count.append(irank[j])
                        index.append(j)
                        # for i in index:
                        # rank[irank[i]] = np.mean(count)
                rank[i] = np.around(np.mean(count), decimals=self.decimals)

            # for i in range(len(irank)):
            #    rank[irank[i]]=i

            ranks.append(list(rank + 1))
        return ranks

    def __init__(self, Evals, order=1, decimals=4):
        """
        Params:
            Evals : Matrix where each row is a problem and each column a system.
                    Evals[p][s] is the measure of the evaluation of the problem
                     p using the system s
            order : 1 if minimizing, -1 if max
            decimals: the number of decimals after rounding

        produces: [FrTest,AvgRanks,CD_01,CD_05,CD_10]
            FrTest  : Boolean value if reject the test. If true the evaluations
                       are not equals.
            AvgRanks: Averege rank of each system
            CD_*    : Critical Distance for alpha.
        """

        self.decimals = decimals
        self.order = order

        # Friedman
        ET = np.transpose(Evals)
        Ins = "ss.friedmanchisquare(";
        for i in range(len(ET)):
            if i > 0:
                Ins = Ins + ","
            Ins = Ins + "ET[{}]".format(i)
        Ins = Ins + ")"
        #print(Ins)
        [f, p] = eval(Ins)
        self.f = f
        self.p = p

        # Nemenyi
        self.ranks = self._calculateRanks(Evals)
        #print(Evals)
        #print(self.ranks)
        self._nemenyi(self.ranks)

    def _calculateCD(self, q_alpha, NALG, NDS):
        return q_alpha[NALG - 1] * np.sqrt(NALG * (NALG + 1) / (6 * NDS))

    def _nemenyi(self, ranks):
        # para alpha 0.01
        q_alpha_01 = [
            0.000, 2.576, 2.913, 3.113, 3.255, 3.364, 3.452, 3.526, 3.590, 3.646, \
            3.696, 3.741, 3.781, 3.818, 3.853, 3.884, 3.914, 3.941, 3.967, 3.992, \
            4.015, 4.037, 4.057, 4.077, 4.096, 4.114, 4.132, 4.148, 4.164, 4.179]

        # para alpha 0.05
        q_alpha_05 = [
            0.000, 1.960, 2.344, 2.569, 2.728, 2.850, 2.948, 3.031, 3.102, 3.164, \
            3.219, 3.268, 3.313, 3.354, 3.391, 3.426, 3.458, 3.489, 3.517, 3.544, \
            3.569, 3.593, 3.616, 3.637, 3.658, 3.678, 3.696, 3.714, 3.732, 3.749]

        # para alpha 0.10
        q_alpha_10 = [
            0.000, 1.645, 2.052, 2.291, 2.460, 2.589, 2.693, 2.780, 2.855, 2.920, \
            2.978, 3.030, 3.077, 3.120, 3.159, 3.196, 3.230, 3.261, 3.291, 3.319, \
            3.346, 3.371, 3.394, 3.417, 3.439, 3.459, 3.479, 3.498, 3.516, 3.533]

        NDS = len(ranks)
        NALG = len(ranks[0])

        if NALG > len(q_alpha_05):
            print("**** too much systems to test ****")
            raise Exception("nemenyi: too much systems to test")

        self.CD_01 = np.around(self._calculateCD(q_alpha_01, NALG, NDS), decimals=self.decimals)
        self.CD_05 = np.around(self._calculateCD(q_alpha_05, NALG, NDS), decimals=self.decimals)
        self.CD_10 = np.around(self._calculateCD(q_alpha_10, NALG, NDS), decimals=self.decimals)

        # return [self.CD_01,self.CD_05,self.CD_10]

    def getF(self):
        return np.around(self.f, decimals=self.decimals)

    def getP(self):
        return np.around(self.p, decimals=self.decimals)

    def getRanks(self):
        return self.ranks

    def getAvgRanks(self):
        return np.around(np.mean(self.ranks, axis=0), decimals=self.decimals)

    def getDeviationRanks(self):
        return np.around(np.std(self.ranks, axis=0), decimals=self.decimals)


    def getCDs(self):
        return [self.CD_01, self.CD_05, self.CD_10]

    def getLatex(self, Evals, headers, DatasetNames):
        ranks = self.getRanks()
        avgrank = self.getAvgRanks()
        # Inicio tabla
        table = '\\begin{table}[]\n'
        L = 'l'
        n = len(Evals)
        m = len(Evals[0])
        for i in range(m):
            L = L + 'l'
        table += '\\begin{tabular}{' + L + '}\n'
        table +='\hline\n'
        # Cabecera
        head = "dataset"
        for i in range(m):
            head = head + '&' + headers[i]
        head = head + '\\\ ' + '\hline\n'
        table+= head
        # Filas
        for i in range(n):  # para cada fila
            row = DatasetNames[i]
            for j in range(m):  # para cada columna
                row = str(row) + '&' + str(Evals[i][j]) + '\ \ ' + str(ranks[i][j])
            row = row + '\\\ \n'
            table+= row
        # Media rankings
        table+='\hline\n'
        avg = 'Avg. Rank'
        for j in range(m):
            avg = str(avg) + '&' + '\ \ \  \ ' + str(avgrank[j])
        avg = avg + '\\\ '
        table+= avg + '\\\ ' + '\hline\n'
        table +='\\end{tabular}\n'
        table +='\\end{table}\n'
        return table


"""
Evals = [[1, 2, 3, 5], \
         [2, 4, 6, 6], \
         [3, 3, 1, 3], \
         [2, 4, 4, 5], \
         [1, 3, 4, 5], \
         [1, 3, 3, 5]]
FN = FriedmanNemenyi(Evals, order=1, decimals=4)
print("f={} p={}".format(FN.getF(), FN.getP()))
print("Ranks=")
print(FN.getRanks())
print(FN.getAvgRanks())
print("CD")
print(FN.getCDs())

FN.getLatex(Evals, FN.getRanks(), ["eval", "CV", "kliep", "mod"], ["d1", "d2", "d3", "d4", "d5", "d6"],
            FN.getAvgRanks())
"""


