# -*- coding: utf-8 -*-
import scipy as sp
import matplotlib.pyplot as pl


def plot_data(x, y):
    pl.clf()

    pl.xlabel('Exam 1 score')
    pl.ylabel('Exam 2 score')
    pl.legend(['Admitted', 'Not admitted'])
