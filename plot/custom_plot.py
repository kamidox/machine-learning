# -*- coding: utf-8 -*-
import pylab as pl
import numpy as np


def plot(fname='../custom_plot.png', dpi=200):
    # custom figure size in inches and dpi
    pl.figure(figsize=(8, 6), dpi=dpi)

    # create subplot from a grid of 1x1
    # pl.subplot(1, 1, 1)

    x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    c, s = np.cos(x), np.sin(x)

    # plot cosine
    pl.plot(x, c, color='blue', linewidth=2.0, linestyle='-', label='cosine')
    pl.plot(x, s, color='red', linewidth=2.0, linestyle='-', label='sine')

    pl.legend(loc='upper left')

    # annotate
    t = 2 * np.pi / 3
    pl.plot([t, t], [0, np.sin(t)], color='red', linewidth=2.0, linestyle='--')
    pl.scatter([t, ], [np.sin(t), ], 50, color='red')
    pl.annotate(r'$sin{\left(\frac{2\pi}{3}\right)} = \frac{\sqrt{3}}{2}$',
                xy=(t, np.sin(t)), xycoords='data',
                xytext=(10, 30), textcoords='offset points', fontsize=16,
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=.2"))

    pl.plot([t, t], [0, np.cos(t)], color='blue', linewidth=2.0, linestyle='--')
    pl.scatter([t, ], [np.cos(t), ], 50, color='blue')
    pl.annotate(r'$cos{\left(\frac{2\pi}{3}\right)} = \frac{1}{2}$',
                xy=(t, np.cos(t)), xycoords='data',
                xytext=(-90, -50), textcoords='offset points', fontsize=16,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    # custom spines
    ax = pl.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    # ax.spines['left'].set_position(('axes', 0.5))
    # ax.spines['bottom'].set_position(('axes', 0.5))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(16)
        label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65))

    pl.xlim(x.min() * 1.1, x.max() * 1.1)
    pl.ylim(c.min() * 1.1, c.max() * 1.1)

    pl.xticks(np.linspace(-np.pi, np.pi, 5, endpoint=True),
              [r'$-\pi$', r'$-\pi/2$', r'$0$',
              r'$\pi/2$', r'$\pi$'])
    pl.yticks(np.linspace(-1.0, 1.0, 5, endpoint=True))

    pl.savefig('../custom_plot.png', dpi=dpi)
