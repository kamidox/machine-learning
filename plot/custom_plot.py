# -*- coding: utf-8 -*-
import pylab as pl
import numpy as np


def plot_basic(fname='../custom_plot.png', dpi=200):
    # custom figure size in inches and dpi
    pl.clf()
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

    pl.savefig(fname, dpi=dpi)


def plot_regular(fname='../regular_plot.png', dpi=200):
    pl.clf()
    n = 265
    x = np.linspace(-np.pi, np.pi, n, endpoint=True)
    y = np.sin(2 * x)
    pl.plot(x, y + 1, color='blue')
    pl.plot(x, y - 1, color='blue')
    pl.fill_between(x, 1, y + 1, color='blue', alpha=0.2)
    pl.fill_between(x, -1, y - 1, where=(y - 1) > -1, color='blue', alpha=0.2)
    pl.fill_between(x, -1, y - 1, where=(y - 1) < -1, color='red', alpha=0.2)

    pl.savefig(fname, dpi=dpi)


def plot_scatter(fname='../scatter_plot.png', dpi=200):
    pl.clf()
    n = 1024
    x = np.random.normal(0, 1, n)
    y = np.random.normal(0, 1, n)
    c = np.arctan(y / x)
    pl.scatter(x, y, s=30, c=c, alpha=0.5)

    pl.xticks(())
    pl.yticks(())
    pl.xlim(-1.8, 1.8)
    pl.ylim(-1.8, 1.8)

    pl.savefig(fname, dpi=dpi)


def plot_bar(fname='../bar_plot.png', dpi=200):
    pl.clf()
    n = 12
    x = np.arange(n)
    y1 = (1 - x / float(n)) * np.random.uniform(0.5, 1.0, n)
    y2 = (1 - x / float(n)) * np.random.uniform(0.5, 1.0, n)

    pl.bar(x, y1, facecolor='#9999ff', edgecolor='white')
    pl.bar(x, -y2, facecolor='#ff9999', edgecolor='white')

    for i, j in zip(x, y1):
        pl.text(i + 0.4, j + 0.05, '%.2f' % j, ha='center', va='bottom', fontsize=9)

    for i, j in zip(x, -y2):
        pl.text(i + 0.4, j - 0.05, '%.2f' % j, ha='center', va='top', fontsize=9)

    pl.xlim(-1, 13)
    pl.ylim(-1.25, 1.25)
    pl.xticks(())
    pl.yticks(())
    pl.savefig(fname, dpi=dpi)


def plot_contour(fname='../contour_plot.png', dpi=200):
    def f(x, y):
        return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

    pl.clf()
    n = 256
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    mx, my = np.meshgrid(x, y)

    pl.contourf(mx, my, f(mx, my), 8, alpha=0.75, cmap=pl.cm.hot)
    c = pl.contour(mx, my, f(mx, my), 8, colors='black', linewidth=0.5)
    pl.clabel(c, fontsize=9)
    pl.savefig(fname, dpi=dpi)


def plot_imshow(fname='../imshow_plot.png', dpi=200):
    def f(x, y):
        return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

    pl.clf()
    n = 10
    x = np.linspace(-3, 3, 3.5 * n)
    y = np.linspace(-3, 3, 3 * n)
    X, Y = np.meshgrid(x, y)

    pl.axes([0.025, 0.025, 0.95, 0.95])
    pl.xticks(())
    pl.yticks(())

    pl.imshow(f(X, Y), interpolation='nearest', origin='lower', cmap=pl.cm.gray)
    pl.colorbar(shrink=0.92)
    pl.savefig(fname, dpi=dpi)


def plot_pie(fname='../pie_plot.png', dpi=200):
    pl.clf()
    n = 20
    z = np.ones(n)
    z[-1] *= 2

    pl.axes([0.1, 0.1, 0.8, 0.8])
    pl.xticks(())
    pl.yticks(())
    pl.axis('equal')
    pl.pie(z, explode=z * 0.05, colors=['%f' % (i / float(n)) for i in range(n)])
    pl.savefig(fname, dpi=dpi)


def plot_quiver(fname='../quiver_plot', dpi=200):
    pl.clf()
    n = 12
    X, Y = np.mgrid[0:n, 0:n]
    t = np.arctan2(Y - n / 2., X - n / 2.)
    radius = 10 + np.sqrt((Y - n / 2.0) ** 2 + (X - n / 2.0) ** 2)
    U, V = radius * np.cos(t), radius * np.sin(t)

    pl.axes([0.025, 0.025, 0.95, 0.95])
    pl.quiver(X, Y, U, V, radius, alpha=.5)
    pl.quiver(X, Y, U, V, edgecolor='red', facecolor='None', linewidth=.5)

    pl.xlim(-1, n)
    pl.xticks(())
    pl.ylim(-1, n)
    pl.yticks(())

    pl.savefig(fname, dpi=dpi)


def main():
    # plot_pie()
    plot_quiver()

if __name__ == '__main__':
    main()
