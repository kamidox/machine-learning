import scipy as sp
import matplotlib.pyplot as plt


def plot_models(x, y, fname):
	""" Display data in plot """

	plt.clf()
	plt.scatter(x, y)
	plt.title('Web traffic over last month')
	plt.xlabel('Time')
	plt.ylabel('Hits/Hour')
	plt.xticks([w * 7 * 24 for w in range(10)], ['Week %i' % w for w in range(10)])

	plt.autoscale(tight=True)
	plt.ylim(ymin=0)
	plt.grid()
	plt.savefig(fname)
	print('Saved models image to %s.' % fname)


def load_samples(fname):
	""" Load training sample dataset """

	data = sp.genfromtxt(fname, delimiter='\t')
	x = data[:, 0]
	y = data[:, 1]

	print('Totally %i entries while %i invalid entries.' % (sp.shape(data)[0], sp.sum(sp.isnan(y))))
	x = x[~sp.isnan(y)]
	y = y[~sp.isnan(y)]
	return (x, y)

x, y = load_samples('web_traffic.tsv')
plot_models(x, y, "../web_traffic_01.png")

