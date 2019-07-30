import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

p = argparse.ArgumentParser()
p.add_argument('--filename', '-f', default='heateq-bootstrap.txt')
p.add_argument('--save', '-s', action='store_true')

args = p.parse_args()

if not args.filename.startswith('data'):
    args.filename = os.path.join('data', args.filename)

print('Reading data from file %s' % args.filename)

data = np.loadtxt(args.filename)

lambda_estim = data[-1]
print(lambda_estim)
data = data[:-1]

#data[data > 10] = float('nan')

kde = stats.gaussian_kde(data)
lambda_range = np.linspace(data.min(), data.max(), num=1001)

fig, axes = plt.subplots(1, 2, figsize=(4, 2.47))

ax = axes[0]
hist = ax.hist(data, bins=10, density=True, range=(0, 2))
#ax.plot(kde(lambda_range))
ax.plot([lambda_estim, lambda_estim], [0.0, hist[0].max()], '--')
ax.set_xlim((0.95, 1.05))

ax = axes[1]
ax.plot(data)
ax.set_ylim(0.99, 2)

if args.save:
    figname = 'heateq-boostrap'
    if 'with-noise' in args.filename:
        figname = figname + '-with-noise'
    figname = figname + '.pdf'
    fig.savefig(figname)
else:
    plt.show()
