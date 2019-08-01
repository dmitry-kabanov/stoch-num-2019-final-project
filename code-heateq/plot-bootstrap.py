import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats


OUTDIR = '_output'


def detect_outliers_with_iqr(y):
    """Classify data points as normal and outliers using Tukey's IQR method."""
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    lb, ub = q1 - 3*iqr, q3 + 3*iqr

    normal_idx = np.where((y >= lb) & (y <= ub))[0]
    outliers_idx = np.where((y < lb) | (y > ub))[0]
    assert len(normal_idx) + len(outliers_idx) == len(y)

    return normal_idx, outliers_idx


p = argparse.ArgumentParser()
p.add_argument('--filename', '-f', default='heateq-bootstrap.txt')
p.add_argument('--save', '-s', action='store_true')

args = p.parse_args()

if 'noise' in args.filename:
    WITH_NOISE = True
else:
    WITH_NOISE = False

if not args.filename.startswith(OUTDIR):
    args.filename = os.path.join(OUTDIR, args.filename)

print('Reading data from file %s' % args.filename)

data = np.loadtxt(args.filename)

lambda_estim = data[-1]
print('Lambda from full sample: %.6f' % lambda_estim)
data = data[:-1]

# Cleaning the data.
normal_idx, outliers_idx = detect_outliers_with_iqr(data)
data = data[normal_idx]
x_lb = data.mean() - 4*data.std()
x_ub = data.mean() + 4*data.std()
# 95% confidence interval.
ci = tuple(np.percentile(data, [2.5, 97.5]))

print('Number of outliers: %d' % (len(outliers_idx)))
print('Lambda mean from clean data: %.6f' % data.mean())
print('Lambda boostrap percentile intervals: (%.6f, %.6f)' % ci)

kde = stats.gaussian_kde(data)
lambda_range = np.linspace(x_lb, x_ub, num=1001)

# plt.plot(range(len(data)), data, 'bo-')
# plt.plot(outliers_idx, data[outliers_idx], 'ro-')
# plt.show()

fig, axes = plt.subplots(1, 1, figsize=(4, 2.47))

ax = axes
vert_line_y = [0.0, kde(lambda_range).max()]
hist = ax.hist(data, bins='auto', density=True)
ax.plot(lambda_range, kde(lambda_range))
ax.plot([data.mean(), data.mean()], vert_line_y, '--')
ax.plot([ci[0], ci[0]], vert_line_y, 'k--')
ax.plot([ci[1], ci[1]], vert_line_y, 'k--')
ax.set_xlim((x_lb, x_ub))
ax.set_xlabel(r'$\widehat\lambda$')
ax.set_ylabel('Probability density')

#ax = axes[1]
#ax.plot(data, '-')
#ax.set_ylim(0.99, 2)

fig.tight_layout(pad=0.1)

if args.save:
    if WITH_NOISE:
        suffix = 'noise'
    else:
        suffix = 'clean'
    figname = 'heateq-bootstrap-' + suffix + '.pdf'
    figname = os.path.join(OUTDIR, figname)
    fig.savefig(figname)
else:
    plt.show()
