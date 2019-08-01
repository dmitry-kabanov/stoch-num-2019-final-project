import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


OUTDIR = '_output'

p = argparse.ArgumentParser()
p.add_argument('--filename', '-f', default='heateq-cross_validation-clean.txt')
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

# Cross-validation parameters: parameter of interest and the number of folds.
gamma_values = np.logspace(-5, 0.7, num=21)
K = 5

gamma_index = data[:, 0]
fold_index = data[:, 1]
error_MSE = data[:, 2]
error_MSE_rel = data[:, 3]
error_lambda = data[:, 4]

lambda_values = data[:, 5]
lambda_values = lambda_values.reshape((-1, K))

qoi = error_MSE_rel
qoi = qoi.reshape((-1, K))
qoi_mean = qoi.mean(axis=1)
qoi_se = qoi.std(axis=1) / np.sqrt(K)
qoi_ci = np.vstack((qoi_mean-qoi_se, qoi_mean+qoi_se))

i = np.argmin(qoi_mean)
values = (qoi_mean[i], gamma_values[i])
print('Minimal mean prediction error is %.2e at gamma=%.2e' % (values))

fig, axes = plt.subplots(2, 1, figsize=(4.00, 2.47), sharex=True)
ax = axes[0]
ax.set_xscale('log')
ax.set_yscale('log')
for k in range(K):
    ax.plot(gamma_values, qoi[:, k], 's', label='Fold %d' % (k+1))
ax.plot(gamma_values, qoi_mean, label='Mean')
smb_1 = r'u_{\mathrm{pred}}'
smb_2 = r'u_{\mathrm{true}}'
ylabel = r'$\frac{\Vert %s - %s \Vert}{\Vert %s\Vert}$' % (smb_1, smb_2, smb_2)
ax.set_ylabel(ylabel)
if WITH_NOISE:
    ax.set_ylim((1e-2, None))
else:
    ax.set_ylim((1e-3, None))

ax = axes[1]
for k in range(K):
    ax.loglog(gamma_values, lambda_values[:, k], 'o', label='Fold %d' % (k+1))
ax.loglog(gamma_values, lambda_values.mean(axis=1), label='Mean')
if WITH_NOISE:
    ax.set_ylim((None, 10))
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\widehat\lambda$')

ax.legend(loc='best', ncol=2)

plt.tight_layout(pad=0.1, h_pad=1)

if args.save:
    if WITH_NOISE:
        suffix = 'noise'
    else:
        suffix = 'clean'
    figname = 'heateq-cross_validation-' + suffix + '.pdf'
    figname = os.path.join(OUTDIR, figname)
    fig.savefig(figname)
else:
    plt.show()
