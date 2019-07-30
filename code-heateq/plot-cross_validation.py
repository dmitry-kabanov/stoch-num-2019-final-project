import argparse
import numpy as np
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--filename', '-f', default='heateq-cross_validation.txt')
p.add_argument('--save', '-s', action='store_true')

args = p.parse_args()

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
qoi_min = qoi.min(axis=1)
qoi_max = qoi.max(axis=1)
errorbar = np.vstack((qoi_min, qoi_max))

fig, axes = plt.subplots(2, 1, figsize=(4.00, 3), sharex=True)
ax = axes[0]
ax.set_xscale('log')
ax.set_yscale('log')
for k in range(K):
    ax.plot(gamma_values, qoi[:, k], 's', label='Fold %d' % (k+1))
ax.plot(gamma_values, qoi_mean, label='Mean')
smb_1 = r'u_{\mathrm{pred}}'
smb_2 = r'u_{\mathrm{true}}'
ylabel = r'$\frac{%s - %s}{%s}$' % (smb_1, smb_2, smb_2)
ax.set_ylabel(ylabel)

ax = axes[1]
for k in range(K):
    ax.loglog(gamma_values, lambda_values[:, k], 'o', label='Fold %d' % (k+1))
ax.loglog(gamma_values, lambda_values.mean(axis=1), label='Mean')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\widehat\lambda$')

ax.legend(bbox_to_anchor=(1.0, -0.5), ncol=3)

plt.tight_layout(pad=0.1)

if args.save:
    figname = 'heateq-cross-validation'
    if 'with-noise' in args.filename:
        figname = figname + '-with-noise'
    figname = figname + '.pdf'
    fig.savefig(figname)
else:
    plt.show()
