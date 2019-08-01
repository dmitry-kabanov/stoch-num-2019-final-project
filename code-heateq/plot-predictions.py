import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from scipy import interpolate

from data import get_all_data, get_data
from neuralnetwork import PhysicsInformedNN


OUTDIR = '_output'


np.random.seed(1234)
tf.set_random_seed(1234)

p = argparse.ArgumentParser()
p.add_argument('--with-noise', action='store_true')
p.add_argument('--save', '-s', action='store_true')

args = p.parse_args()

if args.with_noise:
    WITH_NOISE = True
else:
    WITH_NOISE = False

if WITH_NOISE:
    GAMMA = 2.6
else:
    GAMMA = 1.36e-2
print('Plotting predictions with gamma = %.2e' % GAMMA)

N = 6400
LAYERS = [2, 20, 20, 1]
X_train, u_train, lb, ub = get_data(N, WITH_NOISE)
X_star, u_star, x, t, Exact, X, T = get_all_data(WITH_NOISE)

model = PhysicsInformedNN(X_train, u_train, LAYERS, lb, ub, GAMMA)
model.train(0)
u_pred, f_pred = model.predict(X_star)
error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
U_pred = interpolate.griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
lambda_value = model.get_pde_params()[0]
error_lambda_ = np.abs(lambda_value - 1.0)*100

print('Error u: %e' % (error_u))
print('Error l1: %.2f%%' % (error_lambda_))

# -----------------------------------------------------------------------------
# Plot predictions.
fig, axes = plt.subplots(1, 2, figsize=(4, 2.47), sharey=True)
x_min, x_max = x.min()-0.1, x.max()+0.1

ax = axes[0]
i = 25
ax.plot(x,Exact[i,:], '-', linewidth = 2, label = 'Exact')
ax.plot(x,U_pred[i,:], '--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
ax.set_title('$t = %.2f$' % t[i], fontsize = 10)
ax.axis('square')
ax.set_xlim((x_min, x_max))
ax.set_ylim([-1.1,1.1])

ax = axes[1]
i = 75
ax.plot(x,Exact[i,:], '-', linewidth = 2, label = 'Exact')
ax.plot(x,U_pred[i,:], '--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
ax.axis('square')
ax.set_xlim((x_min, x_max))
ax.set_ylim([-1.1, 1.1])
ax.set_title('$t = %.2f$' % t[i], fontsize = 10)
ax.legend(loc='best')

fig.tight_layout(pad=0.1)

if args.save:
    if WITH_NOISE:
        suffix = 'noise'
    else:
        suffix = 'clean'
    figname = 'heateq-predictions-' + suffix + '.pdf'
    figname = os.path.join(OUTDIR, figname)
    fig.savefig(figname)
else:
    plt.show()
