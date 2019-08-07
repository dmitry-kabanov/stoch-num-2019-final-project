#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from scipy import mean, stats
import pandas as pd
from scipy.io import loadmat
import os
import glob
import pandas as pd
os.chdir('output_images/')


# ## $\gamma$ cross validation

# In[7]:


data = pd.read_csv('burgers-cross_validation_[0.0,5.01].csv')
gammas = data['gamma'].unique()

error_MSE = []

for gamma in gammas:
    error_MSE.append(sum(data[data['gamma'] == gamma]['error_MSE_rel_table'])/5)
    
plt.style.use('bmh')
fig, subplots = plt.subplots(figsize = (8, 5), facecolor = 'w', edgecolor = 'k')
ax = fig.axes[0]

#fig.suptitle('cross validation $\gamma$', fontweight='normal', fontsize=14)
ax.semilogx(gammas, error_MSE, 'blue', linewidth=2., label = 'Error', marker='o')
ax.plot(gammas[np.where(error_MSE == np.min(error_MSE))]*np.ones(len(gammas)), 
        np.linspace(0., np.max(error_MSE), len(gammas)), color = 'k',
        linestyle='--', label='$\gamma$ min')

ax.grid(True)
ax.grid(color='k', linestyle='--', linewidth = 0.5)
ax.legend(fontsize=14, loc=2) 
#ax.set(xlabel = '$\gamma$', ylabel = "$\\frac{||U_{test} - U_{pred}||_2}{||U_{test}||_2}$")
plt.xlabel('$\gamma$', fontsize=14)
plt.ylabel("$\\frac{||U_{true} - U_{pred}||_2}{||U_{true}||_2}$", fontsize = 18)
plt.savefig('burgers-cross_val_gamma.png', format = 'png', transparent = True)

plt.show()


# In[8]:


gammas[error_MSE.index(min(error_MSE))]


# In[9]:


data = pd.read_csv('burgers-cross_validation_noisy_[0.0,5.01].csv')
gammas = data['gamma'].unique()

error_MSE = []

for gamma in gammas:
    error_MSE.append(sum(data[data['gamma'] == gamma]['error_MSE_rel_table'])/5)
    
plt.style.use('bmh')
fig, subplots = plt.subplots(figsize = (8, 5), facecolor = 'w', edgecolor = 'k')
ax = fig.axes[0]

#fig.suptitle('cross validation $\gamma_{noisy}$', fontweight='normal', fontsize=14)

ax.semilogx(gammas, error_MSE, 'blue', linewidth=2., label = 'Error', marker='o')
ax.plot(gammas[np.where(error_MSE == np.min(error_MSE))]*np.ones(len(gammas)), 
        np.linspace(0., np.max(error_MSE), len(gammas)), color = 'k',
        linestyle='--', label='$\gamma_{noisy}$ min')

ax.grid(True)
ax.grid(color='k', linestyle='--', linewidth = 0.5)
ax.legend(fontsize=16, loc=2) 
#ax.set(xlabel = '$\gamma$', ylabel = 'MSE')
plt.xlabel('$\gamma_{noisy}$', fontsize=16)
plt.ylabel("$\\frac{||U_{true} - U_{pred}||_2}{||U_{true}||_2}$", fontsize = 18)
plt.savefig('burgers-cross_val_gamma_noisy.png', format = 'png', transparent = True)

plt.show()


# In[10]:


gammas[error_MSE.index(min(error_MSE))]


# #### Bootstrap

# #### data without any noise

# In[11]:


def conf(array):
    return np.percentile(array, 2.5), np.percentile(array, 97.5)


# In[12]:


# concatenate all CSV files
#all_filenames = [i for i in glob.glob('bootstrap_noisy_*.{}'.format('csv'))]
#combine all files in the list
#combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
#combined_csv.to_csv( "bootstrap_noisy_all.csv", index=False, encoding='utf-8-sig')


# In[13]:


bootstrap_data = pd.read_csv('bootstrap_all.csv')
print(bootstrap_data.shape)
bootstrap_data.head()


# In[14]:


l1, l2 = np.array(bootstrap_data['l1']).flatten(), np.array(bootstrap_data['l2']).flatten()

l1 = np.array([np.float(x[1:-1]) for x in l1])
l2 = np.array([np.float(x[1:-1]) for x in l2])
l1.sort()
l1 = l1[3:]

l2.sort()
l2 = l2[3:]

l1_mean = np.mean(l1)*np.ones(len(l1))
l2_mean = np.mean(l2)*np.ones(len(l2))

l1_kernel = stats.gaussian_kde(l1)
l2_kernel = stats.gaussian_kde(l2)


# In[15]:


left, right = conf(l1)
plt.style.use('bmh')

fig, subplots = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 8), 
                                  facecolor = 'w', edgecolor = 'k')
#fig.suptitle('distribution of $\lambda_1$', fontweight='normal', fontsize=14)
ax = fig.axes[0]
ax.hist(l1, bins = 20, facecolor='orange', alpha=0.75, label = '$\lambda_1$')

x = np.linspace(l1[0]-0.01, l1[-1]+0.01, 340)
ax.plot(x, l1_kernel(x), label = 'KDE')

ax.plot(l1_mean, np.linspace(0., np.max(l1_kernel(x)), len(l1_mean)), 
        linestyle='--', label='$\lambda_1$ mean = {}'.format(np.round(mean(l1), 4)))
ax.set_xlim([l1[0]-0.01, l1[-1]+0.01])

ax.scatter(left, 0, s=300, facecolor='k' , marker='|', label='confidence left: {}'.format(np.round(left, 3)))
ax.scatter(right, 0, s=300, facecolor='k', marker='|', label='confidence right: {}'.format(np.round(right, 3)))

ax.grid(True)
ax.grid(color='k', linestyle='--', linewidth = 0.5)
ax.legend(fontsize=14, loc=2)  
#ax.set(xlabel='$\lambda_1$')
plt.xlabel('$\lambda_1$', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)

plt.savefig('burgers-bootstraped_l1.png', format = 'png', transparent = True)
plt.show()


# In[16]:


left, right = conf(l2)
plt.style.use('bmh')

fig, subplots = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 8), 
                                  facecolor = 'w', edgecolor = 'k')
#fig.suptitle('distribution of $\lambda_2$', fontweight='normal', fontsize=14)
ax = fig.axes[0]
ax.hist(l2, bins = 30, facecolor='steelblue', alpha=0.75, label = '$\lambda_2$')

x = np.linspace(l2[0]-0.0001, l2[-1]+0.0001, 340)
y = l2_kernel(x)/60
ax.plot(x, y, label = 'KDE')

ax.plot(l2_mean, np.linspace(0., np.max(l2_kernel(x)/60), len(l2_mean)), 
        linestyle='--', label='$\lambda_2$ mean = {}'.format(np.round(mean(l2), 4)))
ax.set_xlim([l2[0]-0.0001, l2[-1]+0.0001])

ax.scatter(left, 0, s=300, facecolor='k' , marker='|', label='confidence left: {}'.format(np.round(left, 3)))
ax.scatter(right, 0, s=300, facecolor='k', marker='|', label='confidence right: {}'.format(np.round(right, 3)))

ax.grid(True)
ax.grid(color='k', linestyle='--', linewidth = 0.5)
ax.legend(fontsize=14, loc=1)  
#ax.set(xlabel='$\lambda_2$')
plt.xlabel('$\lambda_2$', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)

plt.savefig('burgers-bootstraped_l2.png', format = 'png', transparent = True)
plt.show()


# In[17]:


mean(l2)


# #### noisy data

# In[18]:


bootstrap_data = pd.read_csv('bootstrap_noisy_all.csv')
print(bootstrap_data.shape)
bootstrap_data.head()


# In[19]:


l1, l2 = np.array(bootstrap_data['l1_noisy']).flatten(), np.array(bootstrap_data['l2_noisy']).flatten()

l1 = np.array([np.float(x[1:-1]) for x in l1])
l2 = np.array([np.float(x[1:-1]) for x in l2])
l1.sort()
l1 = l1[1:]

l2.sort()
l2 = l2[3:]

l1_mean = np.mean(l1)*np.ones(len(l1))
l2_mean = np.mean(l2)*np.ones(len(l2))

l1_kernel = stats.gaussian_kde(l1)
l2_kernel = stats.gaussian_kde(l2)


# In[30]:


left, right = conf(l1)
plt.style.use('bmh')

fig, subplots = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 8), 
                                  facecolor = 'w', edgecolor = 'k')
fig.suptitle('distribution of $\lambda_1$', fontweight='normal', fontsize=14)
ax = fig.axes[0]
ax.hist(l1, bins = 20, facecolor='orange', alpha=0.75, label = '$\lambda_1$')

x = np.linspace(l1[0]-0.01, l1[-1]+0.01, 340)
ax.plot(x, l1_kernel(x), label = 'KDE')

ax.plot(l1_mean, np.linspace(0., np.max(l1_kernel(x)), len(l1_mean)), 
        linestyle='--', label='$\lambda_1$ mean = {}'.format(np.round(mean(l1), 4)))
ax.set_xlim([l1[0]-0.01, l1[-1]+0.01])

ax.scatter(left, 0, s=300, facecolor='k' , marker='|', label='confidence left: {}'.format(np.round(left, 3)))
ax.scatter(right, 0, s=300, facecolor='k', marker='|', label='confidence right: {}'.format(np.round(right, 3)))

ax.grid(True)
ax.grid(color='k', linestyle='--', linewidth = 0.5)
ax.legend(fontsize=14, loc=2)  
#ax.set(xlabel='$\lambda_1$')
plt.xlabel('$\lambda_1$', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)

plt.savefig('burgers-bootstraped_l1_noisy.png', format = 'png', transparent = True)
plt.show()


# In[31]:


left, right = conf(l2)
plt.style.use('bmh')

fig, subplots = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 8), 
                                  facecolor = 'w', edgecolor = 'k')
fig.suptitle('distribution of $\lambda_2$', fontweight='normal', fontsize=14)
ax = fig.axes[0]
ax.hist(l2, bins = 30, facecolor='steelblue', alpha=0.75, label = '$\lambda_2$')

x = np.linspace(l2[0]-0.0001, l2[-1]+0.0001, 340)
y = l2_kernel(x)/60
ax.plot(x, y, label = 'KDE')

ax.plot(l2_mean, np.linspace(0., np.max(l2_kernel(x)/60), len(l2_mean)), 
        linestyle='--', label='$\lambda_2$ mean = {}'.format(np.round(mean(l2), 4)))
ax.set_xlim([l2[0]-0.0001, l2[-1]+0.0001])

ax.scatter(left, 0, s=300, facecolor='k' , marker='|', label='confidence left: {}'.format(np.round(left, 3)))
ax.scatter(right, 0, s=300, facecolor='k', marker='|', label='confidence right: {}'.format(np.round(right, 3)))

ax.grid(True)
ax.grid(color='k', linestyle='--', linewidth = 0.5)
ax.legend(fontsize=14, loc=1)  
#ax.set(xlabel='$\lambda_2$')
plt.xlabel('$\lambda_2$', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)

plt.savefig('burgers-bootstraped_l2_noisy.png', format = 'png', transparent = True)
plt.show()


# In[23]:


mean(l2)


# ### configuration vs error

# In[184]:


layers_array = [[2, 5, 1], [2, 5, 5, 1], [2, 5, 5, 5, 5, 1], [2, 5, 5, 5, 5, 5, 5, 1], 
                [2, 5, 5, 5, 5, 5, 5, 5, 5, 1],
                
                [2, 10, 1], [2, 10, 10, 1], [2, 10, 10, 10, 10, 1], [2, 10, 10, 10, 10, 10, 10, 1], 
                [2, 10, 10, 10, 10, 10, 10, 10, 10, 1],
                
                [2, 20, 1], [2, 20, 20, 1], [2, 20, 20, 20, 20, 1], [2, 20, 20, 20, 20, 20, 20, 1], 
                [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]]


# In[2]:


data = pd.read_csv('layers_neuron_error.csv')
data


# ### Exact solution plot

# In[ ]:


#######################################################################
############################# Plotting ###############################
######################################################################

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
#ax.axis('off')

####### Row 0: u(t,x) ##################
#gs0 = gridspec.GridSpec(1, 2)
#gs0.update(top=1-0.06, bottom=1-1.0/3.0+0.06, left=0.15, right=0.85, wspace=0)
#ax = plt.subplot(gs0[:, :])

h = ax.imshow(Exact.T, interpolation='nearest', cmap='viridis',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto', vmin=Exact.min(), vmax=Exact.max())
#divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(h)
#cbar.set_clim(U_pred.min(), U_pred.max())

#ax.plot(X_u_train[:,1], X_u_train[:,0], 'k.', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 2, clip_on = False)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
#ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
#ax.set_title('$u(t,x)$', fontsize = 10)

fig.tight_layout(pad=0.1)
#savefig('heateq-observations')

