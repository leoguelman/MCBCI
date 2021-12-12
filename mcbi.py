"""
Paper: Model Criticism for Bayesian Inference

@misc{tran2016model,
      title={Model Criticism for Bayesian Causal Inference}, 
      author={Dustin Tran and Francisco J. R. Ruiz and Susan Athey and David M. Blei},
      year={2016},
      eprint={1610.09037},
      archivePrefix={arXiv},
      primaryClass={stat.ME}
}
"""


### Imports

import os
os.chdir('/Users/lguelman/Library/Mobile Documents/com~apple~CloudDocs/LG_Files/Development/papers_reproducibility/Model Criticism for Bayesian Inference/')

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
parameters = {'figure.figsize': (8, 4),
              'font.size': 8, 
              'axes.labelsize': 12}
plt.rcParams.update(parameters)
plt.style.use('fivethirtyeight')
from IPython.display import Image

import pystan
import multiprocessing
import stan_utility
import arviz as az

import seaborn as sns
from stan_utils import stan_model_summary


### Generate Synthetic Data

class DataGeneratingProcess:
    """
    
    Generate sythetic data 
    
     Parameters
    ----------
    N : number of observations
    d : dimensionality of covariate space
    rho : correlation between the potential outcomes
   
    """
    def __init__(self, N: int = 500, d: int = 10, rho: float = 0.0, seed: int = 42):

        self.N = N
        self.d = d
        self.rho = rho
        self.seed = seed
        
        
    def __repr__(self):
            
            items = ("%s = %r" % (k, v) for k, v in self.__dict__.items())
            return "<%s: {%s}>" % (self.__class__.__name__, ', '.join(items))

    def generate_data(self):
        
        np.random.seed(self.seed)
        
        X = np.random.uniform(0.0, 1.0, size=(self.N, self.d))

        phi = np.random.normal(0.0, 1.0, size=(self.d, 1))

        lp = np.matmul(X, phi)

        a = (np.random.binomial(1, p = np.exp(lp) / (1+ np.exp(lp)))).squeeze()

        theta = np.random.normal(0.0, 1.0, size=(self.d+1, 1))

        mu_c = np.squeeze(np.matmul(np.hstack((X, np.zeros(self.N).reshape(self.N,1))), theta))
        mu_t = np.squeeze(np.matmul(np.hstack((X, np.ones(self.N).reshape(self.N,1))), theta))

        sigma_2 = np.random.gamma(1, 1, size=1).squeeze()
        cov_mat = np.array([[sigma_2, self.rho*sigma_2], 
                            [self.rho*sigma_2, sigma_2]])

        po = np.empty([self.N, 2])
        for i in range(self.N):
            po[i,:] = np.random.multivariate_normal(mean = [mu_c[i], mu_t[i]], cov = cov_mat, size = 1) 
    
        y0 = po[:, 0]  # potential outcome if a = 0
        y1 = po[:, 1]  # potential outcome if a = 1
        tau_unit = y1 - y0  # unit-level treatment effect

        # The realization of potential outcomes by the assignment mechanism
        y_obs = y0 * (1 - a) + y1 * a
        y_mis = y0 * a + y1 * (1 - a)

        df = pd.DataFrame({'y_obs':y_obs, 'y_mis': y_mis, 'y0':y0, 'y1': y1, 'tau_unit': tau_unit, 'a':a})
        df = df.join(pd.DataFrame(X, columns = ['X'+str(i) for i in range(1, self.d+1)])) 
        
        self.phi = phi
        self.a = a
        self.theta = theta
        self.sigma_2 = sigma_2
        self.y_obs = y_obs 
        self.X = X
  
        self.df = df
        
        return self


### Test data generator
seed=432
dgp = DataGeneratingProcess(seed=seed)
data = dgp.generate_data()
df = data.df
df

### Fit Models


# Case (a): The correct model 

stan_data_dict = {'N': data.N,
                  'd_a': data.d,
                  'd_o': data.d,
                  'X_a': data.X,
                  'X_o': data.X,
                  'y': data.y_obs,
                  'a': data.a,
                  'rho': 0.0,
                  'a_miss':0,
                  }

sm = pystan.StanModel('stan/mbi_stan.stan') 
multiprocessing.set_start_method("fork", force=True)
fit = sm.sampling(data=stan_data_dict, iter=1000, chains=4)

summary_stan_fit = stan_model_summary(fit)
summary_stan_fit

tau_fs = summary_stan_fit.loc[['tau_fs']]
tau_fs
data.theta[-1]

data.phi
phi_ = summary_stan_fit.iloc[0:10]
phi_

data.theta[0:10]
theta_ = summary_stan_fit.iloc[10:20]
theta_

samples = fit.extract(permuted=True)
samples_phi = samples['phi']

samples_a_rep = samples['a_rep'].T

# Case (b): Misspecified outcome model

stan_data_dict = {'N': data.N,
                  'd_a': data.d,
                  'd_o': data.d-1,
                  'X_a': data.X,
                  'X_o': data.X[:,1:10],
                  'y': data.y_obs,
                  'a': data.a,
                  'rho': 0.0,
                  'a_miss':0,
                  }

sm = pystan.StanModel('stan/mbi_stan.stan') 
multiprocessing.set_start_method("fork", force=True)
fit2 = sm.sampling(data=stan_data_dict, iter=1000, chains=4)

summary_stan_fit2 = stan_model_summary(fit2)
summary_stan_fit2

data.theta[-1]
tau_fs2 = summary_stan_fit2.loc[['tau_fs']]
tau_fs2

data.phi
phi_2 = summary_stan_fit2.iloc[0:10]
phi_2

data.theta[0:10]
theta_2 = summary_stan_fit2.iloc[10:19]
theta_2

samples2 = fit2.extract(permuted=True)
samples_phi2 = samples2['phi']


# Case (c): Misspecified assignment model 

stan_data_dict = {'N': data.N,
                  'd_a': data.d,
                  'd_o': data.d,
                  'X_a': data.X,
                  'X_o': data.X,
                  'y': data.y_obs,
                  'a': data.a,
                  'rho': 0.0,
                  'a_miss':1,
                  }

sm = pystan.StanModel('stan/mbi_stan.stan') 
multiprocessing.set_start_method("fork", force=True)
fit3 = sm.sampling(data=stan_data_dict, iter=1000, chains=4)

summary_stan_fit3 = stan_model_summary(fit3)
summary_stan_fit3

tau_fs3 = summary_stan_fit3.loc[['tau_fs']]
tau_fs3
data.theta[-1]


data.phi
phi_3 = summary_stan_fit3.iloc[0:10]
phi_3 

data.theta[0:10]
theta_3= summary_stan_fit3.iloc[10:20]
theta_3

samples3 = fit3.extract(permuted=True)
samples_phi3 = samples3['phi']


### Criticism - Assignment Model 

samples = fit.extract(permuted=True)

p_phi = samples['phi']
p_a_rep = samples['a_rep']

# reference vs realized discrepancy - use mean probability of assignment instead of log-likelihood.
sns.histplot(data=np.mean(p_a_rep, axis=1), stat="density", label="Reference")
plt.axvline(np.quantile(np.mean(p_a_rep, axis=1), .1), color='g', linestyle='--', label='90% intervals')
plt.axvline(np.quantile(np.mean(p_a_rep, axis=1), .9), color='g', linestyle='--')
plt.axvline(np.mean(data.a), color='r', label='Realized')
plt.legend(loc="upper right", fontsize='small')

samples3 = fit3.extract(permuted=True)

p_phi = samples3['phi']
p_a_rep = samples3['a_rep']

# reference vs realized discrepancy - use mean probability of assignment instead of log-likelihood.
sns.histplot(data=np.mean(p_a_rep, axis=1), stat="density", label="Reference")
plt.axvline(np.quantile(np.mean(p_a_rep, axis=1), .1), color='g', linestyle='--', label='90% intervals')
plt.axvline(np.quantile(np.mean(p_a_rep, axis=1), .9), color='g', linestyle='--')
plt.axvline(np.mean(data.a), color='r', label='Realized')
plt.legend(loc="upper right", fontsize='small')
