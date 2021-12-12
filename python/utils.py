import pandas as pd
import numpy as np


def stan_model_summary(stan_fit):
    """
    Parameters
    ----------
    stan_fit : A Stan fit object
    Returns
    -------
    A pandas data frame with summary of posterior parameters.
    """
    summary_dict = stan_fit.summary()
    summary_df = pd.DataFrame(summary_dict['summary'],
                  columns=summary_dict['summary_colnames'],
                  index=summary_dict['summary_rownames'])
    return summary_df


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
