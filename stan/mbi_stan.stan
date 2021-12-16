data {
  int<lower=0> N;                   // sample size
  int<lower=0> d_a;                 // number of covariates in the assignment model
  int<lower=0> d_o;                 // number of covariates in the outcome model
  matrix[N, d_a] X_a;               // covariate matrix for assignment model
  matrix[N, d_o] X_o;               // covariate matrix for outcome model
  vector[N] y;                      // observed outcome
  int <lower = 0, upper = 1> a[N];  // treatment assigned
  real<lower=-1,upper=1> rho;       // correlation between the potential outcomes (assumed)
  int<lower = 0, upper = 1> a_miss; // assignment model misspecified(1), or not (0). 
}

parameters {
  vector[d_a] phi;                  // treatment assignment model weights 
  vector[d_o] theta;                // outcome model weights (excluding treatment weight)
  real tau;                         // super-population average treatment effect (treatment weight)
  real<lower=0> sigma_2;            // residual variance
}

transformed parameters {
  real<lower=0> sigma;
  vector[N] y_hat;

  sigma = sqrt(sigma_2);
 
  for (n in 1:N) {
  y_hat[n] = X_o[n] * theta + tau * a[n];
  }
}

model {
   
   // PRIORS
   phi ~ normal(0,1);
   theta ~ normal(0,1);
   tau ~ normal(0,1);
   sigma_2 ~ gamma(1,1);


   // LIKELIHOOD

   if(a_miss == 1){
     a ~ bernoulli_logit(0.7 + 0.3 * X_a * phi); 
   } else {
     a ~ bernoulli_logit(X_a * phi);
   }
   y ~ normal(y_hat, sigma);
  }

generated quantities{
  real tau_fs;                      // finite-sample average treatment effect  
  real y0[N];                       // potential outcome if a=0
  real y1[N];                       // potential outcome if a=1
  real tau_unit[N];                 // unit-level treatment effect
  vector [N] a_prob_rep;            // replicated prob predictions from treatment assignment from the posterior distribution. 
  vector [N] a_rep;                 // replicated predictions from treatment assignment from the posterior distribution. 
  vector[N] log_lik_y_0;            // calculate log-likelihood y_0
  vector[N] log_lik_y_1;            // calculate log-likelihood y_1
  vector[N] log_lik_rd;             // calculate log-likelihood reference discrepancy

  for(n in 1:N){
    real mu_c = X_o[n,]*theta;        
    real mu_t = X_o[n,]*theta + tau;

    # stan output required to compute realized discrepancy (outcome model)
    log_lik_y_0[n] = normal_lpdf(y[n] | mu_c, sigma);
    log_lik_y_1[n] = normal_lpdf(y[n] | mu_t, sigma);

    if(a[n] == 1){                
      y0[n] = normal_rng(mu_c + rho*(y[n] - mu_t), sigma*sqrt(1 - rho^2)); 
      y1[n] = y[n];
    } else{                        
      y0[n] = y[n];       
      y1[n] = normal_rng(mu_t + rho*(y[n] - mu_c), sigma*sqrt(1 - rho^2)); 
    }
    
    # stan output required to compute reference discrepancy (outcome model)
    log_lik_rd[n] = normal_lpdf(y0[n] | mu_c, sigma) + normal_lpdf(y1[n] | mu_t, sigma);

    tau_unit[n] = y1[n] - y0[n];
    a_prob_rep[n] = inv_logit(X_a[n] * phi);
    a_rep[n] = bernoulli_rng(inv_logit(X_a[n] * phi));
  }
  tau_fs = mean(tau_unit);        
}
