// The following are inputs which are fed into the model from R.
data {
  int<lower=0> N; // Number of cells.
  // The following are vectors of indicators.
  // They are set to 1 if the ith cell is as per the variable name, otherwise 0.
  vector<lower=0, upper=1>[N] genderFemale;
  vector<lower=0, upper=1>[N] healthBad;
  vector<lower=0, upper=1>[N] occRiskLow;
  vector<lower=0, upper=1>[N] occRiskHigh;
  // Expected deaths for the N cells.
  vector<lower=0>[N] D_expected;
  // Actual deaths for the N cells.
  array[N] int<lower=0> D_actual;
  // Known parameters for our prior parameter distributions.
  real baseAdjMean;
  real<lower=0> baseAdjSD;
  real genderFemaleAdjMean;
  real<lower=0> genderFemaleAdjSD;
  real healthBadAdjMean;
  real<lower=0> healthBadAdjSD;
  real occRiskLowAdjMean;
  real<lower=0> occRiskLowAdjSD;
  real occRiskHighAdjMean;
  real<lower=0> occRiskHighAdjSD;
}

parameters {
  // These correspond to the * versions of the parameters referred to in this report.
  real baseAdjX;
  real genderFemaleAdjX;
  real healthBadAdjX;
  real occRiskLowAdjX;
  real occRiskHighAdjX;
}

transformed parameters {
  // In order to undo the effect of the log-link function, the fitted estimates are exponentiated.
  real baseAdj = exp(baseAdjX);
  real genderFemaleAdj = exp(genderFemaleAdjX);
  real healthBadAdj = exp(healthBadAdjX);
  real occRiskLowAdj = exp(occRiskLowAdjX);
  real occRiskHighAdj = exp(occRiskHighAdjX);
}

model {
  // Our prior beliefs.
  baseAdjX ~ normal(baseAdjMean, baseAdjSD);
  genderFemaleAdjX ~ normal(genderFemaleAdjMean, genderFemaleAdjSD);
  healthBadAdjX ~ normal(healthBadAdjMean, healthBadAdjSD);
  occRiskLowAdjX ~ normal(occRiskLowAdjMean, occRiskLowAdjSD);
  occRiskHighAdjX ~ normal(occRiskHighAdjMean, occRiskHighAdjSD);
  
  // The overall estimate of our fitted adjustment.
  vector[N] overallAdjX =
    baseAdjX +
    genderFemaleAdjX * genderFemale +
    healthBadAdjX * healthBad +
    occRiskLowAdjX * occRiskLow +
    occRiskHighAdjX * occRiskHigh;
  
  // Our likelihood; the 'log' version of the Poisson
  // distribution implements the log-link function.
  D_actual ~ poisson_log(log(D_expected) + overallAdjX);
}
