# Naive Bayes

## Training
For multi-class classification: 
- Compute $P(y=k|x) \propto \delta_k(x)$
- Select class w/ highest prop

$$ P(y=k|x) \propto \delta_k(x) = \log{P(y=k)P(x|y=k)} $$

$$  = \log{P(y=k)\sum_{j=1}^m P(x_j | y=k)} $$

$$ \delta_k(x) = \log{[\theta_k\sum_{j=1}^m \theta_{j, k}^{x_j} (1 - \theta_{j, k})^{1-x_j} ]}$$

where:
$$ \theta_k = P(y=k) = \frac{\#\text{ samples where }y=k}{\#\text{ samples}} $$

$$ \theta_{j, k} = P(x_j=1 | y=k) = \frac{\#\text{ samples where }x_j=1 \text{ and }y=k}{\#\text{ samples where }y=k}  $$ 

Classify the output as:
$$ \text{Output} = \argmax_k \delta_k(x) $$

### Prediction
$$ \delta_k(x) = \log{\theta_k} + \sum_{j=1}^m[ x_j * \log{\theta_{j, k}} +  (1-x_j)*\log{(1 - \theta_{j, k})^ ]}$$

## Laplace Smoothing
To deal w/ unobserved words in training data.

Replace $\theta_{j, k}$ with:
$$ \theta_{j, k} = P(x_j=1 | y=k) = \frac{(\#\text{ samples where }x_j=1 \text{ and }y=k) + 1}{(\#\text{ samples where }y=k) + 2}  $$ 


## SK Discriminants

jll = safe_sparse_dot(X, (feat_log_proba - feat_log_neg_proba).T)
        jll += log_class_prior + feat_log_neg_proba.sum(axis=1)
