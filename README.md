### empca: Weighted Expectation Maximization Principal Component Analysis ###

Classic PCA is great but it doesn't know how to handle noisy or missing
data properly.  This module provides Weighted Expectation Maximization PCA,
an iterative method for solving PCA while properly weighting data.
Missing data is simply the limit of weight=0.

Given data[nobs, nvar] and weights[nobs, nvar],

    m = empca(data, weights, options...)

That returns a Model object m, from which you can inspect the eigenvectors,
coefficients, and reconstructed model, e.g.

    pylab.plot( m.eigvec[0] )
    pylab.plot( m.data[0] )
    pylab.plot( m.model[0] )
    
If you want to apply the model to new data:

    m.set_data(new_data, new_weights)
    
and then it will recalculate m.coeff, m.model, m.rchi2, etc. for the new data.

m.R2() is the fraction of data variance explained by the model, while
m.R2vec(i) is the amount of variance explained by eigenvector i.

This implementation of EMPCA does *not* subtract the mean from the data.
If you don't subtract the mean yourself, it will still work, with
the first eigenvector likely being something similar to the mean.
    
For comparison, two alternate methods are also implemented which also
return a Model object:

    m = lower_rank(data, weights, options...)
    m = classic_pca(data)  #- but no weights or even options...

Everything is self contained in empca.py .  Just put that into your
PYTHONPATH and "pydoc empca" for more details.  For a quick test
on toy example data, run

    python empca.py

This requires numpy and scipy; it will make plots if you have pylab installed.

The paper S. Bailey 2012, PASP, 124, 1015 describes the underlying math
and is available as a pre-print at:

http://arxiv.org/abs/1208.4122

If you use this code in an academic paper, please include a citation
as described in CITATION.txt, and optionally an acknowledgement such as:

> This work uses the Weighted EMPCA code by Stephen Bailey,
> available at https://github.com/sbailey/empca/

The examples in the paper were prepared with version v0.2 of the code.

Stephen Bailey, Summer 2012

