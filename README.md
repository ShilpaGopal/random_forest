# Tree based method

## Regression tree

Mainly there are two steps 

1. Divide the predictor space -- that is, set of possible values for X1, X2, ...Xp into J distinct and non-overlapping regions R1, R2, ...Rj
2. For every observation that falls into the region Rj, make the same prediction which is simple the mean of the response values in th region Rj

### How to construct the region?

* Divide the predictor space into high dimensional rectangles
* Goal is to find the regions that results in minimum mean-square-error(MSE)
* Consider the Top down greedy approach which is know as "Recursive binary splitting" 
* GREEDY - In the tree building process best split is decided by considering particular step only rather than looking at how this split will affect teh further steps
* Select the predictor Xj and cutpoint s such that the predictor space results in the greatest possible reduction in the MSE 
* Stopping criteria will be until no region contain some number of observations


### Algorithm for decision tree

1. Check the data is pure if yes 
    1.a create the leaf (Either classify or predict the value)
2. If data is not pure Identify the best feature to split 
3. Split the data 

Check again whether the data is pure repeat until less minimum observations in leaf is reached