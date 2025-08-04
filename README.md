# machine-learning
Algorithms relating to basic machine learning concepts on Jupyter Notebook.

1. [Document Clustering](https://github.com/devinagabriella/machine-learning/blob/main/Doc_Clustering.ipynb)
2. [Logistic Regression versus Bayesian Classifiers](https://github.com/devinagabriella/machine-learning/blob/main/Logistic_Regression_Bayesian_Classifier.ipynb)
3. [Model Complexity and Model Selection](https://github.com/devinagabriella/machine-learning/blob/main/Model_Complexity_Selection.ipynb)
4. [Perceptron versus Neural Network](https://github.com/devinagabriella/machine-learning/blob/main/Perceptron_Neural_Network.ipynb)
5. [Gradient Descent Linear Regression with Ridge Regularization](https://github.com/devinagabriella/machine-learning/blob/main/Ridge_Regression.ipynb)
6. [Unsupervised Learning](https://github.com/devinagabriella/machine-learning/blob/main/Unsupervised_Learning.ipynb)


## Descriptions

### 1. [Document Clustering](https://github.com/devinagabriella/machine-learning/blob/main/Doc_Clustering.ipynb)
Train the program to group (or cluster) similar documents from a collection of documents based on the words of each document. 
This algorithm assumes both the complete and incomplete data, where the incomplete data means the cluster assignments are not given (latent variable). 
To solve the incomplete data problem, it uses the Expectation-Maximization (EM) algorithm

### 2. [Logistic Regression versus Bayesian Classifiers](https://github.com/devinagabriella/machine-learning/blob/main/Logistic_Regression_Bayesian_Classifier.ipynb)
Compare two of the probabilistic classifier models:
- The discriminative classifier, logistic regression
- The generative classifier, Bayesian Classifier (BC) and its variations (naive Bayes, BC with shared covariance, BC with full covariance).

### 3. [Model Complexity and Model Selection](https://github.com/devinagabriella/machine-learning/blob/main/Model_Complexity_Selection.ipynb)
Analyse the impact of the complexity parameters of $k$-NN regressor models with $L$-fold cross validation on model performance,
then implement the automatic model selector function based on the best complexity-performance pairs. 
(The cross validation complexity parameter is commonly denoted with $k$. Since it has been used, I change it to $L$ to avoid confusion).

### 4. [Perceptron versus Neural Network](https://github.com/devinagabriella/machine-learning/blob/main/Perceptron_Neural_Network.ipynb)
Compare the performance of perceptron and neural network on non-linear data,
since the latter is also known as multilayer perceptron (MLP).

### 5. [Gradient Descent Linear Regression with Ridge Regularization](https://github.com/devinagabriella/machine-learning/blob/main/Ridge_Regression.ipynb)
Implement linear regression optimisation algorithm, gradient descent with added ridge regularisation parameter.
Analyse the impact of the regularisation parameter on model complexity and generalisation.

### 6. [Autoencoder (Unsupervised Learning)](https://github.com/devinagabriella/machine-learning/blob/main/Unsupervised_Learning.ipynb)
Implement a self-taught learning using autoencoder and a 3-layer neural network on real world data. 
Analyse the impact of using the autoencoder hidden layer output as extra feature for the input of the 3-layer neural network.

