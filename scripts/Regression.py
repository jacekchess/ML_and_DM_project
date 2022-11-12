# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import torch

X_test = pd.read_csv('../data/regression/X_test.csv').to_numpy()
X_train = pd.read_csv('../data/regression/X_train.csv').to_numpy()
y_test = pd.read_csv('../data/regression/y_test.csv').to_numpy()
y_train = pd.read_csv('../data/regression/y_train.csv').to_numpy()


# %%
CV = KFold(n_splits=10, shuffle=True)
alphas = [1, 2, 3, 4, 5]
errors = np.zeros((len(alphas) + 1, 10))

k = 0
for train_index, test_index in CV.split(X_train):
    X_train_CV, X_valid_CV = X_train[train_index], X_train[test_index]
    y_train_CV, y_valid_CV = y_train[train_index], y_train[test_index]

    regression = LinearRegression().fit(X_train_CV, y_train_CV)
    errors[0, k] = mean_squared_error(y_valid_CV, regression.predict(X_valid_CV))

    for i, alpha in enumerate(alphas):
        ridge = Ridge(alpha).fit(X_train_CV, y_train_CV)
        errors[i + 1, k] = mean_squared_error(y_valid_CV, ridge.predict(X_valid_CV))

    k += 1

errors = np.mean(errors, axis = 1)
# %%
plt.plot([0] + alphas, errors)

# %%
ridge = Ridge(1).fit(X_train, y_train)
predicted_value = ridge.predict(X_test)[0]
coeficience = ridge.coef_

print('Coeficience: ', np.round(coeficience, 2))
print('Feature values: ', np.round(X_test[0], 2))
print('Predicted value: ', np.round(predicted_value, 2))
print('True value: ', np.round(y_test[0], 2))

# %%
dummy = DummyRegressor().fit(X_train, y_train)
mean_squared_error(y_test, dummy.predict(X_test))

# %%
def train_neural_net(model, loss_fn, X, y,
                     n_replicates=3, max_iter = 10000, tolerance=1e-6):
    
    import torch
    logging_frequency = 1000 
    best_final_loss = 1e100
    for r in range(n_replicates):
        print('\n\tReplicate: {}/{}'.format(r+1, n_replicates))
        # Make a new net (calling model() makes a new initialization of weights) 
        net = model()
        
        # initialize weights based on limits that scale with number of in- and
        # outputs to the layer, increasing the chance that we converge to 
        # a good solution
        torch.nn.init.xavier_uniform_(net[0].weight)
        torch.nn.init.xavier_uniform_(net[2].weight)
        
        # A more complicated optimizer is the Adam-algortihm, which is an extension
        # of SGD to adaptively change the learing rate, which is widely used:
        optimizer = torch.optim.Adam(net.parameters())
        
        # Train the network while displaying and storing the loss
        print('\t\t{}\t{}\t\t\t{}'.format('Iter', 'Loss','Rel. loss'))
        learning_curve = [] # setup storage for loss at each step
        old_loss = 1e6
        for i in range(max_iter):
            y_est = net(X) # forward pass, predict labels on training set
            loss = loss_fn(y_est, y) # determine loss
            loss_value = loss.data.numpy() #get numpy array instead of tensor
            learning_curve.append(loss_value) # record loss for later display
            
            # Convergence check, see if the percentual loss decrease is within
            # tolerance:
            p_delta_loss = np.abs(loss_value-old_loss)/old_loss
            if p_delta_loss < tolerance: break
            old_loss = loss_value
            
            # display loss with some frequency:
            if (i != 0) & ((i+1) % logging_frequency == 0):
                print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
                print(print_str)
            # do backpropagation of loss and optimize weights 
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            
        # display final loss
        print('\t\tFinal loss:')
        print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
        print(print_str)
        
        if loss_value < best_final_loss: 
            best_net = net
            best_final_loss = loss_value
            best_learning_curve = learning_curve
        
    # Return the best curve along with its final loss and learing curve
    return best_net, best_final_loss, best_learning_curve



# %%
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']


N, M = X_train.shape
CV = KFold(n_splits=10, shuffle=True)
n_hidden_units = 1
model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
loss_fn = torch.nn.MSELoss()
max_iter = 10000
print('Training model of type:\n{}\n'.format(str(model())))

# Do cross-validation:
errors = [] 
for k, (train_index, test_index) in enumerate(CV.split(X_train)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1, 10))    
    
    X_train_CV = torch.Tensor(X_train[train_index,:] )
    y_train_CV = torch.Tensor(y_train[train_index] )
    # print(test_index)
    X_valid_CV = torch.Tensor(X_train[test_index,:] )
    y_valid_CV = torch.Tensor(y_train[test_index] )
    
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train_CV,
                                                       y=y_train_CV,
                                                       n_replicates=3,
                                                       max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))

    # Determine estimated class labels for test set
    y_test_est = net(X_valid_CV)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_valid_CV.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_valid_CV)).data.numpy() #mean
    errors.append(mse) # store error rate for current CV fold 
    
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')

################################################################################
### Create summary table
################################################################################
# %%
K1 = 5
K2 = 5

outter_CV = KFold(n_splits=K1, shuffle=True)
inner_CV = KFold(n_splits=K2, shuffle=True)

alphas = [0.5, 1, 1.5, 2, 2.5]
hs = [1, 2, 3, 4, 5, 6]

loss_fn = torch.nn.MSELoss()
max_iter = 10000
        

regularization_best_errors = [None] * K1
ANN_best_errors = [None] * K1
dummy_errors = [None] * K1

best_alphas = [None] * K1
best_hs = [None] * K1

for k_outter, (train_outter, test_outter) in enumerate(outter_CV.split(X_train)):
    print('\nCrossvalidation outter fold: {0}/{1}'.format(k_outter+1, K1))

    X_train_outter, X_valid_outter = X_train[train_outter], X_train[test_outter]
    y_train_outter, y_valid_outter = y_train[train_outter], y_train[test_outter]

    X_train_outter_torch = torch.Tensor(X_train_outter)
    y_train_outter_torch = torch.Tensor(y_train_outter)
    X_valid_outter_torch = torch.Tensor(X_valid_outter)
    y_valid_outter_torch = torch.Tensor(y_valid_outter)

    regularization_errors = np.zeros((len(alphas) + 1, K2))
    ANN_errors = np.zeros((len(hs), K2))

    for k_inner, (train_inner, test_inner) in enumerate(inner_CV.split(X_train_outter)):
        print('\nCrossvalidation inner fold: {0}/{1}'.format(k_inner+1, K2))

        X_train_inner, X_valid_inner = X_train_outter[train_inner], X_train_outter[test_inner]
        y_train_inner, y_valid_inner = y_train_outter[train_inner], y_train_outter[test_inner]

        X_train_inner_torch = torch.Tensor(X_train_inner)
        y_train_inner_torch = torch.Tensor(y_train_inner)
        X_valid_inner_torch = torch.Tensor(X_valid_inner)
        y_valid_inner_torch = torch.Tensor(y_valid_inner)

        # Regularization
        regression = LinearRegression().fit(X_train_inner, y_train_inner)
        regularization_errors[0, k_inner] = mean_squared_error(y_valid_inner, regression.predict(X_valid_inner))

        for i, alpha in enumerate(alphas):
            ridge = Ridge(alpha).fit(X_train_inner, y_train_inner)
            regularization_errors[i + 1, k_inner] = mean_squared_error(y_valid_inner, ridge.predict(X_valid_inner))


        # ANN
        N, M = X_train_inner.shape

        for i, h in enumerate(hs):     
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, h), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(h, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
            
            net, final_loss, learning_curve = train_neural_net(model,
                                                            loss_fn,
                                                            X=X_train_inner_torch,
                                                            y=y_train_inner_torch,
                                                            n_replicates=3,
                                                            max_iter=max_iter)

            ANN_errors[i, k_inner] = mean_squared_error(y_valid_inner, net(X_valid_inner_torch).detach().numpy())


    # Regularization
    regularization_errors = np.mean(regularization_errors, axis = 1)
    best_alpha = ([0] + alphas)[regularization_errors.argmin()]
    best_alphas[k_outter] = best_alpha
    
    if best_alpha == 0:
        regression = LinearRegression().fit(X_train_outter, y_train_outter)
    else:
        regression = Ridge(best_alpha).fit(X_train_outter, y_train_outter)

    regularization_best_errors[k_outter] = mean_squared_error(y_valid_outter, regression.predict(X_valid_outter))

    # ANN
    ANN_errors = np.mean(ANN_errors, axis = 1)
    best_h = hs[ANN_errors.argmin()]
    best_hs[k_outter] = best_h

    N, M = X_train_outter.shape
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, best_h), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(best_h, 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )
    
    net, final_loss, learning_curve = train_neural_net(model,
                                                    loss_fn,
                                                    X=X_train_outter_torch,
                                                    y=y_train_outter_torch,
                                                    n_replicates=3,
                                                    max_iter=max_iter)

    ANN_best_errors[k_outter] = mean_squared_error(y_valid_outter, net(X_valid_outter_torch).detach().numpy())

    # Dummy
    dummy = DummyRegressor().fit(X_train_outter, y_train_outter)
    dummy_errors[k_outter] = mean_squared_error(y_valid_outter, dummy.predict(X_valid_outter))


# %%

pd.concat([
    pd.DataFrame([1,2,3,4,5]),
    pd.DataFrame(best_hs),
    pd.DataFrame(ANN_best_errors),
    pd.DataFrame(best_alphas),
    pd.DataFrame(regularization_best_errors),
    pd.DataFrame(dummy_errors)
], axis = 1).round(2)