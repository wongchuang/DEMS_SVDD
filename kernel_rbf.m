function [Ktrain, Ktest] = kernel_rbf(M_train, M_test, kappa)
    N = size(M_train,2);  
    NN = size(M_test,2);
    Dtrain = ((sum(M_train'.^2,2)*ones(1,N))+(sum(M_train'.^2,2)*ones(1,N))'-(2*(M_train'*M_train)));
    Dtest = ((sum(M_train'.^2,2)*ones(1,NN))+(sum(M_test'.^2,2)*ones(1,N))'-(2*(M_train'*M_test)));
    sigma2 = kappa * mean(mean(Dtrain));  
    A = 2.0 * sigma2;
    Ktrain = exp(-Dtrain/A);  		      
    Ktest = exp(-Dtest/A);
end