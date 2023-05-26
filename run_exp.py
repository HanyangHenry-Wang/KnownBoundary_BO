from know_boundary.GP import optimise,optimise_warp
from know_boundary.utlis import Trans_function, get_initial_points
from know_boundary.acquisition_function import EI_acquisition_opt,MES_acquisition_opt,Warped_TEI2_acquisition_opt
import numpy as np
import matplotlib.pyplot as plt
import GPy
import torch
import botorch
from botorch.test_functions import Ackley,Levy,Beale,Branin,Hartmann,Rosenbrock
from botorch.utils.transforms import unnormalize,normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

function_information = []


temp={}
temp['name']='Branin2D' 
temp['function'] = Branin(negate=False)
temp['fstar'] =  0.397887
temp['min']=True 
function_information.append(temp)

temp={}
temp['name']='Beale2D' 
temp['function'] = Beale(negate=False)
temp['fstar'] =  0.
temp['min']=True 
function_information.append(temp)

temp={}
temp['name']='Hartmann3D' 
temp['function'] = Hartmann(dim=3,negate=False)
temp['fstar'] =  -3.86278
temp['min']=True 
function_information.append(temp)

temp={}
temp['name']='Rosenbrock5D' 
temp['function'] = Rosenbrock(dim=5,negate=False)
temp['fstar'] = 0.
temp['min']=True 
function_information.append(temp)

temp={}
temp['name']='Ackley6D' 
temp['function'] = Ackley(dim=6,negate=False)
temp['fstar'] = 0.
temp['min']=True 
function_information.append(temp)

temp={}
temp['name']='Levy8D' 
temp['function'] = Levy(dim=8,negate=False)
temp['fstar'] = 0.
temp['min']=True 
function_information.append(temp)



for information in function_information:

    fun = information['function']
    dim = fun.dim
    bounds = fun.bounds
    standard_bounds=np.array([0.,1.]*dim).reshape(-1,2)
    
    n_init = 4*dim
    iter_num = 12*dim
    N = 25

    fstar = information['fstar']
    fun = Trans_function(fun,fstar,min=True)
    
    ################################# GP+EI ###########################################
    BO_EI = []

    for exp in range(N):
        
        seed = exp

        X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
        Y_BO = torch.tensor(
            [fun(x) for x in X_BO], dtype=dtype, device=device
        ).reshape(-1,1)

        best_record = [Y_BO.min().item()]

        for i in range(iter_num):
            
                train_Y = (Y_BO - Y_BO.mean()) / Y_BO.std()
                train_X = normalize(X_BO, bounds)
                
                minimal = train_Y.min().item()
                
                train_Y = train_Y.numpy()
                train_X = train_X.numpy()
                
                # train the GP
                res = optimise(train_X,train_Y)
                kernel = GPy.kern.RBF(input_dim=dim,lengthscale= np.sqrt(res[0]),variance=res[1]) 
                m = GPy.models.GPRegression(train_X, train_Y,kernel)
                m.Gaussian_noise.variance.fix(10**(-5))

                standard_next_X = EI_acquisition_opt(m,bounds=standard_bounds,f_best=minimal)
                X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
                Y_next = fun(X_next).reshape(-1,1)

                # Append data
                X_BO = torch.cat((X_BO, X_next), dim=0)
                Y_BO = torch.cat((Y_BO, Y_next), dim=0)
                
                best_record.append(Y_BO.min().item())
                
        best_record = np.array(best_record)+fstar 
        BO_EI.append(best_record)
        
    np.savetxt('exp_res/'+information['name']+'_BO_EI', BO_EI, delimiter=',')
        
    ##################################################### GP+MES ##################################################
    BO_MES = []

    for exp in range(N):

        seed = exp
    
        X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
        Y_BO = torch.tensor(
            [fun(x) for x in X_BO], dtype=dtype, device=device
        ).reshape(-1,1)
        
        fstar_mes = 0. 

        best_record = [Y_BO.min().item()]

        for i in range(iter_num):
            
                train_Y = (Y_BO - Y_BO.mean()) / Y_BO.std()
                train_X = normalize(X_BO, bounds)
                
                
                fstar_standard = (fstar_mes - Y_BO.mean()) / Y_BO.std()
                fstar_standard = fstar_standard.item()
                
                train_Y = train_Y.numpy()
                train_X = train_X.numpy()
                
                # train the GP
                res = optimise(train_X,train_Y)
                kernel = GPy.kern.RBF(input_dim=dim,lengthscale= np.sqrt(res[0]),variance=res[1]) 
                m = GPy.models.GPRegression(train_X, train_Y,kernel)
                m.Gaussian_noise.variance.fix(10**(-5))

                standard_next_X = MES_acquisition_opt(m,standard_bounds,fstar_standard)
                X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
                Y_next = fun(X_next).reshape(-1,1)

                # Append data
                X_BO = torch.cat((X_BO, X_next), dim=0)
                Y_BO = torch.cat((Y_BO, Y_next), dim=0)
                
                best_record.append(Y_BO.min().item())
                
        best_record = np.array(best_record)+fstar 
        BO_MES.append(best_record)
        
    np.savetxt('exp_res/'+information['name']+'_BO_MES', BO_MES, delimiter=',')
    
    ##################################################### log GP+TEI2 ##################################################
    Warped_BO_TEI2 = []

    for exp in range(N):

        seed = exp

        X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
        Y_BO = torch.tensor(
            [fun(x) for x in X_BO], dtype=dtype, device=device
        ).reshape(-1,1)

  

        best_record = [Y_BO.min().item()]

        for i in range(iter_num):
            
                train_Y = Y_BO.numpy()
                train_X = normalize(X_BO, bounds)
                train_X = train_X.numpy()
                
                # train the GP
                res = optimise_warp(train_X, train_Y)
                lengthscale = np.sqrt(res[0])
                variance = res[1]
                c = res[2]
                
                
                warp_Y = np.log(train_Y+c)
                mean_warp_Y = np.mean(warp_Y) # use to predict mean
                warp_Y_standard = warp_Y-mean_warp_Y
                
                
                kernel = GPy.kern.RBF(input_dim=dim,lengthscale= lengthscale,variance=variance)  
                m = GPy.models.GPRegression(train_X, warp_Y_standard,kernel)
                m.Gaussian_noise.variance.fix(10**(-5))
                
                standard_next_X = Warped_TEI2_acquisition_opt(model=m,bounds=standard_bounds,f_best=best_record[-1],c=c,f_mean=mean_warp_Y)
                X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
                Y_next = fun(X_next).reshape(-1,1)

                # Append data
                X_BO = torch.cat((X_BO, X_next), dim=0)
                Y_BO = torch.cat((Y_BO, Y_next), dim=0)
                
                best_record.append(Y_BO.min().item())
                print(best_record[-1])
                
        best_record = np.array(best_record)+fstar         
        Warped_BO_TEI2.append(best_record)
        
    np.savetxt('exp_res/'+information['name']+'_logBO_TEI2', Warped_BO_TEI2, delimiter=',')
        