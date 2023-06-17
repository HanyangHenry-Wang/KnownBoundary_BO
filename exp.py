from known_boundary.GP import optimise,optimise_warp
from known_boundary.utlis import Trans_function, get_initial_points,transform
from known_boundary.acquisition_function_Botorch import EI,MES,LCB,ERM,LogGP_EI,LogGP_TEI
import numpy as np
import math
import torch
from botorch.test_functions import Ackley,Levy,Beale,Branin,Hartmann,Rosenbrock,Powell
from botorch.utils.transforms import unnormalize,normalize


from botorch.optim import optimize_acqf
from gpytorch.kernels import RBFKernel,ScaleKernel
from gpytorch.means import ZeroMean
from botorch.models.gp_regression import FixedNoiseGP

from botorch.exceptions import BadInitialCandidatesWarning

import warnings

from botorch.exceptions.warnings import BotorchTensorDimensionWarning, InputDataWarning
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

#warnings.filterwarnings("ignore", category=UserWarning)


warnings.filterwarnings(
            "ignore",
            message="Input data is not standardized.",
            category=InputDataWarning,
        )




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

# temp={}
# temp['name']='Hartmann3D' 
# temp['function'] = Hartmann(dim=3,negate=False)
# temp['fstar'] =  -3.86278
# temp['min']=True 
# function_information.append(temp)


# temp={}
# temp['name']='Powell4D' 
# temp['function'] = Powell(dim=4,negate=False)
# temp['fstar'] = 0.
# temp['min']=True 
# function_information.append(temp)

# temp={}
# temp['name']='Rosenbrock5D' 
# temp['function'] = Rosenbrock(dim=5,negate=False)
# temp['fstar'] = 0.
# temp['min']=True 
# function_information.append(temp)

# temp={}
# temp['name']='Ackley7D' 
# temp['function'] = Ackley(dim=7,negate=False)
# temp['fstar'] = 0.
# temp['min']=True 
# function_information.append(temp)


for information in function_information:

    fun = information['function']
    dim = fun.dim
    bounds = fun.bounds
    standard_bounds=np.array([0.,1.]*dim).reshape(-1,2)
    
    n_init = 4*dim
    iter_num = min(10*dim,60) 
    exp_num = 20
    batch_size = 1
    NUM_RESTARTS = 3*dim
    RAW_SAMPLES = 30*dim

    fstar = information['fstar']
    fun = Trans_function(fun,fstar,min=True)




    ########################################## EI ######################################################

    BO_EI = []

    for N in range(exp_num):

        print(N)

        seed = N
        X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
        Y_BO = torch.tensor(
            [fun(x) for x in X_BO], dtype=dtype, device=device
        ).reshape(-1,1)


        torch.manual_seed(seed)

        best = Y_BO.min().item()
        best_record = [best]

        for i in range(iter_num):  # Run until TuRBO converges
            #print(i)
            # Fit a GP model
            train_yvar = torch.tensor(10**(-5), device=device, dtype=dtype)

            train_Y = (Y_BO - Y_BO.mean()) / Y_BO.std()
            train_X = normalize(X_BO, bounds)

            train_Y = train_Y.numpy()
            train_X = train_X.numpy()
            
            # train the GP
            res = optimise(train_X,train_Y)



            covar_module = ScaleKernel(RBFKernel())
            model = FixedNoiseGP(torch.tensor(train_X), torch.tensor(train_Y),train_yvar.expand_as(torch.tensor(train_Y)), mean_module = ZeroMean(),covar_module=covar_module)
            
            model.covar_module.base_kernel.lengthscale = torch.sqrt(torch.tensor(res[0]))
            model.covar_module.outputscale = torch.tensor(res[1])


            # Create a batch
            AF = EI(model=model, best_f=torch.tensor(train_Y).min()) .to(device)
            X_next_normalized, _ = optimize_acqf(
                acq_function=AF,
                bounds=torch.tensor([0.,1.]*dim).reshape(-1,2).T,
                q=batch_size,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                options={'maxiter':50*dim,'maxfun':50*dim,'disp': False},
            )

            X_next = unnormalize(X_next_normalized, bounds)


            Y_next = torch.tensor(
                [fun(x) for x in X_next], dtype=dtype, device=device
            ).reshape(-1,1)

            # Append data
            X_BO = torch.cat((X_BO, X_next), dim=0)
            Y_BO = torch.cat((Y_BO, Y_next), dim=0)

            best_record.append(Y_BO.min())
            
        best_record = np.array(best_record)+fstar    
        BO_EI.append(best_record)
    
    np.savetxt('results/'+information['name']+'_EI', BO_EI, delimiter=',')
    
        
    ################################## MES ##################################################  
    BO_MES = []

    for N in range(exp_num):

        print(N)

        seed = N
        X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
        Y_BO = torch.tensor(
            [fun(x) for x in X_BO], dtype=dtype, device=device
        ).reshape(-1,1)


        torch.manual_seed(seed)

        best = Y_BO.min().item()
        best_record = [best]
        
        fstar_mes = 0. 

        for i in range(iter_num):  # Run until TuRBO converges
            #print(i)
            # Fit a GP model
            train_yvar = torch.tensor(10**(-5), device=device, dtype=dtype)

            train_Y = (Y_BO - Y_BO.mean()) / Y_BO.std()
            train_X = normalize(X_BO, bounds)

            train_Y = train_Y.numpy()
            train_X = train_X.numpy()
            
            fstar_standard = (fstar_mes - Y_BO.mean()) / Y_BO.std()
            fstar_standard = fstar_standard.item()
        
            # train the GP
            res = optimise(train_X,train_Y)



            covar_module = ScaleKernel(RBFKernel())
            model = FixedNoiseGP(torch.tensor(train_X), torch.tensor(train_Y),train_yvar.expand_as(torch.tensor(train_Y)), mean_module = ZeroMean(),covar_module=covar_module)
            
            model.covar_module.base_kernel.lengthscale = torch.sqrt(torch.tensor(res[0]))
            model.covar_module.outputscale = torch.tensor(res[1])


            # Create a batch
            AF = MES(model=model, fstar=fstar_standard) .to(device)
            X_next_normalized, _ = optimize_acqf(
                acq_function=AF,
                bounds=torch.tensor([0.,1.]*dim).reshape(-1,2).T,
                q=batch_size,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                options={'maxiter':50*dim,'maxfun':50*dim,'disp': False},
            )

            X_next = unnormalize(X_next_normalized, bounds)


            Y_next = torch.tensor(
                [fun(x) for x in X_next], dtype=dtype, device=device
            ).reshape(-1,1)

            # Append data
            X_BO = torch.cat((X_BO, X_next), dim=0)
            Y_BO = torch.cat((Y_BO, Y_next), dim=0)

            best_record.append(Y_BO.min())
        
        best_record = np.array(best_record)+fstar    
        BO_MES.append(best_record)
        
    np.savetxt('results/'+information['name']+'_MES', BO_MES, delimiter=',')
        
        
    ###################################### ERM ##########################################################

    BO_ERM = []
    for N in range(exp_num):
        
        print(N)

        seed = N
        
        fstar0 = 0.
        Trans = False

        X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
        Y_BO = torch.tensor(
                    [fun(x) for x in X_BO], dtype=dtype, device=device
                ).reshape(-1,1)

        best_record = [Y_BO.min().item()]

        for i in range(iter_num):

            #print(iter)
            train_Y = (Y_BO - Y_BO.mean()) / Y_BO.std()
            train_X = normalize(X_BO, bounds)
            
            train_Y = train_Y.numpy()
            train_X = train_X.numpy()
            

            fstar_standard = (fstar0 - Y_BO.mean()) / Y_BO.std()
            fstar_standard = fstar_standard.item()
            
            train_yvar = torch.tensor(10**(-5), device=device, dtype=dtype)
            
            if not Trans:
                minimal = np.min(train_X)
                res = optimise(train_X,train_Y)
                            
                covar_module = ScaleKernel(RBFKernel())
                model = FixedNoiseGP(torch.tensor(train_X), torch.tensor(train_Y),train_yvar.expand_as(torch.tensor(train_Y)), mean_module = ZeroMean(),covar_module=covar_module)
                
                model.covar_module.base_kernel.lengthscale = torch.sqrt(torch.tensor(res[0]))
                model.covar_module.outputscale = torch.tensor(res[1])
                

                # Create a batch
                AF = EI(model=model, best_f=torch.tensor(train_Y).min()) .to(device)
                
                X_next_normalized, _ = optimize_acqf(
                acq_function=AF,
                bounds=torch.tensor([0.,1.]*dim).reshape(-1,2).T,
                q=batch_size,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                options={'maxiter':50*dim,'maxfun':50*dim,'disp': False},
                    )

                beta = math.sqrt(math.log(train_X.shape[0]))
                
                AF = LCB(model=model, beta=beta) .to(device)
                
                _, val = optimize_acqf(
                acq_function=AF,
                bounds=torch.tensor([0.,1.]*dim).reshape(-1,2).T,
                q=batch_size,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                options={'maxiter':50*dim,'maxfun':50*dim,'disp': False},
                    )
                
                lcb = -val
                
            
                if lcb < fstar_standard:
                    Trans = True
            
            else:                        
                train_Y_transform = transform(y=train_Y,fstar=fstar_standard)
                mean_temp = np.mean(train_Y_transform)
                
                res = optimise(train_X,(train_Y_transform-mean_temp))
                
                covar_module = ScaleKernel(RBFKernel())
                model = FixedNoiseGP(torch.tensor(train_X), torch.tensor(train_Y_transform-mean_temp),train_yvar.expand_as(torch.tensor(train_Y_transform-mean_temp)), mean_module = ZeroMean(),covar_module=covar_module)
                
                model.covar_module.base_kernel.lengthscale = torch.sqrt(torch.tensor(res[0]))
                model.covar_module.outputscale = torch.tensor(res[1])
                
                
                AF = ERM(model=model, fstar=torch.tensor(fstar_standard),mean_temp=torch.tensor(mean_temp)) .to(device)
                
                X_next_normalized, _ = optimize_acqf(
                acq_function=AF,
                bounds=torch.tensor([0.,1.]*dim).reshape(-1,2).T,
                q=batch_size,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                options={'maxiter':50*dim,'maxfun':50*dim,'disp': False},
                    )
                
                
            
            
            X_next = unnormalize(X_next_normalized, bounds).reshape(-1,dim)     
            Y_next = fun(X_next).reshape(-1,1)

            # Append data
            X_BO = torch.cat((X_BO, X_next), dim=0)
            Y_BO = torch.cat((Y_BO, Y_next), dim=0)

            best_value = float(Y_BO.min())
            best_record.append(best_value)


        best_record = np.array(best_record)+fstar
        BO_ERM.append(best_record)
        
    np.savetxt('results/'+information['name']+'_ERM', BO_ERM, delimiter=',')
        
        
    ################################ logGP+EI ##########################################
    Warped_BO_EI = []

    for N in range(exp_num):
        
        print(N)

        seed = N

        X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
        Y_BO = torch.tensor(
            [fun(x) for x in X_BO], dtype=dtype, device=device
        ).reshape(-1,1)

        best_record = [Y_BO.min().item()]

        for i in range(iter_num):
                
                #print(i)
                train_yvar = torch.tensor(10**(-5), device=device, dtype=dtype)
            
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
                
                covar_module = ScaleKernel(RBFKernel())
                model = FixedNoiseGP(torch.tensor(train_X), torch.tensor(warp_Y_standard),train_yvar.expand_as(torch.tensor(warp_Y_standard)), mean_module = ZeroMean(),covar_module=covar_module)
                
                model.covar_module.base_kernel.lengthscale = torch.sqrt(torch.tensor(res[0]))
                model.covar_module.outputscale = torch.tensor(res[1])

                
                log_EI = LogGP_EI(model=model, best_f=Y_BO.min(),c=torch.tensor(c) , f_mean=torch.tensor(mean_warp_Y)) .to(device)
                standard_next_X, _ = optimize_acqf(
                    acq_function=log_EI,
                    bounds=torch.tensor([0.,1.]*dim).reshape(-1,2).T,
                    q=batch_size,
                    num_restarts=NUM_RESTARTS,
                    raw_samples=RAW_SAMPLES,
                    options={'maxiter':50*dim,'maxfun':50*dim,'disp': False},
                )
            
                X_next = unnormalize(standard_next_X, bounds).reshape(-1,dim)            
                Y_next = fun(X_next).reshape(-1,1)

                # Append data
                X_BO = torch.cat((X_BO, X_next), dim=0)
                Y_BO = torch.cat((Y_BO, Y_next), dim=0)
                
                best_record.append(Y_BO.min().item())
                #print(best_record[-1])
                
        best_record = np.array(best_record)+fstar         
        Warped_BO_EI.append(best_record)
        
    np.savetxt('results/'+information['name']+'_logGP_EI', Warped_BO_EI, delimiter=',')
        
    ######################################## logGP +TEI ######################################################

    Warped_BO_TEI = []

    for N in range(exp_num):
        
        print(N)

        seed = N

        X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
        Y_BO = torch.tensor(
            [fun(x) for x in X_BO], dtype=dtype, device=device
        ).reshape(-1,1)

        best_record = [Y_BO.min().item()]

        for i in range(iter_num):
                
                #print(i)
                train_yvar = torch.tensor(10**(-5), device=device, dtype=dtype)
            
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
                
                covar_module = ScaleKernel(RBFKernel())
                model = FixedNoiseGP(torch.tensor(train_X), torch.tensor(warp_Y_standard),train_yvar.expand_as(torch.tensor(warp_Y_standard)), mean_module = ZeroMean(),covar_module=covar_module)
                
                model.covar_module.base_kernel.lengthscale = torch.sqrt(torch.tensor(res[0]))
                model.covar_module.outputscale = torch.tensor(res[1])

                
                log_TEI = LogGP_TEI(model=model, best_f=Y_BO.min(),c=torch.tensor(c) , f_mean=torch.tensor(mean_warp_Y)) .to(device)
                standard_next_X, _ = optimize_acqf(
                    acq_function=log_TEI,
                    bounds=torch.tensor([0.,1.]*dim).reshape(-1,2).T,
                    q=batch_size,
                    num_restarts=NUM_RESTARTS,
                    raw_samples=RAW_SAMPLES,
                    options={'maxiter':50*dim,'maxfun':50*dim,'disp': False},
                )

                X_next = unnormalize(standard_next_X, bounds).reshape(-1,dim)            
                Y_next = fun(X_next).reshape(-1,1)

                # Append data
                X_BO = torch.cat((X_BO, X_next), dim=0)
                Y_BO = torch.cat((Y_BO, Y_next), dim=0)
                
                best_record.append(Y_BO.min().item())
                
        best_record = np.array(best_record)+fstar         
        Warped_BO_TEI.append(best_record)
        
        
    np.savetxt('results/'+information['name']+'_logGP_TEI', Warped_BO_TEI, delimiter=',')
