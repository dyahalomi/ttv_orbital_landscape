import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle as astropy_ls
from scipy import optimize
from tqdm import tqdm


import matplotlib 
matplotlib.rc('xtick', labelsize=36) 
matplotlib.rc('ytick', labelsize=36) 
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)


def get_chisq(observed_data, predicted_data, uncertainties):
    """
    Calculate the chi-square statistic for model evaluation.

    Parameters:
        observed_data (numpy array): Array of observed data points.
        predicted_data (numpy array): Array of model-predicted data points.
        uncertainties (numpy array): Array of uncertainties/standard deviations of the observed data.

    Returns:
        float: The chi-square statistic.
    """
    if len(observed_data) != len(predicted_data) or len(observed_data) != len(uncertainties):
        raise ValueError("All input arrays must have the same length.")

    squared_residuals = ((observed_data - predicted_data) ** 2. / (uncertainties ** 2.) )
    chi_square = np.sum(squared_residuals)

    return chi_square


def get_log_likelihood(chi_square):
    """
    Calculate the log-likelihood from the chi-square value.

    Parameters:
        chi_square (float): The chi-square value of the model.
        num_data_points (int): Number of data points (sample size).

    Returns:
        float: The log-likelihood.
    """
    log_likelihood = -0.5 * chi_square
    return log_likelihood

def get_BIC(num_params, num_data_points, log_likelihood):
    """
    Calculate the Bayesian Information Criterion (BIC).

    Parameters:
        num_params (int): Number of parameters in the model.
        num_data_points (int): Number of data points (sample size).
        log_likelihood (float): Log-likelihood of the model.

    Returns:
        float: The BIC value.
    """
    BIC = num_params * np.log(num_data_points) - 2 * log_likelihood
    return BIC



def get_AIC(observed, predicted, uncertainties, num_parameters):
    """
    Calculate the Akaike Information Criterion (AIC) for a model with data uncertainties.

    Parameters:
        observed (array-like): The observed data.
        predicted (array-like): The model's predicted values.
        uncertainties (array-like): The uncertainties on the observed data.
        num_parameters (int): The number of parameters in the model.

    Returns:
        float: The AIC value.
    """
    # Calculate the number of data points
    n = len(observed)

    # Calculate the weighted sum of squared residuals (RSS)
    residuals = np.array(observed) - np.array(predicted)
    rss = np.sum((residuals / uncertainties)**2)

    # Calculate the AIC value
    aic = n * np.log(rss / n) + 2 * num_parameters

    return aic



def get_times_null_model(epochs, T0, P): 
    """
    Generates null model times based on linear ephemeris.

    Parameters:
    - epochs (np.array): Array of epochs.
    - T0 (float): Reference time of the first transit.
    - P (float): Orbital period.

    Returns:
    - np.array: Array of null model times.
    """
    return T0 + P*epochs


def solve_null_model(epochs, times, times_err):
    """
    Solves for T0 and P in a null model using linear regression.

    Parameters:
    - epochs (np.array): Array of epochs.
    - times (np.array): Array of observed transit times.
    - times_err (np.array): Array of errors associated with observed transit times.

    Returns:
    - Tuple: Tuple containing T0_fit (float) and P_fit (float), the fitted parameters.
    """
    
    # Constructing the design matrix A
    A = np.column_stack([
        np.ones_like(epochs),
        epochs
    ])

    # Constructing the weight matrix W using times_err
    W = np.diag(1.0 / times_err**2.)

    # Constructing the observed times vector y
    y = times

    # Calculating the parameters using np.linalg.solve
    parameters = np.linalg.solve(A.T @ W @ A, A.T @ W @ y)

    T0_fit, P_fit = parameters

    return T0_fit, P_fit



def get_times_n_waves(epochs, T0, P, P_LS_ttv, *fourier_params):
    """
    Calculate the time as a function of epochs with multiple waves.

    Parameters:
    - epochs: array-like, time values
    - T0: float, reference time
    - P: float, linear period
    - P_LS_ttv: float, period for the last wave
    - *fourier_params: variable number of Fourier parameters for each wave

    Returns:
    - times: array-like, calculated times
    """

    # Determine the number of waves
    n = (len(fourier_params) + 1) // 3

    # Calculate linear component
    linear = T0 + P * epochs

    # Initialize the total wave component
    waves = 0

    # Iterate through waves
    for ii in range(n - 1):
        alpha_i_ttv, beta_i_ttv, P_i_ttv = fourier_params[3 * ii: 3 * (ii + 1)]

        # Calculate wave for the current Fourier parameters
        wave_i = alpha_i_ttv * np.sin((2 * np.pi * epochs / P_i_ttv)) + beta_i_ttv * np.cos((2 * np.pi * epochs / P_i_ttv))

        # Add the current wave to the total
        waves += wave_i

    # Calculate the last wave separately
    alpha_LS_ttv, beta_LS_ttv = fourier_params[3 * (n - 1): 3 * n]
    #print("P_LS_ttv", P_LS_ttv)
    #print("epochs", epochs)
    #print("alpha_LS_ttv", alpha_LS_ttv)
    #print("beta_LS_ttv", beta_LS_ttv)
    #print('')
    wave_LS = alpha_LS_ttv * np.sin((2 * np.pi * epochs / P_LS_ttv)) + beta_LS_ttv * np.cos((2 * np.pi * epochs / P_LS_ttv))
    waves += wave_LS

    # Combine linear and wave components to get the final result
    times = linear + waves

    return times



import numpy as np
from scipy import optimize

def fit_multiple_frequencies(epochs, times, times_err, Pgrid, amplitude_cut=10., max_waves = 5):
    """
    Fit multiple frequencies to the given times and return optimized parameters.

    Parameters:
    - epochs: array-like, time values
    - times: array-like, observed times
    - times_err: array-like, errors on observed times
    - Pgrid: array-like, grid of periods to search for optimal fit
    - NOT USED HERE!! amplitude_cut: float, amplitude ratio threshold for wave detectability, default = 10.
    - max_waves: int, maximum number of waves, default = 2

    Returns:
    - optimized_params: array, optimized parameters for the best fit
    """

    # Solve null model and calculate null chisq
    null_model = solve_null_model(epochs, times, times_err)
    times_null_model = get_times_null_model(epochs, *null_model)
    null_chisq = get_chisq(times, times_null_model, times_err)

    n_waves = 1
    amplitude_detectable = True
    
    optimized_params_dict = {}
    while n_waves <= max_waves:
    #while amplitude_detectable and n_waves <= max_waves:
        #print('n waves: ', n_waves)

        # Initial guesses for the parameters
        if n_waves == 1:
            initial_guess = [0, 0, *([0, 0] * n_waves)]
        else:
            linear_params = list(optimized_params[0:2])
            fourier_params = list(optimized_params[2:len(optimized_params)]) + [best_P_ttv] + [0, 0]

            initial_guess = [*linear_params, *fourier_params]
            ntest = (len(fourier_params) + 1) // 3

        delta_chisq_grid = []
        ttv_chisq_grid = []
        
        for P_LS_ttv in Pgrid:
            try:
                # Create a lambda function with fixed P_LS_ttv
                fit_function = lambda epochs, T0, P, *fourier_params: get_times_n_waves(epochs, T0, P, P_LS_ttv,
                                                                                        *fourier_params)

                # Use curve_fit
                optimized_params, _ = optimize.curve_fit(fit_function, epochs, times, sigma=times_err, p0=initial_guess)

                # Extract optimized parameters
                T0_optimized, P_optimized, *fourier_params_optimized = optimized_params

                # Evaluate the function with the best-fit values
                best_fit_times = get_times_n_waves(epochs, T0_optimized, P_optimized, P_LS_ttv,
                                                   *fourier_params_optimized)

                # Calculate delta_chisq and ttv_chisq
                a_chisq = get_chisq(times, best_fit_times, times_err)
                #delta_chisq = max(0., null_chisq - a_chisq)
                delta_chisq = null_chisq - a_chisq
                delta_chisq_grid.append(delta_chisq)
                ttv_chisq_grid.append(a_chisq)

            except RuntimeError:
                # Exclude the problematic P_LS_ttv from the grid
                Pgrid = Pgrid[Pgrid != P_LS_ttv]
                continue
        

        # Find the period with maximum delta_chisq
        max_delta_chisq = -np.inf
        min_ttv_chisq = 0
        best_P_ttv = None
        

        for ii in range(len(delta_chisq_grid)):
            delta_chisq = delta_chisq_grid[ii]
            ttv_chisq = ttv_chisq_grid[ii]
            period = Pgrid[ii]

            if delta_chisq > max_delta_chisq:
                max_delta_chisq = delta_chisq
                min_ttv_chisq = ttv_chisq
                best_P_ttv = period

                
        

        
        
        
        # Recalculate the model with the best period
        P_LS_ttv = best_P_ttv

        # Create a lambda function with fixed P_LS_ttv
        fit_function = lambda epochs, T0, P, *fourier_params: get_times_n_waves(epochs, T0, P, P_LS_ttv,
                                                                                *fourier_params)

        # Use curve_fit
        optimized_params, _ = optimize.curve_fit(fit_function, epochs, times, sigma=times_err, p0=initial_guess)

        # Extract optimized parameters
        T0_optimized, P_optimized, *fourier_params_optimized = optimized_params

        
        #best_fit_times_epochs = get_times_n_waves(epochs, T0_optimized, P_optimized, P_LS_ttv,
        #                                          *fourier_params_optimized)
        
        #add optimized_params for nth wave to dict
        optimized_params_dict[n_waves] = list(optimized_params) + [best_P_ttv] 
        
        #if not evaluated by amplitude cut, skip this!
        '''
        # Get the amplitude of the nth wave
        amplitude_comps = fourier_params_optimized[-2:len(fourier_params_optimized)]
        amplitude = np.sqrt(amplitude_comps[0] ** 2. + amplitude_comps[1] ** 2.)
        print('amplitudes')
        print('----------')
        print(amplitude)

        # If it's not the first wave, compare amplitudes
        if n_waves != 1:
            amplitude_comps_1st = fourier_params_optimized[0:2]
            amplitude_1st = np.sqrt(amplitude_comps_1st[0] ** 2. + amplitude_comps_1st[1] ** 2.)

            print(amplitude_1st)

            print('amp ratio = ', str(amplitude_1st / amplitude))
            if amplitude_1st / amplitude > amplitude_cut:
                amplitude_detectable = False
        '''

        n_waves += 1
        #print('')
        #print('')

    # Return optimized parameters, excluding the last 2 parameters as the nth wave didn't survive
    #return optimized_params[0:-2]
    
    # Return optimized parameters, add in the best Pttv of last wave
    #return list(optimized_params) + [best_P_ttv] 
    return optimized_params_dict





def write_to_csv(filename, data, folder_path):
    csv_file_path = os.path.join(folder_path, filename)

    with open(csv_file_path, 'w', newline='') as f_object:
        writer_object = writer(f_object, delimiter=',')
        writer_object.writerow(data)
        # No need to explicitly close the file object, as it's done automatically in a 'with' statement

    print(f'Data written to CSV file "{csv_file_path}".')













#RUN!!

from tqdm import tqdm
import ttvfast
from csv import writer
import os
from datetime import datetime


ptrans = 100
pperts = (np.logspace(1, 4, 1000))

print('ptrans')
print(ptrans)
print('')

print('pperts')
print(pperts)
print('')



pttv1 = []
pttv2 = []






eccs_array = [[0., 0.], [0., 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.]]

for eccs in eccs_array:
    ecc_low = eccs[0]
    ecc_high = eccs[1]

    ppert_over_ptrans_all = []
    optimized_params_all = []
    transit_times_trans_all = []
    transit_times_pert_all = []
    lstsq_ptrans_all = []
    lstsq_ppert_all = []
    orbital_params_all = []

    for kk in range(0, 10):
        print('')
        print('starting run #' + str(kk+1))
        pttvs_over_ptrans = []
        amplitudes_ttv = []
        lin_periods = []
        lstsq_ptrans = []
        lstsq_ppert = []
        optimized_params_list = []
        transit_times_trans = []
        transit_times_pert = []

        pperts_out = []

        ecc1 = np.random.uniform(low=ecc_low, high=ecc_high)
        ecc2 = np.random.uniform(low=ecc_low, high=ecc_high)

        omega1 = np.random.uniform(low=0, high=360)
        omega2 = np.random.uniform(low=0, high=360)

        mean_anomaly1 = np.random.uniform(low=0, high=360)
        mean_anomaly2 = np.random.uniform(low=0, high=360)
        
        #mass1 = loguniform.rvs(3e-7, 1e-3)
        #mass2 = loguniform.rvs(3e-7, 1e-3)
        mass1 = 3e-6 #earth mass in solar masses
        #mass2 = 3e-6 #earth mass in solar masses
        #mass1 = 3e-7 #1/10 earth mass in solar masses
        #mass2 = 3e-7 #1/10 earth mass in solar masses
        #mass1 = 1e-3 #jupiter mass in solar masses
        mass2 = 1e-3 #jupiter mass in solar masses

        print('ecc1 = ' + str(ecc1))
        print('ecc2 = ' + str(ecc2))
        print('omega1 = ' + str(omega1))
        print('omega2 = ' + str(omega2))
        print('mean_anomaly1 = ' + str(mean_anomaly1))
        print('mean_anomaly2 = ' + str(mean_anomaly2))
        print('mass1 = ' + str(mass1))
        print('mass2 = ' + str(mass2))




        for ppert in tqdm(pperts, desc="Processing", unit="iteration"):

            #print(ppert)
            gravity = 0.000295994511                        # AU^3/day^2/M_sun
            stellar_mass = 1.0                   # M_sun


            planet1 = ttvfast.models.Planet(
                mass=mass1,                         # M_sun
                period=ptrans,              # days
                eccentricity=ecc1,
                inclination=90.,         # degrees
                longnode=0.,           # degrees
                argument=omega1,            # degrees
                mean_anomaly=mean_anomaly1,       # degrees
            )

            planet2 = ttvfast.models.Planet(
                mass=mass2,
                period=ppert,
                eccentricity=ecc2,
                inclination=90.,
                longnode=0.,
                argument=omega2,
                mean_anomaly=mean_anomaly2,
            )

            planets = [planet1, planet2]
            Time = 0.                                    # days
            if ppert < ptrans:
                dt = ppert/20.                              # days
            else:
                dt = ptrans/20.                             # days
            
            Total = ptrans*100                                    # days

            results = ttvfast.ttvfast(planets, stellar_mass, Time, dt, Total)

            planet_mask = np.array(results['positions'][0])
            times = np.array(results['positions'][2])
            transit_times =[times[np.logical_and(times>0,planet_mask==N)] for N in range(2)]

                
            epochs1 = []
            for ii in range(0, len(transit_times[0])):
                epochs1.append(ii)
            epochs1 = np.array(epochs1)



            epochs2 = []
            for ii in range(0, len(transit_times[1])):
                epochs2.append(ii)
            epochs2 = np.array(epochs2)


            #make sure at least 50 epochs
            if len(epochs1) >= 50:
                
            
                # Setup Pgrid
                Pmin = 2
                Pmax = 2*(np.max(epochs1)-np.min(epochs1))
                fmin = 1/Pmax
                fmax = 1/Pmin


                fgrid = np.linspace(fmin, fmax, 10*len(epochs1))
                Pgrid = np.sort(1/fgrid)

                
                #print('n epochs 1, ' + str(len(transit_times[0])))

                #plt.plot(epochs1, transit_times[0], 'ko')
                #plt.show()
            
                # if np.linalg.LinAlgError then orbit unstable...
                try:
                    pperts_out.append(ppert)

                    #define some homoschedastic error for the LS fit
                    times_err = np.full(transit_times[0].shape, 0.01)
                    times_err2 = np.full(transit_times[1].shape, 0.01)


                    #fit linear ephemeris
                    _, Ptran_lin = solve_null_model(epochs1, transit_times[0], times_err)
                    _, Ppert_lin = solve_null_model(epochs2, transit_times[1], times_err2)

                    #optimize for periodic TTVs
                    optimized_params = fit_multiple_frequencies(epochs1, transit_times[0], times_err, Pgrid, max_waves=1)
                    #T0_optimized, P_optimized, *fourier_params_optimized, P_LS_ttv = optimized_params


                    lstsq_ptrans.append(Ptran_lin)
                    lstsq_ppert.append(Ppert_lin)
                    optimized_params_list.append(optimized_params)
                    transit_times_trans.append(list(transit_times[0]))
                    transit_times_pert.append(list(transit_times[1]))

                except np.linalg.LinAlgError:
                    #plt.plot(epochs1, transit_times[0])
                    #plt.show()

                    #plt.plot(epochs2, transit_times[1])
                    #plt.show()
                    pass







         

        ppert_over_ptrans = list(np.array(pperts_out)/ptrans)

        ppert_over_ptrans_all.append(ppert_over_ptrans)
        optimized_params_all.append(optimized_params_list)
        transit_times_trans_all.append(list(transit_times_trans))
        transit_times_pert_all.append(list(transit_times_pert))
        lstsq_ptrans_all.append(lstsq_ptrans)
        lstsq_ppert_all.append(lstsq_ppert)
        orbital_params_all.append([mass1, mass2,
                                   ecc1, ecc2, 
                                   omega1, omega2,
                                   mean_anomaly1, mean_anomaly2])
        

    # Get today's date
    today_date = datetime.now().strftime('%Y%m%d')

    # Create a folder with today's date in the current directory
    folder_path_base = os.path.join(os.getcwd(), today_date + '_1')
    folder_path = folder_path_base

    # Check if the folder already exists and increment a counter if needed
    counter = 2
    while os.path.exists(folder_path):
        folder_path = f'{folder_path_base[:-2]}_{counter}'  # Remove the last "_1" and add the new counter
        counter += 1

    # Create the folder
    os.makedirs(folder_path, exist_ok=True)

    print(f'Folder created: {folder_path}')


    # Writing data to CSV using the function
    write_to_csv('pperts.csv', ppert_over_ptrans_all, folder_path)
    write_to_csv('optimized_params.csv', optimized_params_all, folder_path)
    write_to_csv('transit_times_trans.csv', transit_times_trans_all, folder_path)
    write_to_csv('transit_times_pert.csv', transit_times_pert_all, folder_path)
    write_to_csv('lstsq_ptrans.csv', lstsq_ptrans_all, folder_path)
    write_to_csv('lstsq_ppert.csv', lstsq_ppert_all, folder_path)
    write_to_csv('orbital_params.csv', orbital_params_all, folder_path)




