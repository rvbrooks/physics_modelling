"""
Vincent Brooks
    *** Monte Carlo fitting of statistical thermodynamic model to a release and recapture measurement 
     of a single atom in an optical tweezer trap. ***
     
     For a theoretical discussion of the physics, see my PhD thesis here:
         http://etheses.dur.ac.uk/14468/1/RVB_Thesis.pdf?DDD25+

    - This script is example Monte Carlo simulation code for the measurement of the temperature of an 
      atom in an optical tweezer laser trap which I developed during my PhD.
    - It fits a Monte Carlo simulation to measured data points in order to extract the temperature of
      the atom in the trap.
    1. The measurement is as follows: the laser trapping the atom is turned off for a variable time 
       (usually a few microseconds). If the atom is cold (moving slowly) it is usually recaptured when the 
       trap turns back on. If it is hot (moving fast), it is usually lost.
    2. The outcome is binary (lost or not lost), so we repeat the experiment many 100 times to get a probability.
    3. The probabilty is measured as a function of the trap off time and saved as a csv.
    
    - This script imports that csv file and fits the Monte Carlo simulation.
    
    - The Monte Carlo simulation is a thermodynamic simulation of the experiment described above. By varying the 
      temperature of the atom in the simulation, it is possible to fit to the data and determine a temperature.

"""

import numpy as np                # numpy for array operations
import pandas as pd               # pandas for dataframe manipulations
import matplotlib.pyplot as plt   # matplotlib for plotting
import os                         # os for setting the filepaths
import lmfit                      # lmfit for fitting of advanced functions (Monte Carlo)

########### Physical constants needed for the model ##########################
kB        = 1.38e-23              # Boltzman's constant in international scientific (SI) units.
amu       = 1.66053e-27           # Mass of a proton (kg)
h_plank   = 6.626e-34             # Plank's constant in SI units.
g         = -9.81                 # Gravitational acceleration in SI units 
epsilon_0 = 8.854e-12             # Permittivity of free space in SI units
c         = 299792458             # Speed of light (metres per second)
bohr_rad  = 5.29E-11              # Bohr radius (metres)
##############################################################################

np.random.seed(5) # reproducibly pseudo-random.

def import_data(directory, filename):
    """
    Import data for a temperature measurement experiment form a .csv file.
    The .csv has been generated as the aggregate output of many camera images.
    The columns are:
        1. Release Time (time the laser trap is off)
        2. Probability of catching the atom again when the trap is turned back on
        3. Error on the probability.
    """
    data = pd.read_csv(directory+"\\"+filename)
    t = data["release_time"]
    P = data["recapture_probability"]
    e = data["error_probability"]
    return((t, P, e))

class TemperatureMonteCarloModel():
    """
    Class for fitting Monte Carlo simulations to data.
    Arguments to initialise class:
        - species: initialise the model for Rubidium (Rb) atoms or Caesium (Cs) atoms.
                   (the atom mass strongly affects the results)
        - num_sims: the number of Monte Carlo iterations to do

    """
    def __init__(self, species, num_sims, laser_wavelength, beam_waist, laser_power, polarisibility):
        
        self.num_sims         = num_sims            # how many Monte Carlo simulations per point.
        self.lm               = laser_wavelength    # Light wavelength, m.
        self.wr               = beam_waist          # Beam focus size, m.
        self.power            = laser_power         # Beam power in mW
        self.polarisibility   = polarisibility      # Pol in a0^3

        if species == 'Cs': # set the mass of the atom to that of Caesium133
            self.m = 133 * amu

        elif species == 'Rb':  # set the mass of the atom to that of Rubidium87
            self.m = 87 * amu

        self.initialize_trap_frequencies()
        
    def get_trap_depth(self, waist, power, polarisibility):
        """
            Calculate the depth of the laser trap in SI units and in thermal units.
            Input arguments:
                - Trap radial waist at focus (in nanometre units)
                - Trap laser power (in milliWatt units)
                - Polarisability parameter of the atom in the laser trap (in a0^3 units)
            Output:
                - Sets the trap depth U0 in SI and thermal units.
        """
        intensity = 2 * power * 1e-3 / (np.pi * waist **2)
        polarisibilitySI = polarisibility*4*np.pi*epsilon_0*bohr_rad**3
        trapDepth_SI = np.abs(-0.5*polarisibilitySI*intensity / (c * epsilon_0))
        self.trapDepth_MHz = trapDepth_SI / h_plank / 1e6
        self.trapDepth_mK = trapDepth_SI / kB * 1000
        self.U0 = trapDepth_SI

    def initialize_trap_frequencies(self):
        """
           Given the temperature, wavelength, waist and trap power, we calculate
           trap frequencies (frequency of atom oscillation in the trap) which are
           used in the simulation.
        """

        self.get_trap_depth(self.wr, self.power, self.polarisibility)
        self.gf = (self.lm**2)/(2*(np.pi*self.wr)**2)                # Geometric factor between radial and axial directions.
        self.omega_r = np.sqrt(self.U0 * 4 / (self.m*(self.wr)**2))  # Trap frequency in radial direction
        self.omega_z = np.sqrt(self.gf)*self.omega_r                 # Trap frequency in axial direction

    def get_potential_energy(self, x,y,z):
        """This function returns the potential energy as a function of x,y and z """
        wrz = self.wr*np.sqrt(1 + (z*self.lm/(np.pi*self.wr**2))**2)
        return self.U0*((self.wr/wrz)**2)*np.exp(-2*(x**2+y**2)/wrz**2)    
    
    def randx(self, T):
        """Select random x/y position from thermal distribution"""
        watr = 2*np.sqrt(kB*T/(self.m*(self.omega_r)**2)) # width of atomic spatial probability distribution in x and y direction
        return np.random.normal(loc = 0.0, scale = watr/2, size = None)
    
    def randz(self, T):
        """Select random z position from thermal distribution"""
        watz = 2*np.sqrt(kB*T/(self.m*(self.omega_z)**2)) # width of atomic spatial probability distribution in z direction
        return  np.random.normal(loc = 0.0, scale = watz/2, size = None)
    
    def randv(self, T):
        """Select random velocity from thermal distribution"""
        return np.random.normal(loc = 0.0, scale = np.sqrt(kB*T/self.m), size = None)
    
    def simulate_atom_flight(self, atom_temperature, release_time):
        """
            - Simulate the behaviour of an atom in the laser trap once it is released.
            - Calculate the new position after a release time t and determine if 
              the atom has escaped the laser trap or not.
        """
        Ts = np.ones(self.num_sims)*atom_temperature  # Perform efficient vectorised computation on n
        x0 = self.randx(Ts)
        y0 = self.randx(Ts)
        z0 = self.randz(Ts)
        vx0 = self.randv(Ts)
        vy0 = self.randv(Ts)
        vz0 = self.randv(Ts)
        #Final state
        xt = x0 + vx0 * release_time
        yt = y0 + vy0 * release_time
        zt = z0 + vz0 * release_time - 0.5 * g * release_time**2 # includes gravity
        vxt = vx0
        vyt = vy0 
        vzt = vz0 + g * release_time # includes acceleration due to gravity
        
        #Compare with potential energy and see if the particle has enough escape energy. 
        Ut = self.get_potential_energy(xt,yt,zt)
        number_caught = len(np.where(Ut > 0.5 * self.m * ((vxt**2) + (vyt**2) + (vzt**2)))[0])
        recaptured_fraction = number_caught/self.num_sims  
        
        return(recaptured_fraction)

    def run_simulation(self, amplitude, temperature, times):
        """
        Apply the model to many simulations in a vectorised array for a given simulated temperature.
        Inputs:
            - amplitude: amplitude of the fit.
            - temperature: temperature to run the simulation for
            - times: x values to evaluate the simulation over
        """

        prob_array = amplitude * np.array(list(map(lambda time: self.simulate_atom_flight(temperature, time*1e-6), times))) 
        
        return(prob_array)
           
    def get_residual(self, params, x, y, y_error):
        """Calculate the residual error between the measured data points and
           the Monte Carlo fit. 
           Inputs:
               - params: the temperature and amplitude fitting parameters to optimise
               - x: the time data
               - y: the recapture probability
               - y_error: the error on the recapture probability
           Returns:
               - The residual error.
           """
        amplitude_param = params['amplitude'].value
        temperature_param = params['temperature'].value
        
        model = self.run_simulation(amplitude_param, temperature_param, x)
        residual_error = (model - y) / y_error
        return(residual_error)
    
    def fit_mc(self, data = (1, 1, 0.1), temp_guess=19e-6, amp_guess=1., T_vary=True, amp_vary=False):
        """Fit the measured data with the Monte Carlo simulation.
            Inputs:
                - data=(x, y, y_error): measured data as a tuple.
                - temp_guess: initial estimate of the atom temperature
                - amp_guess: initial estimate of the fit amplitude
                - T_vary: boolean if the temperature should be a variable fit parameter
                - amp_vary: boolean if the amplitude should be a variable fit parameter
            Outputs:
                - A itted Minimizer object containing fitted temperature and amplitude
                  where the residual has been minimized.
        """
        
        self.data = data

        # Define the least-squares fitting parameters we want to optimise for
        # temperature: the simulated temperature of the atom
        # amplitude: the max value of the fitted line (usually 1)
        params = lmfit.Parameters()
        params.add('temperature', value = temp_guess,  min=0e-6, max=500e-6,  vary=T_vary)
        params.add('amplitude', value = amp_guess,  min=0.1, max=1.0,     vary=amp_vary)
        
        # Perform minimizartion of the residual to fit the model to the data:
        minimizer = lmfit.Minimizer(self.get_residual, params, fcn_args=(data[0], data[1], data[2]))
        
        self.results = minimizer.minimize(method='nelder', tol = 0.5)
    
    def plot_mc_results(self):
        """
        Plot the results of the optimised Monte Carlo fit against the data and print
        the temperature extracted and the uncertainty on the value.
        
        The uncertainty is derived from the covariance matrix.
        """
        
        x, y, y_error = self.data
        
        fitted_amplitude = self.results.params["amplitude"]
        fitted_temperature = self.results.params["temperature"]
        
        xfit = np.linspace(0, 110, 200)
        yfit = self.run_simulation(fitted_amplitude, fitted_temperature, xfit)
        
        plt.close('all')
        plt.plot(xfit, yfit) # color = '#006388'
        plt.gca().set_prop_cycle(None)
        plt.ylim(-0.04, 1.05)
        plt.errorbar(x, y, yerr = y_error, marker = 'o', linestyle = 'none') # color = '#7E317B'
        plt.xlabel('$\mathrm{Release \ time \ ( \mu s)}$', fontsize=14)
        plt.ylabel('$\mathrm{Recapture \ probability}$', fontsize=14)        

        final_temperature = np.around(fitted_temperature.value*1e6, 2)
        error_temperature = np.around(fitted_temperature.stderr*1e6, 2)
        
        plt.title("Extracted Temperature = "+str(final_temperature)+" $\pm$ "+str(error_temperature)+" $\mu$K", fontsize=12)

        print("The temperature extracted by fitting the thermodynamical Monte Carlo model is:")
        print("("+str(final_temperature)+" ± "+str(error_temperature)+")", "microKelvin")

if __name__ == "__main__":
    
    # Instantiate the Monte Carlo model:
    model = TemperatureMonteCarloModel(species = 'Rb',
                                       num_sims = 5000,                
                                       laser_wavelength = 814e-9,  # nanometres
                                       beam_waist = 927.5e-9,      # nanometres
                                       laser_power = 1.596,        # mW
                                       polarisibility = 4760,      # a0^3
                                       )
    
    # work in the directory of the scipt.
    script_dir = os.path.dirname(__file__)
    
    # import measured data (taken by the experiment I built)
    measured_data = import_data(script_dir, "temperature_data.csv")
    
    # fit the Monte Carlo model to the data
    model.fit_mc(measured_data, temp_guess=40e-6) 
    
    # plot the fitted model against the measured data points.
    # using the custom rose_pine matplotlib style in the script directory.
    with plt.style.context(script_dir+"/plt_styles/rose_pine.mplstyle"):
        model.plot_mc_results()
    
    plt.savefig("monte_carlo_result.png")
    plt.show()
    






