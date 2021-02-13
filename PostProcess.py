import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

class Turbine(object):

    def __init__(self, turbName):
        self.turbName = turbName
        self.rotor_D = None
        self.tower_H = None
        self.blade_r = None
        self.blade_c = None
        self.blade_twist = None

    def readTurbineProperties(self, turbinePropDir, turbineName):
        '''
        Fill the turbine properties needed
        '''
        if turbinePropDir is None:
            turbinePropDir = "./constant/turbineProperties/"
        if turbineName is None:
            turbineName = 'DTU10MW_POLIMI_WTM'

        # Read turbine properties
        with open(turbinePropDir+turbineName, "r") as turbProp:
            flag = False
            blade_data = []
            numeric_pattern = '[+-]?\d*\.\d+ [+-]?\d*\.\d+ [+-]?\d*\.\d+ [+-]?\d*\.\d+ [+-]?\d*\.\d+ [+-]?\d+ '
            for line in turbProp:
                if 'TipRad' in line:
                    self.rotor_D = 2 * float(re.findall('\d+\.?\d*', line)[0])
                elif 'TowerHt' in line:
                    self.tower_H = float(re.findall('\d+\.?\d*', line)[0])
                elif 'BladeData' in line:
                    flag = True
                elif 'TowerData' in line:
                    flag = False
                if flag:
                    blade_data.append(re.findall(numeric_pattern, line))
        # remove empty elements from blade_data
        blade_data = list(filter(None, blade_data))
        # split sublists
        for i in range(0, len(blade_data)):
            dummy = [el.split() for el in blade_data[i]]
            blade_data[i] = dummy
        # convert to numpy array
        blade_data = np.array(blade_data, dtype=float).reshape(len(blade_data), 6)
        self.blade_r = blade_data[:, 0]
        self.blade_c = blade_data[:, 1]
        self.blade_twist = blade_data[:, 2]

class Probes(Turbine):

    def __init__(self, turbName, probeName, turbinePropDir=None, turbineName=None):
        self.probName = probeName
        Turbine.__init__(self, turbName)
        Turbine.readTurbineProperties(self, turbinePropDir, turbineName)

    def makeProbes(self, probeDir='./', probeSets=1, outCtrl='timeStep', outInt=1, tStart=None, tEnd=None, fields=None, x=np.array([(1, 1)]), y=np.array([(1, 1)]), z=np.array([(1, 1)]), nProbes=None, stepProbes=None):

        # Fill default parameters
        if nProbes is None:
            nProbes = np.ones((probeSets, 3))
        elif stepProbes is None:
            stepProbes = np.zeros((probeSets, 3))

        # Change parameter type
        x = x.astype(np.float)
        y = y.astype(np.float)
        z = z.astype(np.float)
        nProbes = nProbes.astype(np.int)
        stepProbes = stepProbes.astype(np.float)

        # Check for directory or make it
        if not os.path.isdir(probeDir):
            print('Making directory: ' + probeDir)
            os.mkdir(probeDir)

        # Check for field ( fields=['u', 'p', ....] )
        if fields is None:
            raise NameError('FieldListError')

        # Open the file and write
        with open(probeDir + str(self.probName), 'w') as file:
            file.write("{}".format(self.probName))
            file.write("\n{")
            file.write("\n{:4}{:<30}{}".format('', 'type', 'probes;'))
            file.write("\n{:4}{:<30}{}".format('', 'functionObjectLibs', '("libsampling.so");'))
            file.write("\n{:4}{:<30}{}".format('', 'enabled', 'true;'))
            file.write("\n{:4}{:<30}{}{}".format('', 'probName', self.probName, ';'))
            file.write("\n{:4}{:<30}{}{}".format('', 'outputControl', outCtrl, ';'))
            file.write("\n{:4}{:<30}{}{}".format('', 'outputInterval', int(outInt), ';'))
            if tStart and tEnd is not None:
                file.write("\n{:4}{:<30}{}{}".format('', 'timeStart', int(tStart), ';'))
                file.write("\n{:4}{:<30}{}{}".format('', 'timeEnd', int(tEnd), ';'))
            file.write("\n\n{:4}{}".format('', 'fields'))
            file.write("\n{:4}{}".format('', '('))
            for field in fields:
                file.write("\n{:8}{}".format('', field))
            file.write("\n{:4}{}".format('', ');'))
            file.write("\n\n{:4}{}".format('', 'probeLocations'))
            file.write("\n{:4}{}".format('', '('))
            # Write Probe Locations
            minimum = np.zeros((probeSets, 3))
            maximum = np.zeros((probeSets, 3))
            iterValue = np.zeros((probeSets, 3))
            for i in range(0, probeSets):
                count_n = 0
                minimum[i, 0] = x[i, 0]; minimum[i, 1] = y[i, 0]; minimum[i, 2] = z[i, 0]
                maximum[i, 0] = x[i, 1]; maximum[i, 1] = y[i, 1]; maximum[i, 2] = z[i, 1]
                iterValue[i, 0] = x[i, 0]; iterValue[i, 1] = y[i, 0]; iterValue[i, 2] = z[i, 0]
                if not (nProbes[i, :] == np.ones((1, 3))).all():
                    step = np.zeros(3)
                    step[0] = (maximum[i, 0] - minimum[i, 0]) / nProbes[i, 0]
                    step[1] = (maximum[i, 1] - minimum[i, 1]) / nProbes[i, 1]
                    step[2] = (maximum[i, 2] - minimum[i, 2]) / nProbes[i, 2]
                    while (iterValue[i, :] <= maximum[i, :]).all() and (count_n <= max(nProbes[i, :])):
                        file.write("\n{:8}{} {:f} {:f} {:f} {}".format('', '(', iterValue[i, 0], iterValue[i, 1], iterValue[i, 2], ')'))
                        iterValue[i, :] += step
                        count_n += 1
                    print("{} probes in set {} ".format(count_n, i+1))
                if not (stepProbes[i, :] == np.zeros((1, 3))).all():
                    while (iterValue[i, :] <= maximum[i, :]).all():
                        file.write("\n{:8}{} {:f} {:f} {:f} {}".format('', '(', iterValue[i, 0], iterValue[i, 1], iterValue[i, 2], ')'))
                        iterValue[i, :] += stepProbes[i, :]
                        count_n += 1
                    print("{} probes in set {} ".format(count_n, i+1))
            file.write("\n{:4}{}".format('', ');'))
            file.write("\n}")

    def readProbes(self, postProcDir=None):
        '''
        Update vector and scalar field lists
        '''

        if postProcDir is None:
            postProcDir = './postProcessing/'
        probeDir = postProcDir + self.probName + '/0/'
        files = os.listdir(probeDir)

        # Read probes locations
        with open(probeDir + files[0], 'r') as loc:
            numericPattern = '([+-]?\d+\.?\d* [+-]?\d+\.?\d* [+-]?\d+\.?\d*)'
            probeLoc = []
            n_line = 1
            for line in loc:
                probeLoc.append(re.findall(numericPattern, line))
                if 'Probe' not in line:
                    break
        probeLoc = probeLoc[:-2]
        # split sublists
        for i in range(0, len(probeLoc)):
            dummy = [el.split() for el in probeLoc[i]]
            probeLoc[i] = dummy
        # convert to numpy array
        self.probeLoc = np.array(probeLoc, dtype=float).reshape(len(probeLoc), 3)

        # Number of probes
        self.nProbes = len(self.probeLoc)

        # Find number of probe sets and their dimensions
        nProbeSets = 0
        setsDim = []
        old = 0
        for i in range(2, self.nProbes):
            helpSets = 0
            for j in range(0, 3):
                if (self.probeLoc[i, j] != self.probeLoc[i-1, j]) or (self.probeLoc[i, j] != self.probeLoc[i-2, j]):
                    helpSets += 1
            if helpSets == 2 and old != i-1:
                nProbeSets += 1
                setsDim.append(int(i-old))
                old = i
            if i == self.nProbes-1:
                setsDim.append(int(i - old + 1))
        self.nProbeSets = nProbeSets + 1
        self.setsDim = np.array(setsDim)

        vector_fields = ['U', 'UMean']
        scalar_fields = ['p', 'pMean', 'nuSgs']
        for file in files:
            if file in scalar_fields:
                scalar_db = pd.read_csv(probeDir + file, sep='\s+', skiprows=self.nProbes + 2, header=None)
                values = scalar_db.to_numpy(dtype=float)
                self.time = values[:, 0]
                vars(self)[file] = values[:, 1:]

            elif file in vector_fields:
                vector_pattern = '\s+|\(|\)'
                vector_db = pd.read_csv(probeDir + file, sep=vector_pattern, skiprows=self.nProbes + 2,
                                        header=None, engine='python', keep_default_na=True)
                vector_db.dropna(how='all', axis=1, inplace=True)
                values = vector_db.to_numpy(dtype=float)
                self.time = values[:, 0]
                vars(self)[file+'x'] = values[:, 1::3]
                vars(self)[file+'y'] = values[:, 2::3]
                vars(self)[file+'z'] = values[:, 3::3]
            else:
                raise NameError('FieldTypeError')

    def plotEnergySpectrum(self):
        pass


    def plot2ptCorrelations(self):
        pass

    def plotTutbulenceIntensity(self):
        pass

    def plotProfile(self, plotDir=None, var='p'):

        if plotDir is None:
            plotDir = './postProcessing/' + self.probName + '/plots/'
        if not os.path.isdir(plotDir):
            os.mkdir(plotDir)

        # Plot every set of probes
        probeStart = 0
        for i in range(0, self.nProbeSets):
            probeEnd = probeStart + self.setsDim[i]
            x = self.probeLoc[probeStart:probeEnd, 0]
            y = self.probeLoc[probeStart:probeEnd, 1]
            z = self.probeLoc[probeStart:probeEnd, 2]
            # Find x axis for plot
            if x[1] != x[2]:
                ay = x/self.rotor_D
                ylabel = 'x/D'
            elif y[1] != y[2]:
                ay = y/self.rotor_D
                ylabel = 'y/D'
            elif z[1] != z[2]:
                ay = z/self.rotor_D
                ylabel = 'z/D'
            plt.figure()
            plt.plot(vars(self)[var][-1, probeStart:probeEnd], ay)
            plt.title(var+' profile')
            plt.xlabel(var)
            plt.ylabel(ylabel)
            plt.savefig(plotDir + var + str(i) + '.png')
            probeStart = probeEnd

class Slices(Turbine):

    def __init__(self):
        pass



class SOWFA(Turbine):

    def __init__(self, turbName, turbinePropDir=None, turbineName=None):
        Turbine.__init__(self, turbName)
        Turbine.readTurbineProperties(self, turbinePropDir, turbineName)

    def readTurbineOutput(self, turbineOutDir=None):

        if turbineOutDir is None:
            turbineOutDir = "./postProcessing/turbineOutput/0/"

        # Find files in the directory
        files = os.listdir(turbineOutDir)
        self.SOWFAturbine = []
        self.SOWFAblade = []

        # Read turbine output files
        for file in files:
            turb_db = pd.read_csv(turbineOutDir + file, sep=' ', skiprows=1, header=None)
            values = turb_db.values
            if values.shape[1] < 6:
                vars(self)[file] = values[:, -1]
                self.turbineTime = values[:, 1]
                self.SOWFAturbine.append(file)
            else:
                vars(self)[file+'Time'] = values[0:-1:3, 2]
                vars(self)[file] = values[0:-1:3, 4:-2]
                vars(self)[file+'Root'] = values[0:-1:3, 4]
                vars(self)[file+'Tip'] = values[0:-1:3, -2]
                vars(self)[file+'Ft'] = values[-1, 4:-2]
                blade_span = np.linspace(self.blade_r[0], self.blade_r[-1], num=len(vars(self)[file+'Ft']), dtype=float)
                self.blade_span_norm = blade_span/(self.rotor_D/2.0)
                self.SOWFAblade.append(file)

    def plotTurbine(self, plotDir=None, var='all'):

        if plotDir is None:
            plotDir = './postProcessing/turbineOutput/plots/'
        if not os.path.isdir(plotDir):
            os.mkdir(plotDir)

        if var == 'all':
            for file in self.SOWFAturbine:
                plt.figure()
                plt.plot(self.turbineTime, vars(self)[file])
                plt.title(file)
                plt.xlabel('time [s]')
                plt.ylabel(file)
                plt.savefig(plotDir+file+'_turbine.png')
                plt.close()
        else:
            plt.figure()
            plt.plot(self.turbineTime, vars(self)[var])
            plt.title(var)
            plt.xlabel('time [s]')
            plt.ylabel(var)
            plt.savefig(plotDir + var + '_turbine.png')
            plt.close()

    def plotBladeFinalTime(self, plotDir=None, var='all'):

        if plotDir is None:
            plotDir = './postProcessing/turbineOutput/plots/'
        if not os.path.isdir(plotDir):
            os.mkdir(plotDir)

        if var == 'all':
            for file in self.SOWFAblade:
                if 'blade' in file:
                    plt.figure()
                    plt.plot(self.blade_span_norm, vars(self)[file+'Ft'])
                    plt.title(file+' Final Time')
                    plt.xlabel('r/R')
                    plt.ylabel(file)
                    plt.savefig(plotDir+file+'_bladeFt.png')
                    plt.close()
        else:
            plt.figure()
            plt.plot(self.blade_span_norm, vars(self)[var+'Ft'])
            plt.title(var+' Final Time')
            plt.xlabel('r/R')
            plt.ylabel(var)
            plt.savefig(plotDir + var + '_bladeFt.png')
            plt.close()

    def plotBladeOverTime(self, plotDir=None, var='all', rootTip='together'):
        """
        Plot root and tip values over time
        """
        if plotDir is None:
            plotDir = './postProcessing/turbineOutput/plots/'
        if not os.path.isdir(plotDir):
            os.mkdir(plotDir)

        if var == 'all':
            for file in self.SOWFAblade:
                if 'blade' in file:
                    if rootTip == 'together':
                        plt.figure()
                        plt.plot(vars(self)[file + 'Time'], vars(self)[file + 'Root'])
                        plt.plot(vars(self)[file + 'Time'], vars(self)[file + 'Tip'])
                        plt.title(file)
                        plt.xlabel('time [s]')
                        plt.ylabel(file)
                        plt.legend(["Root", "Tip"])
                        plt.savefig(plotDir + file + '_blade.png')
                        plt.close()
                    elif rootTip == 'separate':
                        plt.figure()
                        plt.plot(vars(self)[file + 'Time'], vars(self)[file + 'Root'])
                        plt.title(file)
                        plt.xlabel('time [s]')
                        plt.ylabel(file)
                        plt.savefig(plotDir + file + '_bladeRoot.png')
                        plt.close()
                        plt.figure()
                        plt.plot(vars(self)[file + 'Time'], vars(self)[file + 'Tip'])
                        plt.title(file)
                        plt.xlabel('time [s]')
                        plt.ylabel(file)
                        plt.savefig(plotDir + file + '_bladeTip.png')
                        plt.close()

        else:
            if rootTip == 'together':
                plt.figure()
                plt.plot(vars(self)[var + 'Time'], vars(self)[var + 'Root'])
                plt.plot(vars(self)[var + 'Time'], vars(self)[var + 'Tip'])
                plt.title(var)
                plt.xlabel('time [s]')
                plt.ylabel(var)
                plt.legend(["Root", "Tip"])
                plt.savefig(plotDir + var + '_blade.png')
                plt.close()
            elif rootTip == 'separate':
                plt.figure()
                plt.plot(vars(self)[var + 'Time'], vars(self)[var + 'Root'])
                plt.title(var)
                plt.xlabel('time [s]')
                plt.ylabel(var)
                plt.savefig(plotDir + var + '_bladeRoot.png')
                plt.close()
                plt.figure()
                plt.plot(vars(self)[var + 'Time'], vars(self)[var + 'Tip'])
                plt.title(var)
                plt.xlabel('time [s]')
                plt.ylabel(var)
                plt.savefig(plotDir + var + '_bladeTip.png')
                plt.close()

    def plotBladeFrequency(self):
        pass

'''
class FAST(object):

    def __init__(self, probName):
        self.probName = probName
'''