### This script takes as input the dataframes produced in the previous step (buildDataset), computes train weights to compensate for different
### statistics for each signal mass value and for signal/background, scales each feature according to median and interquartile range 
### and splits the resulting dataframe in train and test samples.
### Histograms of scaled and unscaled variables will be saved if running with --drawPlots = 1

import seaborn
from Functions import ReadArgParser, checkCreateDir, ReadConfig, DrawVariablesHisto, ShufflingData, ComputeTrainWeights, ScalingFeatures, ComputeScaleFactors#, SaveFeatureScaling,
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
import shlex
from termcolor import colored, cprint

#print(sys.executable, ' '.join(map(shlex.quote, sys.argv)))
#from tensorflow.keras.utils import to_categorical

pd.options.mode.chained_assignment = None ### to suppress the SettingWithCopyWarning

### Reading from command line
tag, analysis, channel, preselectionCuts, background, signal, trainingFraction, drawPlots = ReadArgParser()

### Reading from configuration file
dfPath, signalsList, backgroundsList = ReadConfig(tag, analysis, signal) ### remove input features

### Loading input file
inputDir = dfPath + analysis + '/' + channel + '/' + preselectionCuts + '/' + signal + '/' + background
#inputDir = dfPath + analysis + '/' + channel + '/' + preselectionCuts + '/' + 'ggFandVBF/' + signal + '/' + background
fileCommonName = tag + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + signal + '_' + background
cprint('Loading ' + inputDir + '/MixData_' + fileCommonName + '.pkl', 'green')
#quit()

data = pd.read_pickle(inputDir + '/MixData_' + fileCommonName + '.pkl') 

### If not already existing, creating output directory
outputDir = inputDir# + '/PDNN_trainSet9'# + '/ggFsameStatAsVBF'
checkCreateDir(outputDir)
fileCommonName += '_' + str(trainingFraction) + 't'

### Creating log file
logFileName = '/logFile_splitDataset_' + fileCommonName + '.txt'
logFile = open(outputDir + logFileName, 'w')
logFile.write('Command executed: ' + sys.executable + ' '.join(map(shlex.quote, sys.argv)) + '\nTag: ' + tag + '\nAnalysis: ' + analysis + '\nChannel: ' + channel + '\nPreselection cuts: ' + preselectionCuts + '\nSignal: ' + signal + '\nBackground: ' + background)

### Creating the list of backgrounds and signal processes to select
if background == 'all':
    inputOrigins = backgroundsList.copy()
    backgroundLegend = backgroundsList.copy()
    logFile.write(' (' + str(backgroundsList) + ')')
else:
    inputOrigins = list(background.split('_'))
inputOrigins.append(signal)
    
### Selecting events according to their origin 
data_set = data[data['origin'].isin(inputOrigins)]

### Dividing signal from background
dataSetSignal = data_set[data_set['origin'] == signal]
dataSetBackground = data_set[data_set['origin'] != signal]

massesSignalList = sorted(list(set(list(dataSetSignal['mass']))))
cprint('Masses in the signal sample: ' + str(massesSignalList) + ' GeV (' + str(len(massesSignalList)) + ')', 'blue')
logFile.write('\nMasses in the signal sample: ' + str(massesSignalList) + ' GeV (' + str(len(massesSignalList)) + ')')
'''
statDict = {}
statDict['Radion'] = {500: 2587, 600: 7075, 700: 10780, 800: 13074, 1000: 13241, 1200: 11704, 1400: 11451, 1500: 11811, 1600: 12185, 1800: 13458, 2000: 14873, 2400: 17913, 2600: 18750, 3000: 19480, 3500: 17941, 4000: 15948, 4500: 14542, 5000: 13232, 6000: 12604}
statDict['RSG'] = {500: 1622, 600: 5964, 700: 5964, 800: 7282, 1000: 7329, 1200: 7208, 1400: 7431, 1500: 7549, 1600: 7620, 1800: 8410, 2000: 9452, 2400: 11098, 2600: 11505, 3000: 11699, 3500: 11033, 4000: 9492, 4500: 8321, 5000: 7532, 6000: 7532}
statDict['HVTWZ'] = {500: 0, 600: 2735, 700: 4526, 800: 6452, 1000: 7293, 1200: 6853, 1400: 0, 1500: 6886, 1600: 0, 1800: 8544, 2000: 9401, 2400: 7708, 2600: 11763, 3000: 12711, 3500: 12100, 4000: 11888, 4500: 0, 5000: 0, 6000: 0}
statBkgDict = {}
statBkgDict['Radion'] = {'Zjets': 41611, 'Diboson': 2603, 'stop': 8, 'ttbar': 272, 'Wjets': 16} 
statBkgDict['RSG'] = {'Zjets': 41292, 'Diboson': 2602, 'stop': 8, 'ttbar': 272, 'Wjets': 16} 
statBkgDict['HVTWZ'] = {'Zjets': 41611, 'Diboson': 2603, 'stop': 8, 'ttbar': 272, 'Wjets': 16} 

newDataSetSignal = []

for mass in massesSignalList:
    dataSetSignalMass = dataSetSignal[dataSetSignal['mass'] == mass]
    if mass not in statDict[signal]:
        continue
    statMass = statDict[signal][mass]
    dataSetSignalMass = dataSetSignalMass[:statMass]
    newDataSetSignal.append(dataSetSignalMass)
dataSetSignal = pd.concat(newDataSetSignal, ignore_index = True)
dataSetSignal = ShufflingData(dataSetSignal)

newDataSetBkg = []
for bkg in backgroundsList:
    dataSetBkgOrigin = dataSetBackground[dataSetBackground['origin'] == bkg]
    statBkg = statBkgDict[signal][bkg]
    dataSetBkgOrigin = dataSetBkgOrigin[:statBkg]
    newDataSetBkg.append(dataSetBkgOrigin)
dataSetBackground = pd.concat(newDataSetBkg, ignore_index = True)
dataSetBackground = ShufflingData(dataSetBackground)

print(dataSetSignal.shape)
print(dataSetBackground.shape)
'''
### Creating new column in the dataframes with train weight
dataSetSignal, dataSetBackground, logString = ComputeTrainWeights(dataSetSignal, dataSetBackground, massesSignalList, outputDir, fileCommonName, analysis, channel, signal, backgroundLegend, preselectionCuts, drawPlots)
if logString != '':
    cprint(logString, 'green')
    logFile.write(logString)

### Printing and saving information on events numbers 
stringForBkgEvents = 'Number of background events: ' + str(dataSetBackground.shape[0]) + ' raw, ' + str(dataSetBackground['weight'].sum()) + ' with MC weights, ' + str(dataSetBackground['train_weight'].sum()) + ' with train weights'
cprint(stringForBkgEvents, 'blue')
logFile.write('\n' + stringForBkgEvents)

if(len(backgroundsList) > 1):
    for bkg in backgroundsList:
        dataSetSingleBackground = dataSetBackground[dataSetBackground['origin'] == bkg]
        stringToSaveBkg = '   ---> Number of ' + bkg + ' events: ' + str(dataSetSingleBackground.shape[0]) + ' raw, ' + str(dataSetSingleBackground['weight'].sum()) + ' with MC weights, ' + str(dataSetSingleBackground['train_weight'].sum()) + ' with train weights'
        cprint(stringToSaveBkg, 'blue')
        logFile.write('\n' + stringToSaveBkg)

stringForSignalEvents = 'Number of signal events: ' + str(dataSetSignal.shape[0]) + ' raw, ' + str(dataSetSignal['weight'].sum()) + ' with MC weights, ' + str(dataSetSignal['train_weight'].sum()) + ' with train weights'
cprint(stringForSignalEvents, 'blue')
logFile.write('\n' + stringForSignalEvents)
for signalMass in massesSignalList:
    dataSetSignalMass = dataSetSignal[dataSetSignal['mass'] == signalMass]
    stringToSaveSignal = '   ---> Number of signal events with mass ' + str(signalMass) + ' GeV: ' + str(dataSetSignalMass.shape[0]) + ' raw, ' + str(dataSetSignalMass['weight'].sum()) + ' with MC weights, ' + str(dataSetSignalMass['train_weight'].sum()) + ' with train weights'
    cprint(stringToSaveSignal, 'blue')
    logFile.write('\n' + stringToSaveSignal)

### Concatening signal and background dataframes
dataFrame = pd.concat([dataSetSignal, dataSetBackground], ignore_index = True)

### Shuffling the dataframe
dataFrame = ShufflingData(dataFrame)

### Creating a new column in the dataFrame that will store the unscaled mass
dataFrame = dataFrame.assign(unscaledMass = dataFrame['mass'])

### Splitting data into train and test set
data_train, data_test = train_test_split(dataFrame, train_size = trainingFraction)

### Printing and saving information on events numbers 
data_train_bkg = data_train[data_train['origin'] != signal]
data_train_signal = data_train[data_train['origin'] == signal]
data_test_bkg = data_test[data_test['origin'] != signal]
data_test_signal = data_test[data_test['origin'] == signal]

stringForTrainBkgEvents = 'Number of background events in train sample: ' + str(data_train_bkg.shape[0]) + ' raw, ' + str(data_train_bkg['weight'].sum()) + ' with MC weights, ' + str(data_train_bkg['train_weight'].sum()) + ' with train weights'
cprint(stringForTrainBkgEvents, 'blue')
logFile.write('\n' + stringForTrainBkgEvents)

if(len(backgroundsList) > 1):
    for bkg in backgroundsList:
        dataTrainSingleBackground = data_train_bkg[data_train_bkg['origin'] == bkg]
        stringToSaveSingleBkg = '   ---> Number of ' + bkg + ' events: ' + str(dataTrainSingleBackground.shape[0]) + ' raw, ' + str(dataTrainSingleBackground['weight'].sum()) + ' with MC weights, ' + str(dataTrainSingleBackground['train_weight'].sum()) + ' with train weights'
        cprint(stringToSaveSingleBkg, 'blue')
        logFile.write('\n' + stringToSaveSingleBkg)

stringForTrainSignalEvents = 'Number of signal events in train sample: ' + str(data_train_signal.shape[0]) + ' raw, ' + str(data_train_signal['weight'].sum()) + ' with MC weights, ' + str(data_train_signal['train_weight'].sum()) + ' with train weights'
cprint(stringForTrainSignalEvents, 'blue')
logFile.write('\n' + stringForTrainSignalEvents)
for signalMass in massesSignalList:
    dataTrainSignalMass = data_train_signal[data_train_signal['mass'] == signalMass]
    stringToSaveSignal = '   ---> Number of signal events with mass ' + str(signalMass) + ' GeV: ' + str(dataTrainSignalMass.shape[0]) + ' raw, ' + str(dataTrainSignalMass['weight'].sum()) + ' with MC weights, ' + str(dataTrainSignalMass['train_weight'].sum()) + ' with train weights'
    cprint(stringToSaveSignal, 'blue')
    logFile.write('\n' + stringToSaveSignal)

stringForTestBkgEvents = 'Number of background events in test sample: ' + str(data_test_bkg.shape[0]) + ' raw, ' + str(data_test_bkg['weight'].sum()) + ' with MC weights, ' + str(data_test_bkg['train_weight'].sum()) + ' with train weights'
cprint(stringForTestBkgEvents, 'blue')
logFile.write('\n' + stringForTestBkgEvents)

if(len(backgroundsList) > 1):
    for bkg in backgroundsList:
        dataTestSingleBackground = data_test_bkg[data_test_bkg['origin'] == bkg]
        stringToSaveSingleBkg = '   ---> Number of ' + bkg + ' events: ' + str(dataTestSingleBackground.shape[0]) + ' raw, ' + str(dataTestSingleBackground['weight'].sum()) + ' with MC weights, ' + str(dataTestSingleBackground['train_weight'].sum()) + ' with train weights'
        cprint(stringToSaveSingleBkg, 'blue')
        logFile.write('\n' + stringToSaveSingleBkg)

stringForTestSignalEvents = 'Number of signal events in test sample: ' + str(data_test_signal.shape[0]) + ' raw, ' + str(data_test_signal['weight'].sum()) + ' with MC weights, ' + str(data_test_signal['train_weight'].sum()) + ' with train weights'
print(stringForTestSignalEvents, 'blue')
logFile.write('\n' + stringForTestSignalEvents)
for signalMass in massesSignalList:
    dataTestSignalMass = data_test_signal[data_test_signal['mass'] == signalMass]
    stringToSaveSignal = '   ---> Number of signal events with mass ' + str(signalMass) + ' GeV: ' + str(dataTestSignalMass.shape[0]) + ' raw, ' + str(dataTestSignalMass['weight'].sum()) + ' with MC weights, ' + str(dataTestSignalMass['train_weight'].sum()) + ' with train weights'
    cprint(stringToSaveSignal, 'blue')
    logFile.write('\n' + stringToSaveSignal)

'''
### Slicing data train for statistic test
nEvents = int(data_train.shape[0] / 4)
data_train = data_train[:nEvents]
'''

### Scaling InputFeatures of train and test set
logString = ComputeScaleFactors(data_train, outputDir)
'''
data_train, data_test, logString = ScalingFeatures(data_train, data_test, InputFeatures, outputDir)
'''
logFile.write(logString)

'''
if drawPlots:
    scaledHistogramsPath = outputDir + '/trainScaledHistograms'
    print(format('Output directory: ' + Fore.GREEN + scaledHistogramsPath), checkCreateDir(scaledHistogramsPath))
    logFile.write('\nSaving histograms of the scaled train dataset in ' + scaledHistogramsPath)
    DrawVariablesHisto(data_train, InputFeatures, scaledHistogramsPath, fileCommonName, jetCollection, analysis, channel, signal, backgroundLegend, preselectionCuts, True)
'''

### Saving unscaled dataframes
dataTrainName = outputDir + '/data_train_' + fileCommonName + '.pkl'
data_train.to_pickle(dataTrainName)
cprint('Saved ' + dataTrainName, 'green')
logFile.write('\nSaved unscaled train dataframe in ' + dataTrainName)
dataTestName = outputDir + '/data_test_' + fileCommonName + '.pkl'
data_test.to_pickle(dataTestName)
cprint('Saved ' + dataTestName, 'green')
logFile.write('\nSaved unscaled test dataframe in ' + dataTestName)

### Closing log file
logFile.close()
cprint('Saved ' + outputDir + logFileName, 'green')
