import os

import numpy

from scipy.optimize import curve_fit

from matplotlib import pyplot as plt

def readAssociationData(filePath):

  with open(filePath, 'rU') as fp:
    fp.readline() # header
    data = []
    for line in fp:
      line = line.strip()
      if line:
        data.append(float(line))

  data = numpy.array(data)

  return data

def _fitInverseExponentials(xdata, *params):

  nexps = len(params) // 2
  params = list(params)

  ydata = numpy.zeros(len(xdata), dtype='float32')
  for i in range(nexps):
    ydata += params[i] * (1 - numpy.exp(-xdata*params[i+nexps]))

  return ydata

def _initialFitAssociationEstimate(ydata):
  
  # assumes ydata[-1] > 0 (in fact it is 1.0)
  a = ydata[-1]
  b = 0.0
  for m in range(min(len(ydata)-1, 10), 0, -1):
    if ydata[m] < ydata[-1]:
      b = numpy.log((ydata[-1]-ydata[0]) / (ydata[-1]-ydata[m])) / m
      break
      
  return (a, b)
    
def _determineOutputFileName(filePrefix, name):
  
  dirName = os.path.dirname(filePrefix)
  if not dirName:
    dirName = 'data'
  dirName += '_out'
  if not os.path.exists(dirName):
    os.mkdir(dirName)
  baseName = os.path.basename(filePrefix)
  fileName = '%s/%s_%s' % (dirName, baseName, name)
  
  return fileName

def _adjustedAssociationParams(params, binSize=1.0):
  
  # if fit is A exp(-B x) + C exp(-D x) then parameters go from
  # (A, C, B, D) to (A/(A+C), 1/B, C/(A+C), 1/D)
  
  numberExponentials = len(params) // 2
  s = sum(params[:numberExponentials])
  paramsNew = (2*numberExponentials)*[0]
  for i in range(numberExponentials):
    paramsNew[2*i] = params[i] / s
    paramsNew[2*i+1] = binSize / params[i+numberExponentials]
    
  return paramsNew

"""
def _bootstrapFit(xdata, ydata, params_opt, fitFunc, adjustedParamsFunc=None, ntrials=1000, fp=None):
  
  ndata = len(xdata)
  paramsList =  []
  for trial in range(ntrials):
    indices = range(ndata)
    indices = numpy.random.choice(indices, ndata)
    x = xdata[indices]
    y = ydata[indices]
    try:
      params, params_cov = curve_fit(fitFunc, x, y, p0=params_opt)
    except: # fit might fail
      continue
    if adjustedParamsFunc:
      params = adjustedParamsFunc(params)
    if fp:
      fp.write('%s\n' % ','.join(['%.3f' % p for p in params]))
    paramsList.append(params)
    
  paramsArray = numpy.array(paramsList)
  paramsMean = numpy.mean(paramsArray, axis=0)
  paramsStd = numpy.std(paramsArray, axis=0)
  #print('Bootstrap parameter mean = %s' % paramsMean)
  #print('Bootstrap parameter standard deviation = %s' % paramsStd)

  with open('junk_%d.csv' % (len(paramsStd)//2), 'w') as fp:
    for params in paramsArray:
      fields = ['%s' % p for p in params]
      fp.write(','.join(fields) + '\n')
  
  return paramsStd
"""
  
def _bootstrapFit(data, params_opt, fitFunc, adjustedParamsFunc=None, ntrials=1000, fp=None):
    
  ndata = len(data)
  paramsList =  []
  paramsListAdjusted =  []
  for trial in range(ntrials):
    indices = range(ndata)
    indices = numpy.random.choice(indices, ndata)
    trialData = data[indices]
    ydata, minData, binSize = _findAssociationData(trialData)
    xdata = numpy.arange(len(ydata))
    params0 = params_opt # _initialFitAssociationEstimate(ydata)
  
    try:
      params, params_cov = curve_fit(fitFunc, xdata, ydata, p0=params0)
    except: # fit might fail
      continue
    paramsList.append(params)
    if adjustedParamsFunc:
      params = adjustedParamsFunc(params, binSize)
    paramsListAdjusted.append(params)
    if fp:
      fp.write('%s\n' % ','.join(['%.3f' % p for p in params]))
    
  paramsArray = numpy.array(paramsList)
  paramsArrayAdjusted = numpy.array(paramsListAdjusted)
  paramsMean = numpy.mean(paramsArrayAdjusted, axis=0)
  paramsStd = numpy.std(paramsArrayAdjusted, axis=0)
  #print('Bootstrap parameter mean = %s' % paramsMean)
  #print('Bootstrap parameter standard deviation = %s' % paramsStd)

  """
  with open('junk_%d.csv' % (len(paramsStd)//2), 'w') as fp:
    for params in paramsArray:
      fields = ['%s' % p for p in params]
      fp.write(','.join(fields) + '\n')

  with open('junkAdjusted_%d.csv' % (len(paramsStd)//2), 'w') as fp:
    for params in paramsArrayAdjusted:
      fields = ['%s' % p for p in params]
      fp.write(','.join(fields) + '\n')
  """
  
  return paramsStd
    
def _findAssociationData(data):

  nbins = 10
  maxData = numpy.max(data)
  minData = numpy.min(data)
  binSize = (maxData - minData) / nbins
  ydata = numpy.zeros(nbins, dtype='float32')
  for d in data:
    bin = int((d-minData) / binSize)
    bin = min(bin, nbins-1)
    ydata[bin:] += 1

  ydata -= ydata[0]
  ydata /= ydata[-1]

  return ydata, minData, binSize

def _writeFitAssociationHeader(fp, maxNumberExponentials):
  
  data = ['nexp']
  for m in range(maxNumberExponentials):
    data.append('ampl%d' % (m+1))
    data.append('T%d' % (m+1))
  for m in range(maxNumberExponentials):
    data.append('amplErr%d' % (m+1))
    data.append('TErr%d' % (m+1))
  data.append('rss')
  data.append('bic')
  data.append('total')
  data.append('minData')
  data.append('binSize')
    
  data = ','.join(data)
  
  fp.write(data + '\n')
    
def _writeFitAssociationParams(fp, params, paramsStd, rss, maxNumberExponentials, ndata, binSize, minData):
  
  numberExponentials = len(params) // 2
  
  total = sum(params[:numberExponentials])

  #print('_writeFitAssociationParams0', binSize, ndata)
  #print('_writeFitAssociationParams1 params before', params)
  #print('_writeFitAssociationParams2 paramsStd', paramsStd)
  
  params = _adjustedAssociationParams(params, binSize)
  #print('_writeFitAssociationParams3 params after', params)
  
  n = 2 * (maxNumberExponentials - numberExponentials)
  
  data = ['%d' % numberExponentials]
  data.extend(['%.3e' % param for param in params])
  data.extend(n*[''])
  
  data.extend(['%.3e' % param for param in paramsStd])
  data.extend(n*[''])
  
  data.append('%.3e' % rss)
  
  bic = numpy.log(ndata) * (len(params) + 1) + ndata * (numpy.log(2*numpy.pi*rss/ndata) + 1)
  #bic = numpy.log(ndata) * (len(params)) + ndata * (numpy.log(rss/ndata))
  data.append('%.3e' % bic)
  
  data.append('%.3e' % total)
  data.append('%.3e' % minData)
  data.append('%.3e' % binSize)
    
  data = ','.join(data)
  
  fp.write(data + '\n')
  
def fitAssociationData(filePrefix, data, maxNumberExponentials=1, plotDpi=600):

  ydata, minData, binSize = _findAssociationData(data)
  xdata = numpy.arange(len(ydata))
  
  params0 = _initialFitAssociationEstimate(ydata)
  
  fileName = _determineOutputFileName(filePrefix, 'fitAssociation.csv')
  with open(fileName, 'w') as fp:
    _writeFitAssociationHeader(fp, maxNumberExponentials)
    params_list = []
    for numberExponentials in range(1, maxNumberExponentials+1):
      params_opt, params_cov = curve_fit(_fitInverseExponentials, xdata, ydata, p0=params0)
      ss = '' if numberExponentials == 1 else 's'
      params_err = numpy.sqrt(numpy.diag(params_cov))
      params_opt = tuple(params_opt)
      yfit = _fitInverseExponentials(xdata, *params_opt)
      rss = numpy.sum((yfit - ydata)**2)
      print('Fitting association with %d exponential%s, parameters = %s, parameter standard deviation = %s, rss = %f' % (numberExponentials, ss, params_opt, params_err, rss))
      #paramsStd = _bootstrapFit(xdata, ydata, params_opt, _fitInverseExponentials, _adjustedAssociationParams)
      paramsStd = _bootstrapFit(data, params_opt, _fitInverseExponentials, _adjustedAssociationParams)
      _writeFitAssociationParams(fp, params_opt, paramsStd, rss, maxNumberExponentials, len(xdata), binSize, minData)
      params_list.append(params_opt)
      params0 = list(params_opt[:numberExponentials]) + [0.1] + list(params_opt[numberExponentials:]) + [0.0]
    
  colors = ['blue', 'red', 'green', 'yellow', 'black']  # assumes no more than 4 exponentials
  bdata = minData + (xdata+0.5)*binSize
  plt.plot(bdata, ydata, color=colors[-1])
  for n in range(maxNumberExponentials):
    yfit = _fitInverseExponentials(xdata, *params_list[n])
    #print('HERE111', n, params_list[n], bdata, yfit, binSize, minData)
    plt.plot(bdata, yfit, color=colors[n])
  
  fileName = _determineOutputFileName(filePrefix, 'associationFit.png')
  plt.savefig(fileName, dpi=plotDpi, transparent=True)
  #plt.show()
  plt.close()
  
  # individual plots for each exponential fit and drawing it as incremental rather than cumulative
  for n in range(maxNumberExponentials):
    plt.plot(bdata, 1-ydata, color='black', linestyle='dashed')
    yfit = 1 - _fitInverseExponentials(xdata, *params_list[n])
    plt.plot(bdata, yfit, color='blue')
    fileName = _determineOutputFileName(filePrefix, 'associationFit_%d.png' % (n+1))
    plt.savefig(fileName, dpi=plotDpi, transparent=True)
    plt.close()

def processAssociationFile(filePath, maxNumberExponentials=1, plotDpi=600):

  print('Working on file %s' % filePath)

  data = readAssociationData(filePath)

  n = filePath.rfind('.')
  filePrefix = filePath[:n]
  fitAssociationData(filePrefix, data, maxNumberExponentials, plotDpi)

if __name__ == '__main__':

  import sys

  if len(sys.argv) == 1:
    print('Need to specify one or more csv files')
    sys.exit()

  maxNumberExponentials = 3
  plotDpi = 600
  for filePath in sys.argv[1:]:
    processAssociationFile(filePath, maxNumberExponentials, plotDpi)
