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

def _adjustedAssociationParams(params):
  
  # if fit is A exp(-B x) + c exp(-D x) then parameters go from
  # (A, C, B, D) to (A/(A+C), 1/B, C/(A+C), 1/D)
  
  numberExponentials = len(params) // 2
  s = sum(params[:numberExponentials])
  paramsNew = (2*numberExponentials)*[0]
  for i in range(numberExponentials):
    paramsNew[2*i] = params[i] / s
    paramsNew[2*i+1] = 1 / params[i+numberExponentials]
    
  return paramsNew

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
      pass
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

  return ydata

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
    
  data = ','.join(data)
  
  fp.write(data + '\n')
    
def _writeFitAssociationParams(fp, params, paramsStd, rss, maxNumberExponentials, ndata):
  
  numberExponentials = len(params) // 2
  params = _adjustedAssociationParams(params)
  n = 2 * (maxNumberExponentials - numberExponentials)
  
  data = ['%d' % numberExponentials]
  data.extend(['%.3f' % param for param in params])
  data.extend(n*[''])
  
  data.extend(['%.3f' % param for param in paramsStd])
  data.extend(n*[''])
  
  data.append('%.3f' % rss)
  
  bic = numpy.log(ndata) * (len(params) + 1) + ndata * (numpy.log(2*numpy.pi*rss/ndata) + 1)
  data.append('%.3f' % bic)
    
  data = ','.join(data)
  
  fp.write(data + '\n')
  
def fitAssociationData(filePrefix, data, maxNumberExponentials=1, plotDpi=600):

  ydata = _findAssociationData(data)
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
      paramsStd = _bootstrapFit(xdata, ydata, params_opt, _fitInverseExponentials, _adjustedAssociationParams)
      _writeFitAssociationParams(fp, params_opt, paramsStd, rss, maxNumberExponentials, len(xdata))
      params_list.append(params_opt)
      params0 = list(params_opt[:numberExponentials]) + [0.1] + list(params_opt[numberExponentials:]) + [0.0]
    
  colors = ['blue', 'red', 'green', 'yellow', 'black']  # assumes no more than 4 exponentials
  plt.plot(xdata, ydata, color=colors[-1])
  for n in range(maxNumberExponentials):
    yfit = _fitInverseExponentials(xdata, *params_list[n])
    plt.plot(xdata, yfit, color=colors[n])
  
  fileName = _determineOutputFileName(filePrefix, 'associationFit.png')
  plt.savefig(fileName, dpi=plotDpi, transparent=True)
  #plt.show()
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

  maxNumberExponentials = 2
  plotDpi = 600
  for filePath in sys.argv[1:]:
    processAssociationFile(filePath, maxNumberExponentials, plotDpi)
