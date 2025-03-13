'''heatwave_boostrap.py

    Use a stationary bootstrap following Politis, D. N. and Romano, J. P. (1994)
    to estimate confidence intervals of basic heatwave metrics.
    
    - requires a json file with detected heatwave data points and
      a netCDF file with temperature percentile for standardising intensity.
    
    wolfgang.wicker@unil.ch, Jan 2024
'''
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import scipy.stats as stats
import numba
import math
import time
import os

from icon_util import regrid



def pseudo_index(N,p):
    '''
        Integer index to construc pseudo sample
        
        - mean block length is 1/p
    '''
    index = []
    
    while len(index) < N:
        pos = int(stats.uniform.rvs(scale=N))
        length = stats.geom.rvs(p)
        index.extend(list(range(pos,pos+length)))

    index = np.mod(index,N)

    return index[:N]



def pseudo_sample(data,p):
    '''
        select heatwave data points by condition the start date with pseudo_index
    '''
    time = np.unique(data['start'].values)
    index = []

    for i in pseudo_index(len(time),p=0.1):
        index.extend(np.where(data.start == time[i])[0])
    
    index = np.hstack(index)

    return data.iloc[index]



def metrics(data,dist,nseason=200):
    '''
        Produce Dataset with heatwave metrics from DataFrame and temperature distribution
    '''
    frequency = data.groupby('ncells')['length'].sum().to_xarray()
    frequency = frequency / nseason

    length = data.groupby('ncells')['length'].mean().to_xarray()

    ds = xr.Dataset(dict(frequency=frequency,length=length))
    ds = ds.assign_coords(dict(clon=dist.clon,clat=dist.clat))

    return ds



@numba.guvectorize(
    "(float64[:],float64[:],float64[:])",
    "(n), (m) -> (m)",
    forceobj=True
)    
def icdf(a,p,out):
    '''
        Inverse empirical cummulative distribution function of array at percentiles p
    '''
    sort = np.sort(a)
    out[:] = sort[np.int64(p*len(a))]


def confid(dist,alpha):
    '''
        Estimate confidence intervals using the inverse empirical cummulative distribution function 
        for distrubution of bootstrap samples.
    '''
    p = xr.DataArray([alpha/2, 1-alpha/2],dims=('percentile'))
    values = xr.apply_ufunc(icdf,
                          *(dist,p),
                          input_core_dims=[['random'],['percentile']],
                          output_core_dims=[['percentile']],
                          dask='parallelized',
                          output_dtypes=[[dist.dtype]])
    values['percentile'] = p
    
    return values



if __name__ == '__main__':
    
    # Initialize parser
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('jsonFile',help='input file for heatwave data')
    parser.add_argument('ncFile',help='input file for percentiles')
    parser.add_argument('outDir',help='directory for metrics output')
    parser.add_argument('-p','--probability',dest='p',nargs='?',default=0.1,type=float,
                        help='probability to determin block length (default: 0.1)')
    parser.add_argument('-N','--nBootstrap',dest='N',nargs='?',default=1,type=int,
                        help='number of new pseudo samples to generate (default: 1)')
    parser.add_argument('-v','--verbose',action='store_true')
    args = parser.parse_args()
    
    print(args.p)
    
    
    # create outDir if necessary
    os.makedirs(args.outDir, exist_ok=True)
    if args.verbose:
        print('\nDIRECTORY FOR OUTPUT '+args.outDir+':')
        print(os.listdir(args.outDir))
    
    # load diagnostics from disk
    data = pd.read_json(args.jsonFile)
    dist = xr.open_dataarray(args.ncFile)
    
    if args.verbose:
        print(data)
        print(dist)
        
    # compute metrics for pseudo sample
    print('\nGENEREATE %d NEW BOOTSTRAP SAMPLES'%args.N)
    
    for i in range(args.N):
        sample = pseudo_sample(data,args.p)
        ds = metrics(sample,dist)
        ds.to_netcdf(args.outDir+'/sample_'+str(int(time.time()))+'.nc')
        
        
    # estimate zonal-mean metrics for real sample
    ref = metrics(data,dist)
    ref = regrid(ref,lim=(0,90))
    ref = ref.mean('longitude')
    
    if args.verbose:
        print(ref)

        
    # 
    bootstrap = xr.open_mfdataset(args.outDir+'/sample_*.nc',combine='nested',concat_dim='random')
    if args.verbose:
        print(bootstrap)
        
    
    bootstrap = regrid(bootstrap,lim=(0,90))
    bootstrap = bootstrap.mean('longitude').compute()
        

    frequency = confid(bootstrap['frequency'],0.05)
    length = confid(bootstrap['length'],0.05)
    CI = xr.Dataset(dict(frequency=frequency,length=length))
    
    #CI = regrid(CI,lim=(0,90))
    #CI = CI.mean('longitude')
    
    if args.verbose:
        print(CI)
        
        
    # plotting
    print('\nPLOT ESTIAMATE TO '+args.outDir+'/plot_zonal_mean.png')
    
    fig, axes = plt.subplots(nrows=2,figsize=(4,6))

    # Frequency
    l = ref['frequency'].plot(ax=axes[0])
    axes[0].fill_between(CI['latitude'].values,CI['frequency'].sel(percentile=0.025).values,CI['frequency'].sel(percentile=0.975).values,color=l[0].get_c(),alpha=0.4)

    # length
    ref['length'].plot(ax=axes[1])
    axes[1].fill_between(CI['latitude'].values,CI['length'].sel(percentile=0.025).values,CI['length'].sel(percentile=0.975).values,color=l[0].get_c(),alpha=0.4)


    for ax in axes:
        ax.set_title('')
        ax.xaxis.grid()
        ax.set_xlim(0,90)
        ax.set_xlabel('')
    
    axes[0].set_ylabel('Frequency',weight='bold')
    axes[1].set_ylabel('Length',weight='bold')
    
    axes[1].set_xlabel('Latitude [°N]')

    fig.subplots_adjust(0,0,1,1,0,0.3)
    
    plt.savefig(args.outDir+'/plot_zonal_mean.png',bbox_inches='tight',dpi=300)
    
