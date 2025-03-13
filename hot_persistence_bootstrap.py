'''hot_persistence_bootstrap.py

    Create bootstrap samples of hot day persistence.
    
    wolfgang.wicker@unil.ch, Aug 2024
'''
import argparse
import numpy as np
import xarray as xr
import scipy.stats as stats
import numba
import math
import time
import os

from icon_util import regrid


@numba.jit(nopython=True)
def count_duration(array,index):
    '''
        Count occurence of set of consecutive hot days with certain length
        
        - first element of occurence counts sets that are longer than max_duration
    '''
    max_duration = 14
    occurence = np.zeros(max_duration+1,np.int_)
    count = 0
    
    for i in index:
        if array[i]:
            if count > 0:
                if count > max_duration:
                    occurence[0] += 1
                else:
                    occurence[count] += 1
            
            count = 0  
            
        else:
            count +=1
            
    return occurence
    

    
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



def loop_sample(array,p):
    '''
        Create index and loop counter over ncells
    '''
    N = len(array.time)
    index = pseudo_index(N,p)
    
    # prepare array
    len1 = len(count_duration(array.isel(ncells=0).values,index))
    hist = np.zeros((len1,len(array.ncells)),np.int_)
    
    for i in range(len(array.ncells)):
        
        hist[:,i] = count_duration(array.isel(ncells=i).values,index)
        
        
    # prepare data
    length = xr.DataArray(range(1,len1),dims=('length'))
    events = xr.DataArray(hist[1:,:],dims=('length','ncells'),coords=dict(length=length))
    
    days = length * events
    
    missing_events = xr.DataArray(hist[0,:],dims='ncells')
    missing_days = 0.1*N - days.sum('length')
    
    return xr.Dataset(dict(days=days,missing=missing_days,missing_events=missing_events))




    



if __name__ == '__main__':


    # Initialize parser
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', help='string glob in the form "path/to/my/files/*.nc"')
    parser.add_argument('percentiles',help='input file for percentiles')
    parser.add_argument('outDir',help='directory for metrics output')
    parser.add_argument('--mean',dest='reduce',action='store_const',const=np.mean,default=np.max,
                        help='reduce data to daily mean (default: daily max)')
    parser.add_argument('--spinup',action='store_true',help='retain 50 years without spin-up')
    parser.add_argument('--var',dest='var',nargs='?',
                        const='ta',default='ta', help='name of 2d variable')
    parser.add_argument('-p','--probability',dest='p',nargs='?',default=0.1,type=float,
                        help='probability to determin block length (default: 0.1)')
    parser.add_argument('-N','--nBootstrap',dest='N',nargs='?',default=1,type=int,
                        help='number of new pseudo samples to generate (default: 1)')
    parser.add_argument('-v','--verbose',action='store_true')
    args = parser.parse_args()
    
    if args.verbose: print(args.p)

    # create outDir if necessary
    os.makedirs(args.outDir, exist_ok=True)
    if args.verbose:
        print('\nDIRECTORY FOR OUTPUT '+args.outDir+':')
        print(os.listdir(args.outDir))
        
        
    ##############    
    # open dataset
    print('\nOPEN '+args.var+' FROM '+args.paths)
    array = xr.open_mfdataset(args.paths,combine='nested',concat_dim='time')[args.var].squeeze()
    
    if args.verbose: print(array)
    
    if args.spinup: array = array.sel(time=slice(1300,501300))
    
    # aggregate to daily data
    if args.verbose: 
        if args.reduce == np.max:
            print('\nCOMPUTE DAILY MAXIMA')
        elif args.reduce == np.mean:
            print('\nCOMPUTE DAILY MEANS')
    
    array['time'] = array['time'].astype(np.int64)
    array = array.groupby('time').reduce(args.reduce)
    array = array.compute()
    
    if args.verbose: print(array)
    
    
    ##############
    if args.verbose: print('\nIDENTIFY HOT DAYS')
    
    # load diagnostics from disk
    dist = xr.open_dataarray(args.percentiles)
    
    if args.verbose:
        print(dist)
        
    # identify days that fail to exceed the threshold
    failing = (array < dist.sel(p=0.90).squeeze())
    
    
    
    ##############
    print('\nGENEREATE %d NEW BOOTSTRAP SAMPLES'%args.N)
    
    for i in range(args.N):
        sample = loop_sample(failing,p=args.p)
        mean= regrid(sample.assign_coords(dict(clon=dist.clon,clat=dist.clat)),lim=(0,90)).mean('longitude')
        mean.to_netcdf(args.outDir+'/regridded_'+str(int(time.time()))+'.nc')
        
