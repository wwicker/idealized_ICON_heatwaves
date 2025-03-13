'''heatwave_noSeasonalCycle.py

    Heatwave diagnosis for ICON simulations without seasonal cycle
    
    - Open 2d variable from netcdf file and compute daily maxima
    - Estimate percentiles and store as netCDF file
    - Identify heatwaves as exceedances of the 95th percentile
      for at least three days in a row and store as json file
      
      BEWARE SHOULD USE THE 90th PERCENTILE
      
    wolfgang.wicker@unil.ch, Nov 2022
'''
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import numba


@numba.guvectorize(
    "(float64[:],float64[:],float64[:])",
    "(n), (m) -> (m)",
    forceobj=True
)    
def ecdf(a,p,out):
    '''
        Emperical cummulative distribution function of array at percentiles p
    '''
    sort = np.sort(a)
    out[:] = sort[np.int64(p*len(a))]



@numba.jit(nopython=True)
def check_length(array):
    '''
        Check exceedance for length >= 3 days
    '''
    start = []
    length = []
    mean = []
    count = 0
    
    for i in range(len(array)):
        if np.isnan(array[i]):
            if count > 2:
                start.append(i-count)
                length.append(count)
                mean.append(sum([array[j] for j in range(i-count,i)]))
                
            count = 0  
            
        else:
            count +=1
                            
    return start, length, mean


@numba.jit(nopython=True,parallel=False)
def cell_loop(array):
    '''
        Loop check_length over all grid cells
    '''    
    # First loop to infer number of elements
    count = 0
    for j in numba.prange(array.shape[1]):
        s,l,m = check_length(array[:,j])
        
        count += len(s)
        
    # Create empty arrays with correct length    
    ncells = np.zeros(count,np.int_)
    start = np.zeros(count,np.int_)
    length = np.zeros(count,np.int_)
    mean = np.zeros(count,np.float_)
    
    # Second loop to fill arrays
    count = 0
    for j in numba.prange(array.shape[1]):
        s,l,m = check_length(array[:,j])

        for i in range(len(s)):
            ncells[count+i] = j
            start[count+i] = s[i]
            length[count+i] = l[i]
            mean[count+i] = m[i]
        
        count += len(s)
        
    return ncells, start, length, mean


@numba.jit(nopython=True)
def iterrows(start,length):
    '''
        Infer heatwave days from arrays heatwave start and length
    '''
    days = [np.int64(x) for x in range(0)]
    for s, l in zip(start,length):
        for i in range(l):
            day = int((s+i)) % 100
            month = int((s+i)/100) * 100 % 10000
            year = int((s+i)/10000) * 10000
            if day > 30:
                day -= 30
                month += 100
            if month > 1200:
                month -= 1200
                year += 10000
            days.append(year+month+day)
    return days
        

def mask_from_frame(data,ncells=20480,nyears=50):
    '''
        Get gridded heatwave mask from DataFrame of heatwaves
    '''
    one_year = np.array([i*100 + np.arange(1,31) for i in range(1,13)]).flatten()
    time = np.array([i*10000+one_year for i in range(nyears+1)]).flatten()
    mask_array = xr.DataArray(np.zeros((ncells,len(time))),coords=dict(ncells=np.arange(ncells),time=time))
    
    mask_frame = []
    days = [np.int64(x) for x in range(0)]
    day = 0
    month = 0
    year = 0

    for name, group in data.groupby('ncells'):
        days = iterrows(group['start'].to_numpy(),group['length'].to_numpy())
        mask_frame.append(pd.DataFrame(dict(ncells=[name]*len(days),days=days),columns=['ncells','days']))
    mask_frame = pd.concat(mask_frame)
    
    for name, group in mask_frame.groupby('ncells'):
        mask_array.loc[dict(ncells=name,time=group.days.to_numpy())] = 1
    
    return mask_array


if __name__ == '__main__':
    
    # Initialize parser
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', help='string glob in the form "path/to/my/files/*.nc"')
    parser.add_argument('var', help='name of 2d variable')
    parser.add_argument('--ncFile',dest='ncFile',nargs='?',
                        const='./percentiles.nc',default='./percentiles.nc',
                        help='output file for percentiles (default: "./percentiles.nc")')
    parser.add_argument('--jsonFile',dest='jsonFile',nargs='?',
                        const='./heatwaves.json',default='./heatwaves.json',
                        help='output file for heatwaves (default: "./heatwaves.json")')
    parser.add_argument('--mean',dest='reduce',action='store_const',const=np.mean,default=np.max,
                        help='reduce data to daily mean (default: daily max)')
    parser.add_argument('--spinup',action='store_true',help='retain 50 years without spin-up')
    parser.add_argument('-v','--verbose',action='store_true')
    args = parser.parse_args()
    
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
    
    # estimate percentiles
    if args.verbose: print('\nESTIMATE PERCENTILES')
            
    p = xr.DataArray([0.05,0.25,0.5,0.75,0.90,0.95],dims=('p'))
    dist = xr.apply_ufunc(ecdf,
                          *(array,p),
                          input_core_dims=[['time'],['p']],
                          output_core_dims=[['p']],
                          dask='parallelized',
                          output_dtypes=[array.dtype])
    dist['p'] = p
    
    if args.verbose: print(dist)
    
    print('\nSTORE OUTPUT TO '+args.ncFile)
    xr.Dataset(dict(percentiles=dist)).to_netcdf(args.ncFile)
    
    # identify days exceeding the threshold
    if args.verbose: print('\nIDENTIFY HEATWAVES')
    
    masked = array.where(array >= dist.sel(p=0.90).squeeze())
    
    # evaluate heatwave length
    ncells, start, length, mean = cell_loop(masked.values)
    start = masked['time'].isel(time=start)
    start = [str(i).zfill(8) for i in start.values]
    data = pd.DataFrame(dict(ncells=ncells,start=start,length=length,mean=mean),
                        columns=['ncells','start','length','mean'])
    data['mean'] = data['mean'] / data['length']
    
    if args.verbose: print(data)
    
    print('\nSTORE OUTPUT TO '+args.jsonFile)
    data.to_json(args.jsonFile)
    
    