''' hovmoeller_composite.py


'''
import xarray as xr
import numpy as np
import pandas
import dask.distributed
import argparse
import os



def three_year_data(da,central):
    
    data = da.where(da['time.year'].isin(range(central-1,central+2)),drop=True)
    data = data.compute()
    
    # wrap to cylcic
    upper = data.sel(lon=slice(-179,0))
    upper['lon'] = upper['lon'] + 360
    lower = data.sel(lon=slice(0,179))
    lower['lon'] = lower['lon'] - 360
    
    return xr.concat([lower,data,upper],dim='lon')



def contruct_composite(data,heatFile,outFile,lat=False,dt=1/24):
    
    heat = pandas.read_json(heatFile)
    heat = heat[heat['lat'] == lat]
    
    start = heat['start'].values
    lon = heat['lon'].values
    
    composite = []
    
    for day, x in zip(start,lon):
        
        selection = data.sel(time=slice(np.datetime64(day)-np.timedelta64(5,'D'),np.datetime64(day)+np.timedelta64(10,'D')))
        selection = selection.drop('time').rename(time='step')
        selection = selection.assign_coords(onset=np.datetime64(day),loc=x)
        
        selection['lon'] = selection['lon'] - x
        selection = selection.interp(lon=range(-180,180))
        
        if dt<1:
            step = np.arange(-5,10,dt)
        else:
            step = np.arange(-5,11,dt)
        
        if len(selection.step)==len(step):
            composite.append(selection.assign_coords(step=step))
        
    
    composite = xr.concat(composite,dim='onset')
    
    xr.Dataset(dict(anomaly=composite)).to_netcdf(outFile)



if __name__ == '__main__':
    
    work = os.environ.get('WORK')+'/'
    
    client = dask.distributed.Client()
    
    # Initialize parser
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('lat',help='latitude',type=float)
    parser.add_argument('start',help='first year',type=int)
    parser.add_argument('end',help='last year',type=int)
    args = parser.parse_args()
    
    
    lat = args.lat
    years = range(args.start,args.end+1)
    heatDir = work+'wolfgang/ERA5_surf_day_max/heat_1979-2022/'
    varDir = work+'wolfgang/ERA5_surf_day_max/'
    varSuffix = 'era5_an_t2m_reg05_1h_'
    varName = 'var167'
    dt = 1
    outDir = work+'wolfgang/ERA5_surf_day_max/hovmoeller/'
    
    
    # reference with dimension 'dayofyear' to compute anomalies
    clim = xr.open_dataarray(work+'wolfgang/ERA5_surf_day_max/percentiles_31days.nc').sel(p=0.9)
    
    # select meridional range and compute anomalies
    print('\n VARIABLE PREPARATION:')
    
    clim = clim.sel(lat=lat,method='nearest')
    
    files = [varDir+f for f in os.listdir(varDir) if sum([f.startswith(varSuffix+str(y)) for y in years])]
    files.sort()
    
    data = xr.open_mfdataset(files,combine='nested',concat_dim='time')[varName]
    data = data.sel(lat=lat,method='nearest')
    
    data = data.groupby('time.dayofyear') - clim
    
    print(data)
    
    
    # distribute data on Local cluster
    distributed = client.map(three_year_data,[data,]*len(years),years)
    
    # prepare filenames
    heatFiles = [heatDir+'heat_%d.json'%y for y in years]
    outFiles = [outDir+'t2m_composite_%02.1fN_%d.nc'%(lat,y) for y in years]
    
    # construct composite on Local cluster and store to disk
    print('\n STORE COMPOSITES TO:')
    print(outFiles)
    
    composites = client.map(contruct_composite,distributed,heatFiles,outFiles,lat=lat,dt=dt)
    
    # wait end of computation
    client.gather(composites)
    
    