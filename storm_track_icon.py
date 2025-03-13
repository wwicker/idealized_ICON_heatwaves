'''storm_track.py

    Diagnose storm track from ICON (ua,va) output using bandpass filtering.
    
    - vertically integrated 10-day highpass filter EKE
    
    wolfgang.wicker@unil.ch, Sep 2023
'''
import xarray as xr
import numpy as np
import os
import argparse

from icon_util import regrid, bandpass_fileByFile


def filter_variable(variable, experiment, high, low, mode):
    
    data_dir = '/work/FAC/FGSE/IDYST/ddomeise/default/DATA/icon_simulations/'
    
    levels = [10000,20000,25000,30000,40000,50000,70000,85000,100000]
    
    var = []
    
    for lev in levels:
        
        files = [data_dir+experiment+'/3d/'+variable+'/%dpa/'%lev+f 
                 for f in os.listdir(data_dir+experiment+'/3d/'+variable+'/%dpa/'%lev) 
                 if f.endswith('.nc')]
        files.sort()
        
        var.append(bandpass_fileByFile(files,high,low,mode))
        
    return xr.concat(var,dim='plev')



if __name__ == '__main__':
    
    # Initialize parser
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('exp', help='subdirectory of /work/FAC/FGSE/IDYST/ddomeise/default/DATA/icon_simulations/')
    parser.add_argument('highpass', type=int, help='cutoff period in units of timestep')
    parser.add_argument('--lowpass',dest='lowpass',nargs='?',
                        type=int,const=2,default=2,
                        help='cutoff period in units of timesteps (default: %(default)d)')
    parser.add_argument('--mode',dest='mode',nargs='?',
                        default='typical',const='typical',choices=['typical','minimal'],
                        help='bandpass filter kernel width (default: %(default)s)')
    parser.add_argument('--spinup',action='store_true',help='retain 50 years without spin-up')
    parser.add_argument('-v','--verbose',action='store_true')
    args = parser.parse_args()
    
    
    # filter horizontal wind
    
    print('\nLOAD DATA FOR EXPERIMENT '+args.exp+' WITH FILTER SETTING')
    print(args.highpass,args.lowpass,args.mode)
    
    zonal = filter_variable('ua',args.exp,args.highpass,args.lowpass,args.mode)
    meridional = filter_variable('va',args.exp,args.highpass,args.lowpass,args.mode)
    
    if args.spinup:
        print('\n REMOVE SPINUP')
        zonal = zonal.sel(time=slice(1300,501300))
        meridional = meridional.sel(time=slice(1300,501300))
        
    if args.verbose:
        print(zonal)
        print(meridional)
        
        
    # compute vertically integrated energy
    
    dp = [10000,7500,2500,7500,10000,15000,17500,15000,7500]
    dp = xr.DataArray(dp,coords=dict(plev=[10000,20000,25000,30000,40000,50000,70000,85000,100000]))
    
    energy = (zonal**2 + meridional**2) / 19.62
    integral = (energy * dp).sum('plev')
    
    
    # invoke computation on LocalCluster and rechunk
    print('\nEXECUTE VERTICAL INTEGRATION')
    
    integral = integral.persist()
    integral = integral.chunk(dict(time=60))
    
    if args.verbose: print(integral)
        
        
    # regridding for zonal average
    
    res = 0.25
    
    print('\nREGRID TO RESOLUTION %0.2f DEGREES'%res)
    
    storm = regrid(integral,res=res).mean('longitude')
    
    storm = storm.compute()
    
    if args.verbose: print(storm)
    
    
    # store to disk
    
    xr.Dataset(dict(EKE=storm)).to_netcdf('./storm_track_zonal_mean.nc')
    
