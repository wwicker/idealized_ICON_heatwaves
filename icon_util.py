import numpy as np
import xarray as xr
import math
from numba import guvectorize
from scipy import interpolate, ndimage
from dask import bag


########################
## grid related utilities

@guvectorize(   
    "(float64[:],int16,float64[:])",
    "(n), () -> (n)",
    #nopython=True
    forceobj=True
)
def filtering(a,n,out):
    tmp = np.append(a,n*[a[-1]])
    tmp = np.insert(tmp,0,n*[tmp[0]])
    
    for i in range(n):
        tmp = 0.25*tmp[:-2] + 0.5*tmp[1:-1] + 0.25*tmp[2:]
        
    out[:] = tmp
    
    
    
def regrid(array,lim=(-90,90),res=1,nfilter=10):
    
    # 1d arrays of grid specifications
    clon = array.clon.compute()
    clat = array.clat.compute()
    
    # lon-lat mesh grid
    lon, lat = np.meshgrid(np.linspace(-np.pi,np.pi,int(360/res)+1),
                           np.linspace(*np.radians(lim).tolist(),int(np.abs(np.diff(lim))/res+1)))
    
    # nearest neighbour interpolation
    interp = interpolate.NearestNDInterpolator(list(zip(clon,clat)),np.arange(len(clon)))

    # 2d DataArray used for indexing
    ncells = interp(lon,lat)
    ncells = xr.DataArray(ncells,dims=('latitude','longitude'),
                          coords=dict(latitude=np.linspace(*lim,int(np.abs(np.diff(lim))/res+1)),
                                      longitude=np.linspace(-180,180,int(360/res)+1)))

    # regridding by indexing
    
    regridded = array.isel(ncells=ncells)
    if nfilter > 0:
        regridded = xr.apply_ufunc(filtering,
                                   *(regridded,nfilter),
                                   input_core_dims=[['longitude'],[]],
                                   output_core_dims=[['longitude']],
                                   dask='parallelized',
                                   output_dtypes=[np.float64])
        
    return regridded



def eddy_flux(v,Phi):
    '''
        Compute the flux of anomalous Phi minimizing regridding 
        
        - the anomaly is a departure from the time-mean, zonal-mean
        - minimize regridding by delaying zonal averaging
    '''
    flux = (v * Phi).mean('time').persist()
    mean_v = v.mean('time').persist()
    mean_Phi = Phi.mean('time').persist()
    
    flux = regrid(flux).mean('longitude')
    mean_v = regrid(mean_v).mean('longitude')
    mean_Phi = regrid(mean_Phi).mean('longitude')
    
    eddy = flux - (mean_v * mean_Phi)
    
    return eddy


@guvectorize(   
    "(float64[:],float64[:],int16,complex128[:])",
    "(m), (n), () -> (n)",
    forceobj=True
)
def truncated_transform(a,k,N,out):
    
    z = np.fft.rfft(a,norm='forward')
    out[:] = z[:N]
    
    
@guvectorize(
    "(float64[:],float64[:],float64[:],float64[:])",
    "(n), (n), (m) -> (m)",
    nopython=True
)
def bin_average(weights,values,bins,out):
    '''
    '''
    indices = np.searchsorted(bins,values)
    out[:] = np.bincount(indices,weights=weights,minlength=len(bins)) / np.bincount(indices,minlength=len(bins))


def rfft(da,res=5):
    '''
    '''
    
    regridded = regrid(da)

    wavenum = xr.DataArray(np.fft.rfftfreq(int(360/res-1),d=1/int(360/res-1)),dims=('wavenumber'))
    
    coeff = xr.apply_ufunc(truncated_transform,
                           *(regridded,wavenum,int(180/res)),
                           input_core_dims=[['longitude'],['wavenumber'],[]],
                           output_core_dims=[['wavenumber']],
                           dask='parallelized',
                           output_dtypes=[np.dtype('complex128')]
                          )
    coeff['wavenumber'] = wavenum
    
    right = xr.DataArray(np.arange(-90+res,90+res,res),dims=('latitude_bin'))
    centers = xr.DataArray(np.arange(-90+res/2,90.1,res),dims=('latitude_bin'))
    
    # that doesn't work -> do bincount twice (with real/imaginary part of the weight)
    real = xr.apply_ufunc(bin_average,
                            *(np.real(coeff),coeff['latitude'],right),
                            input_core_dims=[['latitude'],['latitude'],['latitude_bin']],
                            output_core_dims=[['latitude_bin']],
                            dask='parallelized',
                            output_dtypes=[np.dtype('float64')]
                           )
    imag = xr.apply_ufunc(bin_average,
                            *(np.imag(coeff),coeff['latitude'],right),
                            input_core_dims=[['latitude'],['latitude'],['latitude_bin']],
                            output_core_dims=[['latitude_bin']],
                            dask='parallelized',
                            output_dtypes=[np.dtype('float64')]
                           )
    binned = real + imag * 1j
    binned['latitude_bin'] = centers
                          
    
    return binned



@guvectorize(
    "(complex128[:],float64[:],float64[:])",
    "(m), (k) -> (k)",
    forceobj=True
)
def spectrum(a,freq,out):
    
    padded = np.pad(a,(0,len(freq)-len(a)))
                    
    z = np.fft.fft(padded)
    
    out[:] = np.real(z * np.conj(z))
    

@guvectorize(   
    "(complex128[:],float64[:],int16,complex128[:])",
    "(m), (n), () -> (n)",
    forceobj=True
)
def hilbert_transform(coeff,x,N,out):
    
    padded = np.pad(coeff,(0,N))
    
    padded = 2 * padded
    padded[0] = padded[0]/2
    padded[N] = padded[N]/2
    
    out[:] = np.fft.ifft(padded,norm='forward')


####################################
## temperature relaxation utilities


def T_eq(lat,plev):
    '''
        Held & Suarez (1994) equilibrium temperature
    '''
    
    phi = np.radians(lat)
    sigma = plev/max(plev)
    
    T = 315 - 60 * np.sin(phi)**2 - 10 * np.log(sigma) * np.cos(phi)**2
    T = T * sigma ** (2/7)
    T = T.where(T>200,200)
    
    return T


def tropical_heating(lat,plev,q_0=0.25,sig_x=0.4,x_0=0,sig_y=0.11,y_0=0.3):
    '''
        Heating term for the "Expansion of the Tropics" experiment by Butler et al. (2010)
    '''
    phi = np.radians(lat)
    sigma = plev/max(plev)
    
    Q = q_0 * np.exp(-0.5*(phi-x_0)**2/sig_x**2 - 0.5*(sigma-y_0)**2/sig_y**2)
    return Q


def polar_heating(lat,plev,q_0=1,x_0=1.57,y_0=1):
    '''
        Heating term for the "Artic Amplification" experiment by Butler et al. (2010)
    '''
    phi = np.radians(lat)
    sigma = plev/max(plev)
    
    Q = q_0 * np.cos(phi-x_0)**(15) * np.exp(2*(sigma-y_0))
    Q = Q.where(phi>0,0)
    
    return Q

#############################
# Statistical utilities

@guvectorize(
    "(complex128[:,:], complex128[:], complex128[:,:], complex128[:,:], float64[:])",
    "(m,n), (k) -> (m,k), (n,k), (k)",
    forceobj=True
)
def complex_svd(X,dummy,U,VS,S2):
    '''
        Vectorized singular value decomposition  -> generalized NumPy universal function
        
        - X = U @ np.diag(S) @ VH
        - U is standardized
        - m is dimension of time, n is stacked dimension, k = min(m,n)
    '''
    u, s, vh = np.linalg.svd(X,full_matrices=False)
    u_std = np.std(u,axis=0)
    U[:,:] = u/u_std
    VS[:,:] = vh.transpose() * s * u_std
    S2[:] = s**2 

    
    

@guvectorize(
    "(float64[:], float64[:], float64[:])",
    "(m), (n) -> (m)",
    forceobj=True
)
def vectorized_convolution(x,kernel,out):
    '''
        Vectorized convolution -> generalized NumPy universal function
        
        - mode='wrap' means that input is assumed being periodic
        - mode='mirror' means that input is extended by reflectinf about the center of the last pixel
    '''
    out[:] = ndimage.convolve(x,kernel,mode='mirror')
    
    
    
def bandpass_fileByFile(files,c1,c2=2,mode='typical',n=None,valid=True):
    '''
        Lanczos filter (Duchon, 1979)
        
        - c1: cutoff period for highpass filter
        - c2: cutoff period for lowpass filter,
          lower bound is Nyquist period to remove aliasing
        - the typical kernel half width provides a sharp filter,
          the minimal kernel half width prevents attenuation a the center of the frequency window
    '''
    # default kernel half width
    if n is None:
        if mode == 'minimal':
            n = math.ceil(1.3/(1/c2-1/c1))
        if mode == 'typical':
            n = math.ceil(1.5*c1)
        
    # Lanczos kernel
    k = np.arange(-n,n+1)
    kernel = (2/c2*np.sinc(2*k/c2) - 2/c1*np.sinc(2*k/c1)) * np.sinc(k/n)
    kernel = xr.DataArray(kernel,dims=('kernel',))
    
    convolution = lambda da: xr.apply_ufunc(vectorized_convolution,
                                            da,kernel,
                                            input_core_dims=[['time',],['kernel']],
                                            output_core_dims=[['time',],],
                                            dask='parallelized',
                                            output_dtypes=[da.dtype])
    
    # fileByFile computation with dask.bag
    data = bag.from_sequence(files).map(xr.open_dataarray,drop_variables=('clat_bnds','clon_bnds'),chunks=dict(time=-1))
    filtered = data.map(convolution)
    
    # select slice with valid convolution
    if valid:
        valid_slice = slice(n,-n)
        filtered = filtered.map(xr.DataArray.isel,time=valid_slice)
        
    # concatenate into one array
    array = xr.concat(filtered.compute(),dim='time')

    return array