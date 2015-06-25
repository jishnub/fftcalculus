#cython: boundscheck=False, wraparound=False
from __future__ import division
import numpy as np
cimport numpy as np
import cython

cpdef integrate_real(np.ndarray arr,int axis=-1,double period=2*np.pi):
    
    if arr.ndim==1: return integrate_real_1D(arr,period=float(period))
    elif arr.ndim==2: return integrate_real_2D(arr,axis=axis,period=float(period))

    cdef int i,N,Nk,ndim
    cdef np.ndarray arrf,coeff,integralf
    cdef np.ndarray integral
    cdef np.ndarray x
    
    arr=np.swapaxes(arr,0,axis)
    N=arr.shape[0]
    ndim=arr.ndim
    coeff=1j*(2*np.pi)*np.fft.rfftfreq(N,period/N)
    Nk=coeff.shape[0]
    shape=[Nk];shape.extend([1]*(ndim-1))
    coeff=coeff.reshape(tuple(shape))
    arravg=np.mean(arr,axis=0,keepdims=True)
    arrf=np.fft.rfft(arr-arravg,axis=0)
    integralf=np.zeros_like(arrf)
    
    x=np.linspace(0,period,N,endpoint=False)
    shape=[N];shape.extend([1]*(ndim-1))
    x=x.reshape(tuple(shape))
    
    integralf[1:]=arrf[1:]/coeff[1:]
    
    integral=np.fft.irfft(integralf,axis=0).real+x*arravg
    
    integral=np.swapaxes(integral,0,axis)
    return integral
    
cpdef integrate_complex(np.ndarray arr,int axis=-1,double period=2*np.pi):

    cdef int i,N,ndim
    cdef np.ndarray arrf,coeff,integralf,integral
    cdef np.ndarray x
    
    arr=np.swapaxes(arr,0,axis)
    N=arr.shape[0]
    ndim=arr.ndim
    arravg=np.sum(arr,axis=0,keepdims=True)/N
    arrf=np.fft.fft(arr-arravg,axis=0)
    integralf=np.zeros_like(arrf)
    coeff=1j*(2*np.pi)*np.fft.fftfreq(N,period/N)
    shape=[N];shape.extend([1]*(ndim-1))
    coeff=coeff.reshape(tuple(shape))
    
    x=np.linspace(0,period,N,endpoint=False)
    x=x.reshape(tuple(shape))
    
    integralf[1:]=arrf[1:]/coeff[1:]
    
    integral=np.fft.ifft(integralf,axis=0)+x*arravg
    
    integral=np.swapaxes(integral,0,axis)
    return integral

cpdef integrate_real_1D(np.ndarray[double,ndim=1] arr,double period=2*np.pi):

    cdef int i,N,Nk
    cdef np.ndarray[complex,ndim=1] arrf,coeff,integralf
    cdef np.ndarray[double,ndim=1] x,integral
    cdef double arravg
    
    N=arr.shape[0]
    coeff=1j*(2*np.pi)*np.fft.rfftfreq(N,period/N)
    Nk=coeff.shape[0]
    arravg=np.mean(arr)
    arrf=np.fft.rfft(arr-arravg)
    integralf=np.zeros(Nk,dtype=complex)
    
    x=np.linspace(0,period,N,endpoint=False)
    
    integralf[1:]=arrf[1:]/coeff[1:]
    
    integral=np.fft.irfft(integralf).real+x*arravg
    
    return integral

cpdef integrate_real_2D(np.ndarray[double,ndim=2] arr,int axis=-1,double period=2*np.pi):

    cdef int i,N,Nk
    cdef np.ndarray[complex,ndim=2] arrf,coeff,integralf
    cdef np.ndarray[double,ndim=2] integral,x,arravg
    
    arr=np.swapaxes(arr,0,axis)
    
    N=arr.shape[0]
    coeff=np.atleast_2d(1j*(2*np.pi)*np.fft.rfftfreq(N,period/N)).T
    Nk=coeff.shape[0]

    arravg=np.mean(arr,axis=0,keepdims=True)
    arrf=np.fft.rfft(arr-arravg,axis=0)
    integralf=np.zeros((Nk,arrf.shape[1]),dtype=complex)
    
    x=np.atleast_2d(np.linspace(0,period,N,endpoint=False)).T

    integralf[1:]=arrf[1:]/coeff[1:]

    integral=np.fft.irfft(integralf,axis=0).real+x*arravg
    
    integral=np.swapaxes(integral,0,axis)    
    
    return integral

cpdef integrate(np.ndarray arr,int axis=-1,period=2*np.pi):
    if np.isreal(arr).all():
        arr=arr.real
        if arr.dtype!=float: arr=arr*1.0
        return integrate_real(arr,axis,float(period))
    else:
        if arr.dtype!=complex: arr=arr*1.0
        return integrate_complex(arr,axis,float(period))

cpdef differentiate(np.ndarray arr,int axis=-1,double period=2*np.pi):   
    if np.isreal(arr).all():
        arr=arr.real
        if arr.dtype!=float: arr=arr*1.0
        return differentiate_real(arr,axis,float(period))
    else:
        if arr.dtype!=complex: arr=arr*1.0
        return differentiate_complex(arr,axis,float(period))
    
cpdef differentiate_real(np.ndarray arr,int axis=-1,double period=2*np.pi,discontinuous=False,double discontinuity=0):
    
    if arr.ndim==1: return differentiate_real_1D(arr,period=float(period),discontinuity=discontinuity)
    elif arr.ndim==2: return differentiate_real_2D(arr,axis=axis,period=float(period),discontinuity=discontinuity)

    cdef int i,N,Nk,ndim
    cdef np.ndarray arrf,coeff,derivf
    cdef np.ndarray deriv
    
    if not discontinuous: discontinuity=0
    
    arr=np.swapaxes(arr,0,axis)
    N=arr.shape[0]
    ndim=arr.ndim
    coeff=1j*(2*np.pi)*np.fft.rfftfreq(N,period/N)
    Nk=coeff.shape[0]
    shape=[Nk];shape.extend([1]*(ndim-1));
    coeff=coeff.reshape(tuple(shape))

    arrf=np.fft.rfft(arr-discontinuity,axis=0)
    
    derivf=arrf*coeff
    
    deriv=np.fft.irfft(derivf,axis=0).real
    
    deriv=np.swapaxes(deriv,0,axis)
    return deriv

cpdef differentiate_real_1D(np.ndarray arr,double period=2*np.pi,discontinuous=False,double discontinuity=0):

    cdef int i,N,Nk
    cdef np.ndarray[complex,ndim=1] arrf,coeff,derivf
    cdef np.ndarray[double,ndim=1] deriv
    
    if not discontinuous: discontinuity=0
    
    N=arr.shape[0]
    coeff=1j*(2*np.pi)*np.fft.rfftfreq(N,period/N)
    Nk=coeff.shape[0]

    arrf=np.fft.rfft(arr-discontinuity)
    
    derivf=arrf*coeff
    
    deriv=np.fft.irfft(derivf).real
    
    return deriv
    
cpdef differentiate_real_2D(np.ndarray arr,int axis=-1,double period=2*np.pi,discontinuous=False,double discontinuity=0):


    cdef int i,N,Nk,ndim
    cdef np.ndarray arrf,coeff,derivf
    cdef np.ndarray deriv
    
    if not discontinuous: discontinuity=0
    
    arr=np.swapaxes(arr,0,axis)
    N=arr.shape[0]
    ndim=arr.ndim
    coeff=1j*(2*np.pi)*np.fft.rfftfreq(N,period/N)
    Nk=coeff.shape[0]
    shape=[Nk];shape.extend([1]*(ndim-1));
    coeff=coeff.reshape(tuple(shape))

    arrf=np.fft.rfft(arr-discontinuity,axis=0)
    
    derivf=arrf*coeff
    
    deriv=np.fft.irfft(derivf,axis=0).real
    
    deriv=np.swapaxes(deriv,0,axis)
    return deriv

cpdef differentiate_complex(np.ndarray arr,int axis=-1,double period=2*np.pi,discontinuous=False,double discontinuity=0):

    cdef int i,N,Nk,ndim
    cdef np.ndarray arrf,coeff,derivf
    cdef np.ndarray deriv
    
    if not discontinuous: discontinuity=0
    
    arr=np.swapaxes(arr,0,axis)
    N=arr.shape[0]
    ndim=arr.ndim
    coeff=1j*(2*np.pi)*np.fft.fftfreq(N,period/N)
    Nk=coeff.shape[0]
    shape=[Nk];shape.extend([1]*(ndim-1));shape=tuple(shape)
    coeff=coeff.reshape(shape)

    arrf=np.fft.fft(arr-discontinuity,axis=0)
    
    derivf=arrf*coeff
    
    deriv=np.fft.ifft(derivf,axis=0).real
    
    deriv=np.swapaxes(deriv,0,axis)
    return deriv
