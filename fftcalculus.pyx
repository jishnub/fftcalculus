#cython: boundscheck=False, wraparound=False
from __future__ import division
import numpy as np
cimport numpy as np
import cython

cpdef integrate_real(np.ndarray arr,int axis=-1,double period=2*np.pi):
    
    if arr.ndim==1: return integrate_real_1D(arr,period=float(period))
    elif arr.ndim==2: return integrate_real_2D(arr,axis=axis,period=float(period))

    cdef int i,N,Nk
    cdef np.ndarray arrf,coeff,integralf
    cdef np.ndarray integral
    cdef np.ndarray[double,ndim=1] x
    
    arr=np.swapaxes(arr,0,axis)
    N=arr.shape[0]
    coeff=1j*(2*np.pi)*np.fft.rfftfreq(N,period/N)
    Nk=coeff.shape[0]
    arravg=np.mean(arr,axis=0,keepdims=True)
    arrf=np.fft.rfft(arr-arravg,axis=0)
    integralf=np.zeros_like(arrf)
    
    x=np.linspace(0,period,N,endpoint=False)
    
    for i in xrange(1,Nk):  integralf[i]=arrf[i]/coeff[i]
    
    integral=np.fft.irfft(integralf,axis=0).real
    
    for i in xrange(N): integral[i]=integral[i]+x[i]*arravg
    
    integral=np.swapaxes(integral,0,axis)
    return integral
    
cpdef integrate_complex(np.ndarray arr,int axis=-1,double period=2*np.pi):

    cdef int i,N
    cdef np.ndarray arrf,coeff,integralf,integral
    cdef np.ndarray[double,ndim=1] x
    
    arr=np.swapaxes(arr,0,axis)
    N=arr.shape[0]
    arravg=np.sum(arr,axis=0,keepdims=True)/N
    arrf=np.fft.fft(arr-arravg,axis=0)
    integralf=np.zeros_like(arrf)
    coeff=1j*(2*np.pi)*np.fft.fftfreq(N,period/N)
    
    x=np.linspace(0,period,N,endpoint=False)
    
    for i in xrange(1,N):  integralf[i]=arrf[i]/coeff[i]
    
    integral=np.fft.ifft(integralf,axis=0)
    
    for i in xrange(N): integral[i]=integral[i]+x[i]*arravg
    
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
