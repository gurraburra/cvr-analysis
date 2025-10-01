#%%
import numpy as np
from process_control import ProcessNode
from numpy.fft import rfft, irfft, rfftfreq
#%%
class TauDelay(ProcessNode):
    outputs = ("tau", "delay", "beta", "sse", "r2")

    def _run(self, signals, probe, confounds, tau_min, tau_max, lag_min, lag_max, time_step, nr_tau = 40, tau_log_space = True, phat = False):
        Ts, M = signals.shape; Tp = len(probe)
        # resid
        if confounds is not None:
            Q, _ = np.linalg.qr(confounds, mode='reduced')
            def resid(v): return v - Q @ (Q.T @ v)
        else:
            def resid(v): return v

        signals_res = resid(signals)
        E = np.sum(signals_res**2, axis=0)

        # fft
        N0 = Tp + Ts - 1
        N = int(2**np.ceil(np.log2(N0)))
        FFT_signals = rfft(signals_res[::-1], n = N, axis=0)
        FFT_probe = rfft(probe, n = N); 
        w = 2*np.pi*rfftfreq(N, d=1.0)

        # lags
        lag_idx = np.arange(Ts-1, -Tp, -1) * time_step
        lag_mask = np.ones(len(lag_idx),dtype=bool)
        if lag_min is None:
            lag_min = (1 - Tp) * time_step
        if lag_max is None:
            lag_max = (Ts - 1) * time_step
        lag_mask[lag_idx <= lag_min] = False
        lag_mask[lag_idx >= lag_max] = False
        if np.all(~lag_mask):
            lag_mask[np.argmin(np.abs(lag_idx - (lag_min + lag_max) / 2))] = True
        lag_idx_mask = lag_idx[lag_mask]

        # taus
        if tau_log_space:
            taus = np.logspace(np.log10(tau_min), np.log10(tau_max), nr_tau)
        else:
            taus = np.linspace(tau_min, tau_max, nr_tau)
        
        # best values
        best_sse = np.full(M, np.inf)
        best_tau = np.full(M, np.nan)
        best_lag = np.full(M, np.nan)
        best_beta = np.full(M, np.nan)

        # loop taus
        for tau in taus:
            if np.isclose(tau,0):
                FFT_conv = FFT_probe
            else:
                # convolve hrf (exp)
                a = np.exp(-time_step/tau)
                H = 1.0/(1.0 - a*np.exp(-1j*w)) * (1 - a)
                FFT_conv = FFT_probe * H
            s_tau = irfft(FFT_conv, n = N)[:Tp]
            s_tilde = resid(s_tau)
            G_tilde = np.dot(s_tilde, s_tilde)
            if G_tilde < 1e-12: continue
            FFT_s = rfft(s_tilde, n = N)
            
            # convolve signals
            FFT_delay = FFT_s[:,None] * FFT_signals
            XC_full = irfft(FFT_delay, n=N, axis=0)[:N0]
            XC = XC_full[lag_mask,:]
            # _XC is used to find delay, XC is used to find correlation
            if phat:
                weight = np.abs(FFT_delay)
                weight[weight < 1e-6] = 1e-6
                _FFT_delay = FFT_delay / weight
                _XC_full = irfft(_FFT_delay, n=N, axis=0)[:N0]
                _XC = _XC_full[lag_mask,:]   
            else:
                _XC = XC

            # find best
            k0 = np.argmax(np.abs(_XC), axis=0)
            r_star = XC[k0, np.arange(M)]
            lag_hat = lag_idx_mask[k0]

            # compute fit
            beta = np.real(r_star) / G_tilde
            sse = E - np.abs(r_star)**2 / G_tilde

            # improve
            improve = sse < best_sse
            best_sse[improve] = sse[improve]
            best_tau[improve] = tau
            best_lag[improve] = lag_hat[improve]
            best_beta[improve] = beta[improve]

        return best_tau, best_lag, best_beta, best_sse, 1 - best_sse / E
