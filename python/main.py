'''
Author: Okrio
Date: 2022-02-21 22:50:04
LastEditTime: 2022-02-22 23:27:27
LastEditors: Please set LastEditors
Description: nbss-aec refer to "Semi-Blind Source Separation for Nonlinear Acoustic Echo Cancellation"
FilePath: /bss-naec/matlab/main.py
'''
import numpy as np

import librosa as lib
# import soundfile as sf


def bss_naec(x, y, p, nfft, eta):
    """
    e : estimate of the near-end signal (L*1)
    W : demixing matrices (p+1 * p+1 * nfft/2+1)
    x : far-end input singal(L * 1)
    y : microphone signal(L * 1)
    p : expansion order 
    nfft : number of FFT points
    eta : learning rate
    """
    L = len(x)
    xp = np.zeros((L, p))
    for i in range(p):
        tmp = x**(2 * (i + 1) - 1)
        xp[:, i] = np.squeeze(tmp)

    # yxp = np.stack((y, xp), axis=-1)
    yxp = np.concatenate((y, xp), axis=-1)

    win = np.hanning(nfft)
    shift = nfft // 4
    N = int(np.fix((L + nfft) / shift) - 1) - 2
    nf = int(np.fix(nfft / 2) + 1)
    Y = np.zeros((p + 1, N, nf), dtype=np.complex128)

    for i in range(p + 1):
        tmp = lib.stft(
            yxp[:, i],
            n_fft=nfft,
            window=win,
            win_length=nfft,
            hop_length=shift,
        )
        Y[i, :, :] = tmp.transpose(1, 0)

    W = np.zeros((p + 1, p + 1, nf), dtype=np.complex128)
    En = np.zeros((p + 1, 1, nf), dtype=np.complex128)
    R = np.zeros(W.shape, dtype=np.complex128)
    dW = np.zeros(W.shape, dtype=np.complex128)
    E = np.zeros((nf, N), dtype=np.complex128)

    for k in range(nf):
        W[:, :, k] = np.eye(p + 1)
    # online sbss algorithm
    for n in range(N):
        Yn = Y[:, n, :]
        for k in range(nf):
            # tmp = W[:, :, k] * Yn[:, k]
            tmp = np.matmul(W[:, :, k], np.expand_dims(Yn[:, k], axis=-1))
            En[:, :, k] = tmp
        Ssq = np.sqrt(np.sum(abs(En)**2, axis=-1))
        Ssq1 = (Ssq + 1e-6)**-1
        for k in range(nf):
            # compute multivariate score function
            Phi = Ssq1 * En[:, :, k]

            # compute scaling factors
            # R[:,:,k] = np.matmul(Phi, np.transpose(np.conjugate()))
            # tt = En[:, :, k].transpose().conjugate()
            R[:, :, k] = np.matmul(Phi, En[:, :, k].transpose().conjugate())
            dk = np.sum(np.sum(abs(R[:, :, k]))) / (p + 1)
            ck = 1 / dk

            # update demixing matrices using constrained scaled natural gradient strategy
            dW[:, :, k] = (np.eye(p + 1) - R[:, :, k] / dk) * W[:, :, k]
            dW[1:p + 2, :, k] = 0
            W[:, :, k] = ck * (W[:, :, k] + eta * dW[:, :, k])
            W[1, :, k] = W[1, :, k] / W[1, 1, k]
            W[1:p + 1, 1:p + 1, k] = np.eye(p)
            t1 = np.expand_dims(W[1, :, k], axis=0)
            tt = np.expand_dims(Yn[:, k].transpose().conjugate(), axis=-1)
            t2 = np.matmul(t1, tt).squeeze()
            E[k, n] = t2

    # reconstruct the estimated near-end signal using inverse stft
    e = lib.istft(E, window=win, win_length=nfft, hop_length=shift, length=L)
    return e, W


if __name__ == "__main__":
    L = 2048
    p = 3
    nfft = 256
    x = np.random.random((L, 1))
    y = np.random.random((L, 1))
    bss_naec(x, y, p, nfft=nfft, eta=0.001)
