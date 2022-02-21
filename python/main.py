'''
Author: Okrio
Date: 2022-02-21 22:50:04
LastEditTime: 2022-02-21 23:49:19
LastEditors: Please set LastEditors
Description: nbss-aec
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
    Y = np.zeros((p + 1, N, nf))

    for i in range(p + 1):
        tmp = lib.stft(
            yxp[:, i],
            n_fft=nfft,
            window=win,
            win_length=nfft,
            hop_length=shift,
        )
        Y[i, :, :] = tmp.transpose(1, 0)

    W = np.zeros((p + 1, p + 1, nf))
    En = np.zeros((p + 1, 1, nf))
    R = np.zeros(np.size(W))
    dW = np.zeros(np.size(W))
    E = np.zeros((nf, N))

    for k in range(nf):
        W[:, :, k] = np.eye(p + 1)

    for n in range(N):
        Yn = Y[:, n, :]
        for k in range(nf):
            En[:, :, k] = W[:, :, k] * Yn[:, :, k]
        Ssq = np.sqrt(np.sum(abs(En)**2, axis=-1))
        Ssq1 = (Ssq + 1e-6)**-1
        for k in range(nf):
            Phi = Ssq1 * En[:, :, k]

            R[:, :, k] = np.matmul(Phi, En[:, :, k].transpose().conj())
            dk = np.sum(np.sum(abs(R[:, :, k]))) / (p + 1)
            ck = 1 / dk

            dW[:, :, k] = (np.eye(p + 1) - R[:, :, k] / dk) * W[:, :, k]
            dW[1:p + 2, :, k] = 0
            W[:, :, k] = ck * (W[:, :, k] + eta * dW[:, :, k])
            W[1, :, k] = W[1, :, k] / W[1, 1, k]
            W[1:p + 2, 1:p + 2, k] = np.eye(p)
            E[k, n] = W[1, :, k] * Yn[:, :, k]
    e = lib.istft(E, window=win, win_length=nfft, hop_length=shift, length=L)
    return e, W


if __name__ == "__main__":
    L = 2048
    p = 3
    nfft = 256
    x = np.random.random((L, 1))
    y = np.random.random((L, 1))
    bss_naec(x, y, p, nfft=nfft, eta=0.001)
