import pandas as pd
import pathlib 
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import fftpack
import sympy
import sys
import statsmodels.api as sm
import math
#sys.path.append('../')

# obtain current directry
current_path = os.getcwd()
print(current_path)
#import font_set
p_temp  =  list( pathlib.Path(current_path).glob('*.csv') )
df_1 = pd.read_csv(p_temp[0])
df_2 = pd.read_csv(p_temp[1])
df_3 = pd.read_csv(p_temp[2])
df_4 = pd.read_csv(p_temp[3])
df_5 = pd.read_csv(p_temp[4]) 
df_6 = pd.read_csv(p_temp[5]) 
df_7 = pd.read_csv(p_temp[6]) 
df_8 = pd.read_csv(p_temp[7]) 
df_1[['u_velo^2 [m/s]','vm_velo^2 [m/s]']] = df_1[['u_velo','vm_velo']] ** 2
df_2[['u_velo^2 [m/s]','vm_velo^2 [m/s]']] = df_2[['u_velo','vm_velo']] ** 2
df_3[['u_velo^2 [m/s]','vm_velo^2 [m/s]']] = df_3[['u_velo','vm_velo']] ** 2
df_4[['u_velo^2 [m/s]','vm_velo^2 [m/s]']] = df_4[['u_velo','vm_velo']] ** 2
df_5[['u_velo^2 [m/s]','vm_velo^2 [m/s]']] = df_5[['u_velo','vm_velo']] ** 2
df_6[['u_velo^2 [m/s]','vm_velo^2 [m/s]']] = df_6[['u_velo','vm_velo']] ** 2
df_7[['u_velo^2 [m/s]','vm_velo^2 [m/s]']] = df_7[['u_velo','vm_velo']] ** 2
df_8[['u_velo^2 [m/s]','vm_velo^2 [m/s]']] = df_8[['u_velo','vm_velo']] ** 2
df_1['velo [m/s]'] = np.sqrt(df_1['u_velo^2 [m/s]'] + df_1['vm_velo^2 [m/s]'])
df_2['velo [m/s]'] = np.sqrt(df_2['u_velo^2 [m/s]'] + df_2['vm_velo^2 [m/s]'])
df_3['velo [m/s]'] = np.sqrt(df_3['u_velo^2 [m/s]'] + df_3['vm_velo^2 [m/s]'])
df_4['velo [m/s]'] = np.sqrt(df_4['u_velo^2 [m/s]'] + df_4['vm_velo^2 [m/s]'])
df_5['velo [m/s]'] = np.sqrt(df_5['u_velo^2 [m/s]'] + df_5['vm_velo^2 [m/s]'])
df_6['velo [m/s]'] = np.sqrt(df_6['u_velo^2 [m/s]'] + df_6['vm_velo^2 [m/s]'])
df_7['velo [m/s]'] = np.sqrt(df_6['u_velo^2 [m/s]'] + df_7['vm_velo^2 [m/s]'])
df_8['velo [m/s]'] = np.sqrt(df_6['u_velo^2 [m/s]'] + df_8['vm_velo^2 [m/s]'])


def lowpass(data):
    N = len(data)    # サンプル数
    dt = 0.1          # サンプリング間隔
    fq1 = 1    # 周波数
    fc = 0.1 # カットオフ周波数
    t = np.arange(0, N*dt, dt)  # 時間軸
    freq = np.linspace(0, 1.0/dt, N)  # 周波数軸
    f = data
    #F = fftpack.fft(f)
    F = np.fft.fft(f,n=N)
    # 正規化 + 交流成分2倍
    F = F/(N/2)
    F[0] = F[0]/2
    # 配列Fをコピー
    F2 = F.copy()
    # ローパスフィルタ処理（カットオフ周波数を超える帯域の周波数信号を0にする）
    F2[(freq > fc)] = 0
    # 高速逆フーリエ変換（時間信号に戻す）
    f2 = fftpack.ifft(F2)
    # 振幅を元のスケールに戻す
    f2 = np.real(f2*N)
    return f2
def kalman_filter(df, input):
    # set initial data
    Pi = np.pi
    qr     = 0.09
    rr     = 5
    ganma  = 0.2
    #set size data
    n   = len(df)
    arn = 200
    # observe data
    y_input = df[input].values.tolist()
    t_list = np.arange(0, len(df) / 10, 0.1)
    '''標本自己共分散を求める
    ck:自己共分散関数 ，rk:自己相関関数'''
    vv = np.random.normal(loc = 0, scale = 0.5, size = n)
    # set initial conditions
    ck  = np.zeros(arn)
    rk  = np.zeros(arn)
    aic = np.zeros(arn)
    c0  = np.var(y_input) #variance
    k   = 0
    t   = range(arn + 1)
    # calculate mean value
    myu = sum(y_input) / n
    # calculate autocorrelation function
    rk = sm.tsa.stattools.acf(y_input, nlags = arn)
    k = 0
    # drow figure
    # plt.plot(t, rk)
    # plt.show()
    # print(rk)
    ''' calculate Yule-wakaer equation'''
    for it in range(arn):
        k   += 1
        aa  = np.zeros((k, k))
        bb  = np.zeros(k)
        an  = np.zeros((k, k))
        dd  = np.zeros(k)
        var = 0.0
        k   = 0
        l   = arn
        aa[0,0] = rk[0]
        bb[0]   = rk[1]
        #rangeの中を変数にした方がいい
        for i in range(it):
            for j in range(it - k):
                aa[i, j + k] = rk[j]
            k += 1
            l = l - 1
            bb[i] = rk[i + 1]
        #print(bb)
        for i in range(it):
            for j in range(it):
                aa[j, i] = aa[i, j]
        #行列計算により連立方程式を解いた
        an  = np.dot(bb, np.linalg.inv(aa))
        #Calculate AIC
        for i in range(it):
            var   -= rk[i] * an[i]
            #print(rk[i], fx[i])
        var = an[0] - var
        aic[it] = arn * math.log(2.0 * Pi * var + 1.0) + 2.0 * (it + 1.0)
    #print(aic)
    arn = np.argmin(aic[1:arn]) + 1
    arn = 3
    # print(arn)
    # print(aic)
    # plt.plot(t, aic)
    # plt.show()
    #初期値の設定
    a  = 0.0
    r  = 1 #number of data
    s  = arn
    b  = np.zeros((arn, r))
    #c  = np.zeros((arn, arn))
    c  = np.zeros((s, arn))
    e  = np.eye(arn, arn)
    x  = np.zeros(arn)
    y  = np.zeros((arn, 1))
    v  = np.eye(r, r)
    w  = np.eye(s, s)
    #制御量の設定
    b[:]= 1.0
    for k in range(arn):
        c[k, k] = 1.0
    #予測ステップ
    #k         = 100
    e         = np.eye(arn, arn)
    p         = np.zeros((arn, arn))
    f         = np.zeros((arn, arn))
    xhat      = np.zeros((arn, 1))
    bu        = np.eye(arn)
    u         = np.eye(arn)
    #誤差共分散の初期値設定
    p = e * ganma
    df_kalman = np.zeros((n, 3)) #set output array
    #状態空間モデルを作る
    for ii in range(arn - 1):
        f[0, ii]    = an[ii]
        f[ii + 1, ii] = 1.0
    for j in range(n - arn):
        for k in range(arn - 1):
            y[k] = y_input[j + arn - k]
        yy = np.dot(f, y)
        df_kalman[j, 0] = j
        df_kalman[j, 1] = y[0]
        df_kalman[j, 2] = yy[0]
    # fig = plt.figure(figsize = (15, 11))
    # plt.xlim(0, 200)
    # plt.plot(df_kalman[:, 0],df_kalman[:, 1])
    # plt.plot(df_kalman[:, 0],df_kalman[:, 2])
    # plt.show()
    #ARモデルによる線形カルマンフィルタの関数
    w = w * rr
    v = v * qr
    #kalman filter
    def kalman_linear(y, xhat, p, f, b, c, v, r, n):
        #estimate step
        xhatm = np.zeros((n, 1))
        pm    = np.zeros((n, n))
        xhatm = np.dot(f, xhat)
        pm0   = np.dot(np.dot(b, v), b.T)
        pm    = np.dot(np.dot(f, p), f.T) + pm0
        #filteling step
        yy    = y - np.dot(c, xhatm)
        #r0    = np.eye(n,n) * r[0]
        ss    = np.dot(c, pm)
        S     = np.dot(ss, c.T) + r
        K     = np.dot(np.dot(pm, c.T), np.linalg.inv(S))
        xhat  = xhatm + np.dot(K, yy)
        p     = np.dot((e - np.dot(K, c)), p)
        return xhat, p
    for ii in range(n - arn):
        for j in range(arn - 1):
            y[j] = y_input[ii + arn - j]
        df_kalman[ii, 2] = y[0]
        xhat, p = kalman_linear(y, xhat, p, f, b, c, v, w, arn)
        df_kalman[ii, 0] = ii
        df_kalman[ii, 1] = xhat[0]
        #print(df_kalman[ii, 1],df_kalman[ii, 2])
    return df_kalman[:, 1]
def a_b_filter(df, input):
    # intial condition
    n     = len(df)
    dt    = 0.1
    xs    = 0.0
    xs0   = 0.0
    vs    = 0.0
    vs0   = 0.0
    vp0   = 0.0
    alpha = 0.0
    beta  = 0.0
    x     = np.zeros(n)
    t     = np.zeros(n)
    xs0   = np.zeros(n)
    vs0   = np.zeros(n)
    #set dumy data
    x = df[input].values.tolist()
    # filter function
    def alpha_beta_filter(xo, xs0, vs0, vp0, t, alpha, beta):
        #estimate step
        xp = xs0 + t * vs0
        vp = (xp - xs0) / t
        #filteling step
        xs = xp + alpha * (xo - xp)
        vs = vp0 + beta * (xo - xp) / t
        return xs,vs,vp
    # set alpha,beta value
    for i in range(n):
    #     alpha      = 2 * ( 2 * i + 1 ) / ( (i+1) * (i+2) )
    #     beta       = 6 / ( (i+1) * (i+2) )
    #    alpha      = 0.1
        beta       = 0.004
        alpha = beta * -0.5 + np.sqrt(beta)
        xs,vs,vp0  = alpha_beta_filter(x[i], xs, vp0, vs, dt, alpha, beta)
        xs0[i]     = xs
    return xs0
def compare(df):
    t_list = np.arange(0, len(df)/10, 0.1)
    dif_list_U     = df['velo [m/s]']
    gps_list_U_raw = df['U_gps_raw_midship'].values.tolist()
    gps_list_U_lp  = lowpass(gps_list_U_raw)
    gps_list_U_ab  = a_b_filter(df, 'U_gps_raw_midship')
    gps_list_U_kalman  = kalman_filter(df, 'U_gps_raw_midship')
    dif_list_u  = df['u_velo']
    gps_list_u_raw = df['u_gps_raw_midship'].values.tolist()
    gps_list_u_lp  = lowpass(gps_list_u_raw)
    gps_list_u_ab  = a_b_filter(df, 'u_gps_raw_midship')
    gps_list_u_kalman  = kalman_filter(df, 'u_gps_raw_midship')
    dif_list_vm = df['vm_velo']
    gps_list_vm_raw = df['vm_gps_raw_midship'].values.tolist()
    gps_list_vm_lp  = lowpass(gps_list_vm_raw)
    gps_list_vm_ab = a_b_filter(df, 'vm_gps_raw_midship')
    gps_list_vm_kalman = kalman_filter(df, 'vm_gps_raw_midship')
    fig = plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    #plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.xlim(0, len(df)/10)
    plt.ylim(min(dif_list_u), max(dif_list_u) + 0.05)
    plt.plot(t_list, gps_list_u_raw, color = "#4FC3F7", label = 'u_GPS_raw', alpha = 0.6)
#     plt.plot(t_list, gps_list_u_lp, color="#0277BD", label = "u_GPS_lp")
#     plt.plot(t_list, dif_list_u, color = "red", label = "u_dif")
    plt.plot(t_list, gps_list_u_kalman, color="red", label = "u_GPS_kf")
    plt.plot(t_list, gps_list_u_ab, color="green", label = "u_GPS_αβ")
    plt.xlabel('$\it{t [s]}$', fontsize = 30)
    plt.ylabel('$\it{u [m/s]}$', fontsize = 30)
    plt.legend(bbox_to_anchor = (1.05, 1), shadow = True)
    plt.xticks(np.arange(0, len(df)/10, 20))
    plt.yticks(np.arange(min(dif_list_u), max(dif_list_u) + 0.05, 0.05))
    plt.tight_layout()
    plt.subplot(3, 1, 2)
    #plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.xlim(0, len(df)/10)
    plt.ylim(-0.2, 0.2)
    plt.plot(t_list, gps_list_vm_raw, color = "#4FC3F7", label = 'vm_GPS_raw', alpha = 0.6)
#     plt.plot(t_list, gps_list_vm_lp, color="#0277BD", label = "vm_GPS_lp")
#     plt.plot(t_list, dif_list_vm, color="red", label = "vm_dif")
    plt.plot(t_list, gps_list_vm_kalman, color="red", label = "vm_GPS_kf")
    plt.plot(t_list, gps_list_vm_ab, color="green", label = "vm_GPS_αβ")
    plt.xlabel('$\it{t [s]}$', fontsize = 30)
    plt.ylabel('$\it{vm [m/s]}$', fontsize = 30)
    plt.legend(bbox_to_anchor = (1.05, 1), shadow = True)
    plt.xticks(np.arange(0, len(df)/10, 20))
    plt.yticks(np.arange(-0.2, 0.2, 0.03))
    plt.tight_layout()
    plt.subplot(3, 1, 3)
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.xlim(0, len(df)/10)
    plt.ylim(0, max(gps_list_U_raw) + 0.05)
    plt.plot(t_list, gps_list_U_raw, color = "#4FC3F7", label = 'U_GPS_raw', alpha = 0.6)
    #plt.plot(t_list, gps_list_U_lp, color="#0277BD", label = "U_GPS_lp")
    #plt.plot(t_list, dif_list_U, color="red", label = "U_dif")
    plt.plot(t_list, gps_list_U_kalman, color="red", label = "U_GPS_kf")
    plt.plot(t_list, gps_list_U_ab, color="green", label = "U_GPS_αβ")
    plt.xlabel('$\it{t [s]}$', fontsize = 30)
    plt.ylabel('$\it{U [m/s]}$', fontsize = 30)
    plt.legend(bbox_to_anchor = (1.05, 1), shadow = True)
    plt.xticks(np.arange(0, len(df)/10, 20))
    plt.yticks(np.arange(0, max(dif_list_U) + 0.05, 0.1))
    plt.tight_layout()
    plt.show()
compare(df_1)


