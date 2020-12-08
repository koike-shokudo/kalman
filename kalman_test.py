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


# data size 
datasize_N = 2000

df_dummy = np.zeros(datasize_N)
for i in range(datasize_N):
    df_dummy[i] = np.sin( float(i) * 0.05 ) #+ np.random.normal(0,0.1)

def KT_cal(r,psi,psi0):
    # KT parameters
    dtime = 0.1
    Tnomoto = 1.0
    Knomoto = -2.0
    coef = 0.0
    Kp = 2.0
    # cal defferential eqation
    '''defferential'''
    r_dot = ( Knomoto * Kp * ( psi - psi0 ) - r ) / Tnomoto
    psi_dot = r
    '''eular method'''
    r = r + r_dot * dtime
    psi = psi + psi_dot * dtime
    '''Cal AR coefficients'''
    a1 = 2 - dtime / Tnomoto
    a2 = dtime / Tnomoto + ( Knomoto * Kp * dtime ** 2 ) / Tnomoto - 1
    return r,psi,a1,a2

def lsm(y,u,n,m,omega,data_size): 
    y_vector = y[n:]        
    al0 = np.dot( omega.T,omega )
    al1 = np.linalg.inv( al0 )
    al2 = np.dot( al1, omega.T )
    alpha = np.dot( al2,y_vector )
    a  = alpha[:n]
    #b  = alpha[(n+1):]
    return a#,b

def pre_make_omega_matrix(yi,ui,yo,uo,arn,arm,data_size,icount):
    y_list = []
    u_list = []
    y_list.append(yi)
    u_list.append(ui)
    if  icount < (arn + data_size + 1)  or icount < (arm + data_size):
        yo.extend(y_list) 
        uo.extend(u_list)
    else:
        yo.extend(y_list) 
        uo.extend(u_list)
        del yo[0]
        del uo[0]
    return yo,uo

def make_omega_matrix(y,u,n,m,k):
    
    # input data
    '''y ... time serise data(array)
       u ... control input(array)
       n ... dimension of time serise data(integer)
       m ... dimension of control input(integer)
       k ... size of data set for calculate least(integer)
             squares method                  '''
    
    # out put data
    '''omega   ... matrix for calulate least squares method(array) 
       omega_t ... transfer matrix of omega(array)            '''

    # make time serise matrix
    omega = []
    for j in range(n):
        y_slice = y[ (n-j-1):( n + k  - j ) ]
        #y_slice = y_slice[::-1]
        omega.extend(np.array(y_slice)) 

    # make control matrix
    for j in range(m):
        u_slice = u[ (m+j-1):( m + k - j ) ]
        #u_slice = u_slice[::-1]
        omega.extend(np.array(u_slice))

    omega_t = np.array(omega)
    omega_t = omega_t.reshape((n+m),(k+1))
    omega = omega_t.T
    return omega,omega_t


# control term
#u_input = np.random.normal(0,0.3,datasize_N)
u_input  = list( range( (datasize_N + 1) , 2 * datasize_N ) )

'''KT_model'''
r = 0.1
psi = 30
psi0 = 0
t_list = np.arange(0, datasize_N, 1)
r_array = np.zeros(datasize_N)
psi_array = np.zeros(datasize_N)
psi_Rec_array = np.zeros(datasize_N)
ar_coef   = np.zeros((2,datasize_N)) 

'''AR_coef'''
# psi(n+2) = ( 2 - dtime / T ) * psi(n+1) + ( dtime / T + ( K * Kp * dtime^2) / T -1 ) * psi(n) - ( K * Kp * dtime^2) / T ) * psi0(n) 
# a1 = 2 - dtime / T
# a2 = dtime / T + ( K * Kp * dtime^2) / T -1

for i in range(datasize_N):
    r,psi,ar_coef[0,i],ar_coef[1,i] = KT_cal(r,psi,psi0)
    r_array[i] = r
    psi_array[i] = psi
    #df_dummy[i] = psi
    if i > 2 :
        psi_Rec_array[i] = ar_coef[0,i] * psi_array[i-1] + ar_coef[1,i] * psi_array[i-2]   

# plt.xlim( 0, datasize_N /100)
# plt.ylim( np.argmin(psi_array)*-2, np.argmin(psi_array)*2 )
# #plt.plot( t_list, r_array, color="green",label='r' )
# plt.plot( t_list, psi_array, color="red",label='psi' )
# plt.plot( t_list, psi_Rec_array, color="blue",label='psi_rec' )
# #plt.legend(box_to_anchor = (1.05, 1), shadow = True)
# plt.legend()
# plt.show()

df = pd.DataFrame({'a':df_dummy})

def kalman_filter(df, input):
    # set initial data
    Pi = np.pi
    qr     = 0.00001
    rr     = 0.00001
    ganma  = 10000
    #set size data
    n   = len(df)
    arn = 200
    # observe data
    y_input = df[input].values.tolist()
    y_input = list( range(n) ) 
    t_list  = np.arange(0, len(df) / 10, 0.1)
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
    rk = rk * c0
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
        an = np.dot(bb, np.linalg.inv(aa))
        
        #Calculate AIC
        var0 = np.var(y_input)
        for i in range(it):
            var   -= rk[i] * an[i] * var0
            #print(rk[i], fx[i])
        var = var0 - var
        aic[it] = arn * ( math.log(2.0 * Pi * var ) + 1.0 ) + 2.0 * (it + 1.0)
    #print(aic)
    #model= sm.tsa.AR(y_input)
    #arn = model.select_order(maxlag=10,ic='aic')
    #result  = model.fit(maxlag=2)
    #an_func = result.params
    arn = np.argmin(aic[1:arn]) + 1
    #print(an_func)
    # for jj in range( int(arn) ):
    #     an[jj] = an_func[jj]

    arn = 2
    arm = 1

    '''least squar method'''
    data_size = 10
    #y_set_data = np.zeros( ( arn + data_size + 1,1) )
    y_set_data = []
    u_set_data = []
    omega = []
    
    for i in range(n):
        # y_set_data = y_input[ i : ( i + arn + data_size ) ]
        # u_set_data = u_input[ i : ( i + arm + data_size ) ]
        y_set_data,u_set_data = pre_make_omega_matrix(y_input[i],u_input[i+1],y_set_data,u_set_data,arn,arm,data_size,i)
       
        if  i > (arn + data_size + 1)  or i > (arm + data_size):
            omega,omega_t = make_omega_matrix(y_set_data,u_set_data,arn,arm,data_size)
            print(type(omega))
            #an = lsm(y_set_data,0,arn,0,omega,data_size)
    
    print(an[0],ar_coef[0,0])
    print(an[1],ar_coef[1,0])

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
        #b[k, k] = 1.0#
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
    '''-------------------stady condition-----------------------'''
    #状態空間モデルを作る
    for ii in range(arn - 1):
        f[0, ii]    = an[ii]
        f[ii + 1, ii] = 1.0
    for j in range(n - arn):
        for k in range(arn):
            y[k] = y_input[j + arn - k]
        yy = np.dot(f, y)
        df_kalman[j, 0] = j
        df_kalman[j, 1] = y[0]
        df_kalman[j, 2] = yy[0]
    '''---------------------------------------------------------'''
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
        #print(K[0])
        xhat  = xhatm + np.dot(K, yy)
        p     = np.dot((e - np.dot(K, c)), p)
        return xhat, p,xhatm[0]
    for ii in range(n - arn -1 ):
        for j in range(arn):
            y[j] = y_input[ii + arn - j  ]#-1いらない？
        '''--------------------------Unstady condition--------------------------------'''
        #状態空間モデルを作る
        '''--------------------------least squar method-------------------------------'''
        # data_size = 10
        # y_set_data = np.zeros( n + data_size )
        # for j in range(arn+data_size):
        #     y_set_data[data_size+arn-j] = y_input[i+arn+data_size-j]
        # an = lsm(y_input,0,arn,0,data_size)
        # for kk in range(arn - 1):
        #     f[0, ii]    = an[kk]
        #     f[ii + 1, ii] = 1.0
        # for j in range(n - arn):
        #     for k in range(arn):
        #         y[k] = y_input[j + arn - k]
        #     yy = np.dot(f, y)
        #     df_kalman[j, 0] = j
        #     df_kalman[j, 1] = y[0]
        #     df_kalman[j, 2] = yy[0]
        '''----------------------------------------------------------------------------'''
        xhat, p,df_kalman[ii+arn+1, 2]  = kalman_linear(y, xhat, p, f, b, c, v, w, arn)
        df_kalman[ii+arn, 0] = ii
        df_kalman[ii+arn+1, 1] = xhat[0]
        #print(df_kalman[ii, 1],df_kalman[ii, 2])
    return df_kalman[:, 1],df_kalman[:, 2]

plt.xlim(0, 1000)
plt.ylim(-2, 2)
d1_kalman,d2_kalman = kalman_filter(df,'a')
t_list = np.arange(0, 2000, 1)
plt.plot(t_list, df_dummy, color="green",label='observed')
plt.plot(t_list, d1_kalman, color="red",label='Kalman_filter')
plt.plot(t_list, d2_kalman, color="blue",label='AR_model')
plt.legend(bbox_to_anchor = (1.05, 1), shadow = True)
plt.show()