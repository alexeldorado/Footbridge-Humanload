# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 13:02:30 2021

@author: Alex Almeida
"""
import numpy as np
import scipy.linalg as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import pandas as pd
from pandas import ExcelWriter

#-------------------------------------------------------------------------    
#1. Dados de Entrada  
#-------------------------------------------------------------------------  

nos = pd.read_excel('dados_de_entrada_final.xlsx','nos')
barras = pd.read_excel('dados_de_entrada_final.xlsx','barras')
M_lajes = pd.read_excel('dados_de_entrada_final.xlsx','lajes')

#-------------------------------------------------------------------------     
#2. Vetor de coordenadas ( X e Y dos nós) 
#------------------------------------------------------------------------- 

nn  = len(list(nos['Cx']))     # número de nós
ngl = len(list(nos['Cx']))*2   # número de graus de liberdade 
cx  = list(nos['Cx'])[0:nn]    # coordenada X
cy  = list(nos['Cy'])[0:nn]    # coordenada Y
cz  = list(nos['Cz'])[0:nn]    # coordenada Z

#-------------------------------------------------------------------------     
#3. Matriz de identificação dos nós( Nó inicial (N) e Nó Final (F) de cada barra)
#-------------------------------------------------------------------------  

noN     = list(barras['noN'])  #Nós Iniciais
noF     = list(barras['noF'])  #Nós Finais
nb      = len(noN)             #Número de Barras
IDN      = np.zeros((2,nb))
IDN[0,:] = noN
IDN[1,:] = noF

#-------------------------------------------------------------------------     
#4. Propriedades de cada barra
#-------------------------------------------------------------------------  

A   = list(barras['Area(m2)'])
RHO = list(barras['Densidade(kg/m³)'])
E   = 2.0*10**11                        ##[N/m2]

#------------------------------------------------------------------------- 
#5. Matriz de identificação das Barras em relação aos Graus de Liberdade
#-------------------------------------------------------------------------  

IDB = np.zeros((4,nb)) #Graus de liberdade por nó, número de barras

for i in range(2):

    IDB[i,:]   = IDN[0,:]*2-1+i
    IDB[i+2,:] = IDN[1,:]*2-1+i

#-------------------------------------------------------------------------        
#6. Comprimento de cada barra e cossenos diretores
#-------------------------------------------------------------------------  

Lx   = np.zeros(nb)
Ly   = np.zeros(nb)
cosx = np.zeros(nb)
cosy = np.zeros(nb)
L    = np.zeros(nb)

for n in range (nb):

    k1      = int(IDN[0,n] -1)  # Indexador da matriz IDN
    k2      = int(IDN[1,n] -1)  # Indexador da matriz IDN
    Lx[n]   = cx[k2] - cx[k1]
    Ly[n]   = cy[k2] - cy[k1]
    L[n]    = np.sqrt(Lx[n]**2 + Ly[n]**2)
    cosx[n] = Lx[n]/L[n]
    cosy[n] = Ly[n]/L[n]
    
    
#-------------------------------------------------------------------------        
#7. Matrizes de massa e rigidez
#-------------------------------------------------------------------------  

K = np.zeros((ngl,ngl))
M = np.zeros((ngl,ngl))

for i in range (nb):

#7.1 Matriz de rigidez local da barra
   
    k = np.array([[E*A[i]/L[i], 0, -E*A[i]/L[i], 0],
                  [0, 0, 0, 0], 
                  [-E*A[i]/L[i], 0, E*A[i]/L[i], 0],
                  [0, 0, 0, 0]])
                     
#7.2 Matriz de massa local da barra
    
    m = ((RHO[i]*A[i]*L[i])/6)*np.array([[2,0,1,0],
                                         [0,2,0,1],
                                         [1,0,2,0],
                                         [0,1,0,2]])
               
#7.3, Matriz de rotação 
    
    tau = np.array([[ cosx[i], cosy[i], 0, 0],
                    [-cosy[i], cosx[i], 0, 0],
                    [0, 0,  cosx[i], cosy[i]],
                    [0, 0, -cosy[i], cosx[i]]])
                   
#7.4 Matrizes locais rotacionadas

    k_r = np.dot(np.dot(tau.T, k),tau)  
    m_r = np.dot(np.dot(tau.T, m),tau)
    
#7.5 Alocação das matrizes locais na matriz global

    k_rG = np.zeros((ngl,ngl))    
    m_rG = np.zeros((ngl,ngl))

    a1 = int(IDB[0,i]-1)
    a2 = int(IDB[1,i])
    a3 = int(IDB[2,i]-1)
    a4 = int(IDB[3,i])

    k_rG[a1:a2,a1:a2] = k_r[0:2,0:2]
    k_rG[a3:a4,a1:a2] = k_r[2:5,0:2]
    k_rG[a1:a2,a3:a4] = k_r[0:2,2:5]
    k_rG[a3:a4,a3:a4] = k_r[2:5,2:5]

    K += k_rG 
    
    m_rG[a1:a2,a1:a2] = m_r[0:2,0:2]
    m_rG[a3:a4,a1:a2] = m_r[2:5,0:2]
    m_rG[a1:a2,a3:a4] = m_r[0:2,2:5]
    m_rG[a3:a4,a3:a4] = m_r[2:5,2:5]

    M += m_rG
    
#7.6 Somando a massa das Lajes à Matriz de Massa (Consistente + Lumped)

ML = M_lajes.values
M = M+ML

#-------------------------------------------------------------------------     
#1. Montar array com os Graus de Liberdade Restringidos 
#------------------------------------------------------------------------- 

gl         = np.array(list(nos['u'])+list(nos['v']))
id_glr     = np.array(list(nos['ur'])+list(nos['vr']))
glr        = np.trim_zeros(sorted(gl*id_glr))
remover_gl = np.array(glr)-1

#-------------------------------------------------------------------------     
#2. Deletar Linhas e Colunas restritas das matrizes K e M
#------------------------------------------------------------------------- 

Ki = np.delete(K, remover_gl,axis=0)
Kf = np.delete(Ki, remover_gl,axis=1)

Mi = np.delete(M, remover_gl, axis=0)
Mf = np.delete(Mi, remover_gl, axis=1)

#-------------------------------------------------------------------------
#1. Autovalores e autovetores
#-------------------------------------------------------------------------

lamb,Phi  = sc.eig(Kf,Mf)
index_eig = lamb.argsort()    #indexando em ordem crescente
lamb      = lamb[index_eig]   #Aplicando indexador
Phi       = Phi[:,index_eig]  #Aplicando indexador
w2        = np.real(lamb)     #Extraindo apenas a parte real de lambda
wk        = np.sqrt(w2)       #rad/s
fk        = wk/2/np.pi        #Hz 

    # Kf : Matriz de rigidez restringida
    # Mf : Matriz de rigidez restringida
    # wk : Frequencias naturais(rad/s) do primeiro e segundo modos

#-------------------------------------------------------------------------
#1. Coeficientes do sistema de Rayleigh
#------------------------------------------------------------------------- 

z = 0.4/100                        #Razão de amortecimento típica para passarela de aço

a = z*2*wk[0]*wk[1]/(wk[0]+wk[1])  # parâmetro da matriz de massa
b = z*2/(wk[0]+wk[1])              # parâmetro da matriz de rigidez

Cf = a*Mf + b*Kf

#------------------------------------------------------------------------- 
#1. Dados Necessários
#------------------------------------------------------------------------- 
# Caminha [C] e corrida [S] de uma pessoa: frequência, velocidade e comprimento de passo.

C_normal = [2.0, 1.5, 0.75]

G = 800.00               #peso em Newton de uma pessoa de 80 kg.
dG1 = 0.4*G
dG2 = dG3 = 0.1*G
fase2 = fase3 = np.pi/2

fs = C_normal[0]
Vs = C_normal[1]
Ls = C_normal[2]

tp = Ls/Vs


Lf = 33.62

T0 = Lf/Vs

#------------------------------------------------------------------------- 
# Parâmetros de análise
#-------------------------------------------------------------------------  
Tc  = 7.5867                         #Tempo de amortecimento livre (s)  
dt  = 0.005                          #Passo de tempo de integração
t   = np.arange(0,(T0+Tc+dt),dt)     #lista do tempo de Integração
tf  = int(len(t))                    #Tempo Final

#-------------------------------------------------------------------------
# Determinar o Conjunto vetorial de distribuição de forças
#-------------------------------------------------------------------------

x1  = len(gl)-len(glr)
y1  = len(t)
Fa1 = np.zeros((x1,y1))     #Conjunto vetorial de distribuição de forças

ti = list(t)

for i in (t):    
    if 0.5 <= i <1:
        i1=ti.index(i)
        Fa1[1,i1]=0.55
    elif 1 <= i < 1.5:
        i2=ti.index(i)
        Fa1[1,i2]=0.935
        Fa1[3,i2]=0.065
    elif 1.5 <= i < 2:
        i3=ti.index(i)
        Fa1[1,i3]=0.56
        Fa1[3,i3]=0.44
    elif 2 <= i < 2.5:
        i4=ti.index(i)
        Fa1[1,i4]=0.185
        Fa1[3,i4]=0.815
    elif 2.5 <= i < 3:
        i5=ti.index(i)
        Fa1[3,i5]=0.81
        Fa1[5,i5]=0.19
    elif 3 <= i < 3.5:
        i6=ti.index(i)
        Fa1[3,i6]=0.435
        Fa1[5,i6]=0.565
    elif 3.5 <= i < 4:
        i7=ti.index(i)
        Fa1[3,i7]=0.006
        Fa1[5,i7]=0.94
    elif 4 <= i < 4.5:        
        i8=ti.index(i)
        Fa1[5,i8]=0.685
        Fa1[7,i8]=0.315
    elif 4.5 <= i < 5:
        i9=ti.index(i)
        Fa1[5,i9]=0.31
        Fa1[7,i9]=0.69
    elif 5 <= i < 5.5:
        i10=ti.index(i)
        Fa1[7,i10]=0.935
        Fa1[9,i10]=0.065
    elif 5.5 <= i < 6:
        i11=ti.index(i)
        Fa1[7,i11]=0.56
        Fa1[9,i11]=0.44
    elif 6 <= i < 6.5:        
        i12=ti.index(i)
        Fa1[7,i12]=0.185
        Fa1[9,i12]=0.815
    elif 6.5 <= i < 7:
        i13=ti.index(i)
        Fa1[9,i13]=0.81
        Fa1[11,i13]=0.19
    elif 7 <= i < 7.5:
        i14=ti.index(i)
        Fa1[9,i14]=0.435
        Fa1[11,i14]=0.565
    elif 7.5 <= i < 8: 
        i15=ti.index(i)
        Fa1[9,i15]=0.06
        Fa1[11,i15]=0.94        
    elif 8 <= i < 8.5:
        i16=ti.index(i)
        Fa1[11,i16]=0.685
        Fa1[13,i16]=0.315
    elif 8.5 <= i < 9:
        i17=ti.index(i)
        Fa1[11,i17]=0.31
        Fa1[13,i17]=0.69
    elif 9 <= i < 9.5: 
        i18=ti.index(i)
        Fa1[13,i18]=0.935
        Fa1[15,i18]=0.065
    elif 9.5 <= i < 10:
        i19=ti.index(i)
        Fa1[13,i19]=0.56
        Fa1[15,i19]=0.44
    elif 10 <= i < 10.5:
        i20=ti.index(i)
        Fa1[13,i20]=0.185
        Fa1[15,i20]=0.815
    elif 10.5 <= i < 11:      
        i21=ti.index(i)
        Fa1[15,i21]=0.81
        Fa1[17,i21]=0.19
    elif 11 <= i < 11.5:
        i22=ti.index(i)
        Fa1[15,i22]=0.435
        Fa1[17,i22]=0.565
    elif 11.5 <= i < 12:
        i23=ti.index(i)
        Fa1[15,i23]=0.06
        Fa1[17,i23]=0.94
    elif 12 <= i < 12.5: 
        i24=ti.index(i)
        Fa1[17,i24]=0.685
        Fa1[19,i24]=0.315
    elif 12.5 <= i < 13:
        i25=ti.index(i)
        Fa1[17,i25]=0.31
        Fa1[19,i25]=0.69
    elif 13 <= i < 13.5:
        i26=ti.index(i)
        Fa1[19,i26]=0.935
        Fa1[21,i26]=0.065
    elif 13.5 <= i < 14:
        i27=ti.index(i)
        Fa1[19,i27]=0.56
        Fa1[21,i27]=0.44
    elif 14 <= i < 14.5:        
        i28=ti.index(i)
        Fa1[19,i28]=0.185
        Fa1[21,i28]=0.815
    elif 14.5 <= i < 15:
        i29=ti.index(i)
        Fa1[21,i29]=0.81
        Fa1[23,i29]=0.19
    elif 15 <= i < 15.5:
        i30=ti.index(i)
        Fa1[21,i30]=0.435
        Fa1[23,i30]=0.565
    elif 15.5 <= i < 16:
        i31=ti.index(i)
        Fa1[21,i31]=0.06
        Fa1[23,i31]=0.94
    elif 16 <= i < 16.5: 
        i32=ti.index(i)
        Fa1[23,i32]=0.685
        Fa1[25,i32]=0.315
    elif 16.5 <= i < 17:
        i33=ti.index(i)
        Fa1[23,i33]=0.31
        Fa1[25,i33]=0.69
    elif 17 <= i < 17.5:
        i34=ti.index(i)
        Fa1[25,i34]=0.935
        Fa1[27,i34]=0.065
    elif 17.5 <= i < 18:
        i35=ti.index(i)
        Fa1[25,i35]=0.56
        Fa1[27,i35]=0.44
    elif 18 <= i < 18.5:        
        i36=ti.index(i)
        Fa1[25,i36]=0.185
        Fa1[27,i36]=0.815
    elif 18.5 <= i < 19:
        i37=ti.index(i)
        Fa1[27,i37]=0.81
        Fa1[29,i37]=0.19
    elif 19 <= i < 19.5:
        i38=ti.index(i)
        Fa1[27,i38]=0.435
        Fa1[29,i38]=0.565
    elif 19.5 <= i < 20:
        i39=ti.index(i)
        Fa1[27,i39]=0.06
        Fa1[29,i39]=0.94
    elif 20 <= i < 20.5:        
        i40=ti.index(i)
        Fa1[29,i40]=0.685
        Fa1[31,i40]=0.315
    elif 20.5 <= i < 21:
        i41=ti.index(i)
        Fa1[29,i41]=0.31
        Fa1[31,i41]=0.69
    elif 21 <= i < 21.5:
        i42=ti.index(i)
        Fa1[31,i42]=0.942        
    elif 21.5 <= i < 22:        
        i43=ti.index(i)
        Fa1[31,i43]=0.6088        
    elif 22 <= i < 22.5:        
        i44=ti.index(i)
        Fa1[31,i44]=0.2755

#-------------------------------------------------------------------------
#1. Determinar a Função de Força do caminhar humano (01 pessoa)
#------------------------------------------------------------------------- 
x = len(t)

Fp1 = []
for k in range(x):    
    Fp0 = G + dG1*np.sin(2*np.pi*fs*t[k])+dG2*np.sin(4*np.pi*fs*t[k] - fase2)+dG3*np.sin(6*np.pi*fs*t[k] - fase3)
    Fp1.extend([Fp0]) 
    
#-------------------------------------------------------------------------
#2. Considerar o efeito de público
#------------------------------------------------------------------------- 

Ap  = 2*33.62           #Área efetiva de tabuleiro (m²)
pm2 = 0.3333            #Densidade de pessoas por m² - tráfego nível 1 (livre)
maj = np.sqrt(Ap*pm2)   #Fator de amplificação

Fa2 = maj*Fa1

     # Kf: Matriz de rigidez restringida
     # Mf : Matriz de massa restringida
     # Cf : Matriz de amortecimento
     # F : Vetor de força discretizado no tempo
     # t : lista do tempo discretizado
#------------------------------------------------------------------------- 
#1. Montar arrays Aceleração, Velocidade e Deslocamento
#-------------------------------------------------------------------------       
tf   = int(len(t))
F    = np.zeros([len(Kf),tf]) 
n    = len(F[:,0])
A    = np.zeros((n,tf)) #Aceleração (m/s²)
v    = np.zeros((n,tf)) #Velocidade (m/s)
U    = np.zeros((n,tf)) #Deslocamento (m)   

#------------------------------------------------------------------------- 
#2. Determinar as constantes do método de Newmark
#-------------------------------------------------------------------------
delta  = 0.5
alfa   = 0.25
a10    = 1/(alfa*(dt**2))
a11    = 1/(alfa*dt)
a12    = (1/(2*alfa))-1
a13    = delta/(dt*alfa)
a14    = (delta/alfa) - 1
a15    = (dt/2)*((delta/alfa) - 2)
C1     = np.linalg.inv(a10*Mf + a13*Cf + Kf)
A[:,0] = np.dot(np.linalg.inv(Mf),(F[:,0]-np.dot(Cf,v[:,0])-np.dot(Kf,U[:,0]))) #aceleração no tempo zero

#------------------------------------------------------------------------- 
#3. Resolver a equação de equilíbrio dinâmico
#-------------------------------------------------------------------------

for i in range(tf-1):
    
    F[:,i+1] = -1*Fp1[i+1]*Fa2[:,i+1]
    var1     = F[:,i+1]+np.dot(Mf,(a10*U[:,i]+ a11*v[:,i] + a12*A[:,i]))+np.dot(Cf,(a13*U[:,i]+ a14*v[:,i] 
                                                                                    + a15*A[:,i]))
    U[:,i+1] = np.dot(C1,var1)
    v[:,i+1] = a13*(U[:,i+1] - U[:,i]) - a14*v[:,i] - a15*A[:,i]
    A[:,i+1] = a10*(U[:,i+1] - U[:,i]) - a11*v[:,i] - a12*A[:,i] 
    

#As análises serão feitas para o nó mais próximo do centro do vão (Nó 10)

Umax = min(U[17,:])
Amax = max(A[17,:])
vmax = max(v[17,:])

# f40 = plt.figure(40, figsize=(15,10))
# plt.plot(t, U[17,:]*1000)
# plt.xlim(0,30);
# plt.ylim(-0.5,0.5);
# plt.legend(("Deslocamento máximo = {0:3.6f}m".format(Umax),"\n"))
# plt.grid(True)
# plt.xlabel('Tempo(s)')
# plt.ylabel('Deslocamento x 10³(m)')
# plt.title('Deslocamento')
# plt.rcParams['xtick.labelsize'] = 10
# plt.rcParams['ytick.labelsize'] = 10  

# f41 = plt.figure(41, figsize=(15,10))
# plt.plot(t, A[17,:])
# plt.xlim(0,30);
# plt.ylim(-1,1);
# plt.legend(("Aceleração máxima = {0:3.6f} m/s²".format(Amax),"\n"))
# plt.grid(True)
# plt.xlabel('Tempo(s)')
# plt.ylabel('Aceleração(m/s²)')
# plt.title('Aceleração')
# plt.rcParams['xtick.labelsize'] = 10
# plt.rcParams['ytick.labelsize'] = 10  

# f42 = plt.figure(42, figsize=(15,10))
# plt.plot(t, v[17,:])
# plt.xlim(0,30);
# plt.ylim(-0.006,0.006);
# plt.legend(("Velocidade máxima = {0:3.6f} m/s".format(vmax),'\n'))
# plt.grid(True)
# plt.xlabel('Tempo(s)')
# plt.ylabel('Velocidade(m/s)')
# plt.title('Velocidade')
# plt.rcParams['xtick.labelsize'] = 10
# plt.rcParams['ytick.labelsize'] = 10  






# desloc
plt.figure(31,figsize=(6,5),dpi=300);plt.xlim(0,30) ; plt.ylim(-0.4,0.1)      
plt.plot(t, U[17,:]*1000,c='black',linewidth=1, label='Central node')
plt.xlabel('t (s)',fontsize=14);plt.ylabel('Displacement (mm)',fontsize=16) 
xlista = list(U[17,:]*1000)
xmax = max(xlista)
tposmax = xlista.index(xmax)
tmax = t[tposmax]
xmin = min(xlista)
tposmin = xlista.index(xmin)
tmin   = t[tposmin]
plt.scatter(tmin,xmin, c='black')
plt.annotate('{0:3.1f}'.format(np.round(xmin,1)),fontsize=11, xy=(tmin, xmin), xytext=(tmin+1, xmin-0.005)) 
plt.legend(loc='upper right',fontsize=10, ncol=1, shadow=True, fancybox=True) 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)   
plt.grid(True)


# Veloc
plt.figure(32,figsize=(6,5),dpi=300);plt.xlim(0,30) ; plt.ylim(-4,5)      
plt.plot(t, v[17,:]*1000,c='black',linewidth=1, label='Central node')
plt.xlabel('t (s)',fontsize=14);plt.ylabel('Velocity (mm/s)',fontsize=16) 
xlista = list(v[17,:]*1000)
xmax = max(xlista)
tposmax = xlista.index(xmax)
tmax = t[tposmax]
xmin = min(xlista)
tposmin = xlista.index(xmin)
tmin   = t[tposmin]
plt.scatter(tmax,xmax, c='black')
plt.scatter(tmin,xmin, c='black')
plt.annotate('{0:3.1f}'.format(np.round(xmax,1)),fontsize=11, xy=(tmax, xmax), xytext=(tmax+0.5, xmax+0.05)) 
plt.annotate('{0:3.1f}'.format(np.round(xmin,1)),fontsize=11, xy=(tmin, xmin), xytext=(tmin+0.5, xmin-0.2)) 
plt.legend(loc='upper right',fontsize=10, ncol=1, shadow=True, fancybox=True) 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)   
plt.grid(True)


# Acc
plt.figure(33,figsize=(6,5),dpi=300);plt.xlim(0,30) ; plt.ylim(-400,600)      
plt.plot(t, A[17,:]*1000,c='black',linewidth=1, label='Central node')
plt.xlabel('t (s)',fontsize=14);plt.ylabel('Acceleration (mm/s²)',fontsize=16) 
xlista = list(A[17,:]*1000)
xmax = max(xlista)
tposmax = xlista.index(xmax)
tmax = t[tposmax]
xmin = min(xlista)
tposmin = xlista.index(xmin)
tmin   = t[tposmin]
plt.scatter(tmax,xmax, c='black')
plt.scatter(tmin,xmin, c='black')
plt.annotate('{0:3.1f}'.format(np.round(xmax,1)),fontsize=11, xy=(tmax, xmax), xytext=(tmax+0.5, xmax+0.05)) 
plt.annotate('{0:3.1f}'.format(np.round(xmin,1)),fontsize=11, xy=(tmin, xmin), xytext=(tmin+0.5, xmin-0.2)) 
plt.legend(loc='upper right',fontsize=10, ncol=1, shadow=True, fancybox=True) 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)   
plt.grid(True)




Umax = abs(U[17,:])
Amax = abs(A[17,:])
vmax = abs(v[17,:])
