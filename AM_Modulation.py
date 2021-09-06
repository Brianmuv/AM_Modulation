#Laboratorio 6  Modulación AM

##Librerias
"""

#Importe de Librerias
from scipy.fftpack import fft, fftfreq, fftshift
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from scipy import signal
pi=np.pi

"""## 1. 
Moduladora (Información)
Onda sinusoidal con frecuencia f=200hz.

"""

Am = 1 #Amplitud 
fm = 200 #Frecuencia 200 HZ


dt = (1/(1000)) # pasos de una milésima para el vector temporal
t = np.linspace(-3,3,6000)*(1/fm)

m_t = Am*np.sin(2*pi*fm*t) #señal moduladora

#Gráfico de la Señal
plt.figure(figsize=(12,6))
plt.plot(t,m_t)
plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Moduladora: 200 Hz",FontSize=15)
plt.grid()
plt.show()

"""##2.
Genere y grafique una portadora de al menos 10 veces la frecuencia de la
moduladora. La ventana temporal debe ser igual a la de la señal moduladora.
"""

Ac = 1 #Amplitud portadora
fc = 10*fm #Frecuencia portadora  -> 2 khz

c_t=np.sin(2*pi*fc*t) #señal portadora

#grafico de la señal
plt.figure(figsize=(12,6))
plt.plot(t,Ac*c_t)
plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Portadora 2 KHz",FontSize=15)
plt.grid()
plt.show()

"""##3.
Modulación Doble Banda Lateral, portadora suprimida
 
"""

x_am_ps=(m_t)*c_t  #señal Modulada

#grafico de moduladora
plt.figure(figsize=(20,10))
plt.subplot(2,2,1)
plt.plot(t,m_t)
plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Moduladora 200 hz",FontSize=15)
plt.grid()

#grafico de portadora
plt.subplot(2,2,2)
plt.plot(t,c_t)
plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Portadora" ,FontSize=15)
plt.grid()

#grafico modulada + Envolvente
plt.subplot(2,2,3)
plt.plot(t,x_am_ps,'C0') #Modulada
plt.plot(t,m_t,'C1')     #Envolvente => Moduladora
plt.plot(t,(-1*m_t),'C1') #Envolvente => -Moduladora
plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Modulada",FontSize=15)
plt.grid()


plt.subplots_adjust(hspace = 0.4)
plt.show()

"""**¿Qué puede observar? ¿La señal
modulada se parece a la señal de información?**

Anotaciones: 
- Se puede observar que la señal modulada contiene la moduladora y la portadora,
-

##4. 
Gran Portadora -- Indice de Modulación 100%
"""

im=1 #Indice de modulacion 100%
Ac=Am/im

x_am_gp = (Ac+m_t)*c_t  # señal Gran Portadora


#Grafico Moduladora
plt.figure(figsize=(20,10))
plt.subplot(2,2,1)
plt.plot(t,m_t)
plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Moduladora: ",FontSize=15)
plt.grid()
plt.tick_params(labelsize=15)

#Grafico Portadora
plt.subplot(2,2,2)
plt.plot(t,Ac*c_t)
plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Portadora ",FontSize=15)
plt.grid()
plt.tick_params(labelsize=15)

plt.subplot(2,2,3)
plt.plot(t,x_am_gp,'C0')
plt.plot(t,m_t+Ac,'C1')
plt.plot(t,(-1*m_t)-Ac,'C1')
plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Señal Am Gran portadora",FontSize=15)
plt.grid()
plt.tick_params(labelsize=15)

plt.subplot(2,2,4)
plt.plot(t,x_am_ps,'C0')
plt.plot(t,m_t,'C1')
#plt.plot(t,(-1*m_t),'C1')
plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Portadora $c(t)$ con amplitud $Ac$="+str(Ac),FontSize=15)
plt.grid()
plt.tick_params(labelsize=15)

plt.subplots_adjust(hspace = 0.37)

plt.show()

"""**¿Qué puede observar? ¿La señal modulada se parece a la señal
de información?**

Anotaciones:

-la envolvente es casi identica a la moduladora, con la novedad que cuenta tambien con un efecto espejo, es decir,m la envolvente representa la señal moduladora, y la versiíon negativa de la señal moduladora.

##5.
Gran Portadora con Indice de modulacion de  10% 150% en pasos de 10%)
"""

for i in range(1,15,1):
  im = i/10
  Ac = Am/im
  x_am_gp = (Ac+m_t)*c_t
  print("--------------------------------------------------------")
  print("\n\nIndice de modulacion %.4f(%2.f"%(im,im*100)+"%)\n")
  print("--------------------------------------------------------")

#moduladora
  plt.figure(figsize = (20,8))
  plt.subplot(2,2,1)
  plt.plot(t,m_t)
  plt.ylabel("Amplitud",FontSize=15)
  plt.title("Moduladora: ",Fontsize = 15)
  plt.grid()
  plt.tick_params(labelsize = 15)

 #Portadora
  plt.subplot(2,2,2)
  plt.plot(t,Ac*c_t)
  plt.xlabel("Tiempo",FontSize=15)
  plt.ylabel("Amplitud",FontSize=15)
  plt.title("Portadora:"+str(np.round(Ac,2))+"cos(w*t)$",Fontsize = 15)
  plt.grid()
  plt.tick_params(labelsize = 15)

#Modulada
  plt.subplot(2,2,3)
  plt.plot(t,x_am_gp,'C0') # modulada
  plt.plot(t,m_t+Ac,'C1')  #envolvente
  plt.plot(t,(-1*m_t)-Ac,'C1') #envolvente 
  plt.xlabel("Tiempo",FontSize=15)
  plt.ylabel("Amplitud",FontSize=15)
  plt.title("Señal Am Gran Portadora",Fontsize = 15)
  plt.grid()
  plt.tick_params(labelsize = 15)

  plt.subplots_adjust(hspace = 0.4)

  plt.show()

"""**¿Qué puede concluir sobre el
comportamiento de la señal modulada respecto al índice de modulación?**

Anotaciones: 
- Con la variación del índice de modulación para el caso de valores mayores al 100%, se ve afectada la aparición de nuevos picos y por ende, la pósible pérdida de información a lo largo de la transmisión de la misma.
- con indices de modulacion muy bajos, es decir, del orden del 10 % ,  no es evidente la modulacion de la amplitud, es decir, al diferencia entre las crestas y los valles es minima, por lo que al ser trasnmitida, yla saeñal se contamine con ruido, puede llegar a ser irrecuperable la informacion, o al menos, presentar mas errores

##6.
Use la transformada de Fourier para observar el comportamiento espectral de la
moduladora, la portadora y la señal modulada. Muestre los resultados obtenidos
usando subplots. El procedimiento debe hacerse para cada modulación (para
portadora suprimida y para gran portadora). En el caso de gran portadora use
nuevamente un índice de modulación del 100%.
"""

def espectral_graph(signal,fm,t,fp=0,limit=3):
  f = np.fft.fft(signal)
  f = np.fft.fftshift(f)
  freq = np.fft.fftfreq(len(signal),d = t[1]-t[0])
  freq = np.fft.fftshift(freq)
  plt.plot(freq,(1/len(signal)*abs(f)))
  plt.xlim((-fp-limit*fm),(fp + limit*fm))
  plt.grid()

im = 1
Ac = Am/im
x_am_gp = (Ac + m_t)*c_t
plt.figure(figsize = (15,18))

plt.subplot(4,1,1)
espectral_graph(m_t,fm,t)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Espectro Señal Moduladora",Fontsize = 15)
plt.grid()
plt.tick_params(labelsize = 15)

plt.subplot(4,1,2)
espectral_graph(Ac*c_t,fm,t,fc)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Espectro Señal Portadora",Fontsize = 15)
plt.grid()
plt.tick_params(labelsize = 15)

plt.subplot(4,1,3)
espectral_graph(x_am_ps,fm,t,fc)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Espectro Señal Moduladora Portadora Suprimida",Fontsize = 15)
plt.grid()
plt.tick_params(labelsize = 15)

plt.subplot(4,1,4)
espectral_graph(x_am_gp,fm,t,fc)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Espectro Señal Moduladora Gran Portadora",Fontsize = 15)
plt.grid()
plt.tick_params(labelsize = 15)

plt.subplots_adjust(hspace = 0.4)

"""##7.
Tren de pulsos-> Señal Moduladora
"""

m_t_square = signal.square(2*np.pi*fm*t, duty = 0.5)
x_am_ps_square = m_t_square*c_t
im = 1
Ac = Am/im
x_am_gp_square = (Ac + m_t_square)*c_t

plt.figure(figsize=(18,12))
plt.subplot(2,2,1)
plt.plot(t,m_t_square)
plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Moduladora: $m(t)$",FontSize=15)
plt.grid()
plt.tick_params(labelsize=15)

plt.subplot(2,2,2)
plt.plot(t,Ac*c_t)
#plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Portadora $c(t)$="+str(np.round(Ac,2))+"cos(2*\pi*fc*t)s",FontSize=15)
plt.grid()
plt.tick_params(labelsize=15)

plt.subplot(2,2,3)
plt.plot(t,x_am_ps_square,'C0')
plt.plot(t,m_t_square,'C1')
plt.plot(t,(-1*m_t_square),'C1')
plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Señal Am Portadora Suprimida --> $c(t)=cos(2*\pi*fc*t)$",FontSize=15)
plt.grid()
plt.tick_params(labelsize=15)

plt.subplot(2,2,4)
plt.plot(t,x_am_gp_square,'C0')
plt.plot(t,m_t_square + Ac,'C1')
plt.plot(t,(-1*m_t_square) - Ac,'C1')
plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Señal Am Gran Portadora",FontSize=15)
plt.grid()
plt.subplots_adjust(hspace = 0.4)
plt.tick_params(labelsize=15)

limite = 10
plt.figure(figsize=(15,18))
plt.subplot(4,1,1)
espectral_graph(m_t_square,fm,t,fc,limite)
#plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Espectro Señal Moduladora",FontSize=15)
plt.grid()
plt.tick_params(labelsize=15)

plt.subplot(4,1,2)
espectral_graph(Ac*c_t,fm,t,fc,limite)
#plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Espectro Señal Portadora",FontSize=15)
plt.grid()
plt.tick_params(labelsize=15)

plt.subplot(4,1,3)
espectral_graph(x_am_ps_square,fm,t,fc,limite)
#plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Espectro Señal Am Portadora Suprimida",FontSize=15)
plt.grid()
plt.tick_params(labelsize=15)

plt.subplot(4,1,4)  
espectral_graph(x_am_gp_square,fm,t,fc,limite)
plt.xlabel("Frecuencia [Hz]",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Espectro Señal Am Gran Portadora",FontSize=15)
plt.grid()
plt.tick_params(labelsize=15)
plt.subplots_adjust(hspace = 0.4)

plt.show()

"""##8.
Considerando como información (mensaje), la señal sinusoidal generada en el punto
1, realice la demodulación del formato portadora suprimida y del formato gran
portadora mediante detección coherente. Para ello, multiplique la señal modulada
por la portadora. Observe el resultado en el dominio de la frecuencia. Para filtrar la
componente de alta frecuencia que se ve en espectro, y que no es parte de la señal
de información, use un filtro pasa bajas. Analice las diferencias en fase y en amplitud
de la señal resultante.
"""

def demoduladorAm(modulacionAm,signal_portadora,fp,order=5):
  
  def butter_lowpass(cutoff,fs,order=5):
    nyq = 0.7*fs
    normal_cutoff = cutoff/nyq
    b , a = butter(order,normal_cutoff,btype='low',analog = False)
    return b , a
  def butter_lowpass_filter(data,cutoff,fs,order = 5):
    b,a = butter_lowpass(cutoff,fs,order=order)
    y = lfilter(b,a,data)
    return y

  fs = len(t)/max(t)
  y = butter_lowpass_filter(modulacionAm*signal_portadora,fp,fs,order)
  return y

sig_demodulada = demoduladorAm(x_am_ps,c_t,fc,5)
  
plt.plot(t,m_t,label="Señal de Informacion Tx")
plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)

plt.plot(t,sig_demodulada,label="Señal de Informacion Rx")
plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)

plt.title("Demodulacion Am Portadora Suprimida",FontSize = 15)
plt.grid()

plt.tick_params(labelsize=15)
plt.show()

sig_demodulada = demoduladorAm(x_am_gp,Ac*c_t,fc,5)
  
plt.plot(t,m_t,label="Señal de Informacion Tx")
plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)

plt.plot(t,sig_demodulada,label="Señal de Informacion Rx")
plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)

plt.title("Demodulacion Am Gran Portadora",FontSize = 15)
plt.grid()

plt.tick_params(labelsize=15)
plt.show()

"""##9.
Repita el procedimiento anterior (punto 8) cuando el mensaje es un tren de pulsos.
Analice los resultados obtenidos.
"""

sig_demodulada = demoduladorAm(x_am_ps_square,c_t,fc,5)
  
plt.plot(t,m_t_square,label="Señal de Informacion Tx")
plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)

plt.plot(t,sig_demodulada,label="Señal de Informacion Rx")
plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)

plt.title("Demodulacion Am Portadora Suprimida",FontSize = 15)
plt.grid()

plt.tick_params(labelsize=15)
#plt.legend(loc = 'Best')
plt.show()

def espectral_graph2(signal,fm,t,fp=0,factor=2):
  f = np.fft.fft(signal)
  f = np.fft.fftshift(f)
  freq = np.fft.fftfreq(len(signal),d = t[1]-t[0])
  freq = np.fft.fftshift(freq)
  plt.plot(freq,(1/len(signal)*abs(f)))
  plt.xlim((-factor*fp-fp/factor),(factor*fp+fp/factor))
  plt.grid()


im = 1
Ac = Am/im
x_am_gp = (Ac + m_t)*c_t


plt.figure(figsize=(15,8))
plt.subplot(4,1,1)
espectral_graph2(x_am_ps,fm,t,fc)
#plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Espectro Señal Am Portadora Suprimida",FontSize=15)
plt.grid()
plt.tick_params(labelsize=15)

plt.subplot(4,1,2)
espectral_graph2(x_am_ps*c_t,fm,t,fc)
#plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Espectro Señal Am Portadora Suprimida * $c(t)=cos(2*\pi*fc*t)$",FontSize=15)
plt.grid()
plt.tick_params(labelsize=15)

plt.subplot(4,1,3)
espectral_graph(x_am_gp,fm,t,fc)
#plt.xlabel("Tiempo",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Espectro Señal Am Gran Portadora",FontSize=15)
plt.grid()
plt.tick_params(labelsize=15)

plt.subplot(4,1,4)  
espectral_graph2(x_am_gp,fm,t,fc)
plt.xlabel("Frecuencia [Hz]",FontSize=15)
plt.ylabel("Amplitud",FontSize=15)
plt.title("Espectro Señal Am Gran Portadora * $c(t)="+str(np.round(Ac,2))+"cos(2/pi*fc*t)$" ,FontSize=15)
plt.grid()
plt.tick_params(labelsize=15)
plt.subplots_adjust(hspace = 0.5)

plt.show()
