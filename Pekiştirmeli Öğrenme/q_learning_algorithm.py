import gymnasium as gym
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt



environemnt=gym.make("FrozenLake-v1",is_slippery=False,render_mode="ansi") 
#render_mode görselleştirme yapmak için gerekli
#slippery ise ajanımızın enviroment içerisinde kayma olsaılığını kontrol eden parametre mesela buzlu ortamdaysa bir stateden başka bit state geçerken kaymazsa yanlıs stateye geçmez 
environemnt.reset()

nb_states=environemnt.observation_space.n
nb_actions=environemnt.action_space.n #yukarı aşağı sağ sol 

print(nb_states)
print(nb_actions)

qtable=np.zeros((nb_states,nb_actions))
print("Q Table :")
print(qtable)

#Parametreler

#bölüm sayısı 
#kaç kere oynuyacağını ifade ediyor.Environmenti kaç kere oynuyacağını tanımılıyoruz  
#Başaramazsa biter.Hedefe giderse bölüm biter
episodes=1000

#learning rate alpha 
alpha=0.5
#gama discount faktörümüz
gamma=0.9

#başarı(rewarda ulaşma durumu) ve başarısızlıkları burda depolucaz
outcomes=[]

#training
for _ in range(episodes):
    state,_=environemnt.reset()
    #initial state başlangıc state (start)
    done=False # başarıp başarmadığı daha başındayız başarısız
    outcomes.append("Failure")

    #ne kadar environment içinde dolaşıcak modelimiz ??
    #done == True olana kadar 

    while not done :#ajan başarılı olana kadar state içerisinde hareket et
        
        #actionlarımızı qtable göre seçicez

        if np.max(qtable[state])>0:
            action=np.argmax(qtable[state])
        else:
            action=environemnt.action_space.sample()

        #hareket ediyoruz
        new_state,reward,done,info,_=environemnt.step(action)

        #update q table 
        qtable[state,action]=qtable[state,action]+alpha*(reward+gamma*np.max(qtable[new_state]-qtable[state,action]))

        #update state
        state=new_state

        if reward:
            outcomes[-1]="Succes"
print("Qtable after training:")
print(qtable)

plt.bar(range(episodes),outcomes)
plt.show()


#Testing
episodes2=100
nb_succes=0

for _ in range(episodes2):
    state,_=environemnt.reset()
    done=False

    while not done :
        if np.max(qtable[state])>0:
            action=np.argmax(qtable[state])
        else:
            action=environemnt.action_space.sample()

            new_state,reward,done,info,_=environemnt.step(action)
            state=new_state

            nb_succes += reward

print("Succes Rate :",100*nb_succes/episodes2)
