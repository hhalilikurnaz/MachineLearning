import gymnasium as gym

import random
import numpy as np

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

#Nasl hareket edicez ?
action=environemnt.action_space.sample()
"""
sol:0
asagı:1
sag:2
yukarı:3
"""
print(action)
t=environemnt.step(action)

print(t)
#return ettiği şeyler
"""
sırasıyla;
observation
reward(float)
terminated,done 
info(dictionary)
probability

"""
#biz bunları farklı değişkenlere atıcaz

new_state,reward,done,info,_=environemnt.step(action)
print(new_state)
