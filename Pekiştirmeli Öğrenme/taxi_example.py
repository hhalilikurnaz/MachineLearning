import gymnasium as gym
import numpy as np
import random


env=gym.make("Taxi-v3",render_mode="ansi")
env.reset() #initial state getiriyoruz 

print(env.render())

"""
6 tane actionumuz söz konusu 
0:güney
1:kuzey
2:doğu
3:batı
4:yolcu almak 
5:yolcu indirmek


"""
action_space=env.action_space.n
print(action_space) #6 tane action spacemiz  var
state_space=env.observation_space.n
print(state_space) #kaç tane space oldugunu görüyoruz 

q_table=np.zeros((state_space,action_space)) #qtablenin satırları state sutunlarıa actionlardan oluşuyordu.
print(q_table)

alpha= 0.1 #learning rate 
gamma=0.6 #discount rate 
#yeni parametre epsion
epsilon=0.1
for i in range(1,100001):
    state,_=env.reset()
    done=False

    while not done :
        if random.uniform(0,1) <epsilon: #(explore) keşfetme paramtresi ekliyoruz 
            action=env.action_space.sample()
            #%10 luk bir olasılıklı çevreyi keşfetmesini sağlıyoruz 

            
        else:#exploit
            action=np.argmax(q_table[state]) #marximum değerin indexini al.q table git state bak yapmamız gereken action değerini bul ve uygula 
        env.step(action)

        next_state,reward,done,info,_=env.step(action)

        q_table[state,action]=q_table[state,action] + alpha *(reward+gamma*np.max(q_table[next_state])-q_table[state,action])

        state=next_state

print("Training Finished")


#Testing
total_epoch,total_penalties=0,0
episodes=100
for i in range(episodes):
    state,_=env.reset()
    epochs,penalties,reward=0,0,0
    done=False

    while not done :
        
            
        
        action=np.argmax(q_table[state]) #marximum değerin indexini al.q table git state bak yapmamız gereken action değerini bul ve uygula 
        

        next_state,reward,done,info,_=env.step(action)


        state=next_state

        if reward == -10:
            penalties +=1
        epochs +=1

    total_epoch +=epochs
    total_penalties += penalties
print("Result after {} episodes".format(episodes))
print("Avarage timestep per a episodes :",total_epoch/episodes)
print("Avarage penalties per episodes :",total_penalties/episodes)

