# -*- coding: utf-8 -*-
"""HMAX

!pip install PymoNNto
from PymoNNto import *
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist
from copy import deepcopy

(train_X, train_y), (test_X, test_y) = mnist.load_data()

def update_W(delta_t, alpha=1, betta=0, fr=1, fp=0.0, ar_p=0.004, ar_n=-0.003, ap_p=0.0, ap_n=0.0):
  if delta_t == 1 :
    return alpha*fr*ar_p + betta*fp*ap_n
  elif delta_t == -1 :
    return alpha*fr*ar_n + betta*fp*ap_p

def STDP(pre_layer, post_layer, feature):   
    #pre layer  -> (3,h_j,w_j) 
    #post layer -> (3,h_i,w_i)
    #(spike pattern, #channels, time step)
    
    h_i= post_layer[0].shape[0]
    w_i= post_layer[0].shape[1]
    
    for r_i in range(h_i):
        for c_i in range(w_i):
            if post_layer[0][r_i,c_i]==False:
                W= feature[int(post_layer[1][r_i,c_i])]
                t_winner= post_layer[2][r_i,c_i]
                h_j= W.shape[0]
                w_j= W.shape[1]
                
                for r_j in range(h_j):
                    for c_j in range(w_j):
                        if pre_layer[0][r_i+r_j,c_i+c_j]==True:
                            W[r_j,c_j]+= update_W(-1)*W[r_j,c_j]*(1-W[r_j,c_j])
                        else:
                            if pre_layer[2][r_i+r_j,c_i+c_j]>t_winner:
                                W[r_j,c_j]+= update_W(-1)*W[r_j,c_j]*(1-W[r_j,c_j])
                            else:
                                W[r_j,c_j]+= update_W(1)*W[r_j,c_j]*(1-W[r_j,c_j])
                
                W = W - np.min(W)
                W = W/np.max(W)
                feature[int(post_layer[1][r_i,c_i])]= W
    return feature

def R_STDP(pre_layer, post_layer, W, id):
    #pre layer -> [T/F, feature, time_step]
    #post layer -> [T/F, 0, time_step]
    #W -> (post_layer, pre_layer)
    
    l_i= post_layer[0].shape[0] 
    l_j= pre_layer[0].shape[0]
    min_t = np.min(post_layer[2])
    for n in range(l_i):
        if post_layer[0][n]==False:
            if n==id and post_layer[2][n]==min_t:
                A=1
                B=0
            else:
                A=0
                B=1
            t_winner= post_layer[2][n]
            
            for n_j in range(l_j):  
                for c in range(len(W)):
                    if pre_layer[0][n_j]==False and int(pre_layer[1][n_j])==c and pre_layer[2][n_j]<=t_winner:

               #if pre_layer[0][n_j]==False and pre_layer[2][n_j]<=t_winner:  
                        W[c][n,n_j]+= update_W(1, A, B, 0.5, 0.5, 0.004, -0.003, 0.0005, -0.004)
                    else:
                        W[c][n,n_j]+= update_W(-1, A, B, 0.5, 0.5, 0.004, -0.003, 0.0005, -0.004)
    W = W - np.min(W)
    W = W/np.max(W)
    return W

class LIF_main(Behaviour):

    #dic_m[layer] = [spike pattern, feature, time step]
    def set_variables(self, n):
        global dic_m
        self.set_init_attrs_as_variables(n)
        n.v = n.v_rest
        n.fired = n.get_neuron_vec() > 0
        n.dt = 0.01
        if n.tags[1] not in dic_m:
          dic_m[n.tags[1]] = [n.get_neuron_vec() >= 0, n.get_neuron_vec()*0-1, n.get_neuron_vec()*0+float('inf')]
        
    def new_iteration(self, n):
        n.v +=  n.I    #*n.R*n.dt/n.tau_m
        n.fired = n.v > n.v_threshold    
        if np.sum(n.fired) > 0:
            n.v[n.fired] = n.v_reset
            n.fired = n.fired*dic_m[n.tags[1]][0]
            dic_m[n.tags[1]][0][n.fired] = False
            dic_m[n.tags[1]][1][n.fired] = n.tags[2]
            dic_m[n.tags[1]][2][n.fired] = n.iteration
            
class LIF_convolution(Behaviour):
    
    def set_variables(self, n):
          for s in n.afferent_synapses['All']:
              #n.l, n.w, n.filter
              s.W = s.get_synapse_mat()
              fw = n.filter.shape[0]
              fl = n.filter.shape[1] 
              for i in range(s.W.shape[0]):
                  D = s.W[i].reshape([n.w,n.l])
                  D[i//(n.l-fl+1):i//(n.l-fl+1)+fw, i%(n.l-fl+1):i%(n.l-fl+1)+fl] = n.filter
                  s.W[i] = D.flatten()

          n.I = n.get_neuron_vec()
        
    def new_iteration(self, n):
        n.I = n.get_neuron_vec()
        for s in n.afferent_synapses['All']:
            n.I += np.sum(s.W[:, s.src.fired], axis=1)     #/((n.filter.shape[0]*n.filter.shape[1])*0.2)
        if n.iteration == n.stop :
          for s in n.afferent_synapses['All']:
            del s.W

class LIF_pooling(Behaviour):

    def set_variables(self, n):
          for s in n.afferent_synapses['All']:
              #n.l, n.w
              s.W = s.get_synapse_mat()
              n.filter = np.ones(shape=[2,2])

              for i in range(s.W.shape[0]):
                    D = s.W[i].reshape([n.w,n.l])
                    lp = (n.l//2)
                    D[2*(i//lp):2*(i//lp+1), 2*(i%lp):2*(i%lp+1)] = n.filter
                    s.W[i] = D.flatten()

          n.I = n.get_neuron_vec()
        
    def new_iteration(self, n):
        n.I = n.get_neuron_vec()
        for s in n.afferent_synapses['All']:
                n.I += np.sum(s.W[:, s.src.fired], axis=1)
        
        if n.iteration == n.stop :
          for s in n.afferent_synapses['All']:
            del s.W

class LIF_input(Behaviour):

    def set_variables(self, n):
          n.I = n.get_neuron_vec()
        
    def new_iteration(self, n):
        n.I = n.get_neuron_vec()
        n.I[n.spike_pattern[-1:]] = 16
        del n.spike_pattern[-1:]

class LIF_output(Behaviour):

    def set_variables(self, n):
          itr = 0
          for s in n.afferent_synapses['All']:
            s.W = W[itr]
            itr+=1
          n.I = n.get_neuron_vec()
        
    def new_iteration(self, n):
        n.I = n.get_neuron_vec()
        for s in n.afferent_synapses['All']:
                n.I += np.sum(s.W[:, s.src.fired], axis=1)
        if n.iteration == n.stop :
          for s in n.afferent_synapses['All']:
            del s.W

class LIF_main(Behaviour):

    #dic_m[layer] = [spike pattern, feature, time step]
    def set_variables(self, n):
        global dic_m
        self.set_init_attrs_as_variables(n)
        n.v = n.v_rest
        n.fired = n.get_neuron_vec() > 0
        n.dt = 0.01
        if n.tags[1] not in dic_m:
          dic_m[n.tags[1]] = [n.get_neuron_vec() >= 0, n.get_neuron_vec()*0-1, n.get_neuron_vec()*0+float('inf')]
        
    def new_iteration(self, n):
        n.v +=  n.I    #*n.R*n.dt/n.tau_m
        n.fired = n.v > n.v_threshold    
        if np.sum(n.fired) > 0:
            n.v[n.fired] = n.v_reset
            n.fired = n.fired*dic_m[n.tags[1]][0]
            dic_m[n.tags[1]][0][n.fired] = False
            dic_m[n.tags[1]][1][n.fired] = n.tags[2]
            dic_m[n.tags[1]][2][n.fired] = n.iteration
            
class LIF_convolution(Behaviour):
    
    def set_variables(self, n):
          for s in n.afferent_synapses['All']:
              #n.l, n.w, n.filter
              s.W = s.get_synapse_mat()
              fw = n.filter.shape[0]
              fl = n.filter.shape[1] 
              for i in range(s.W.shape[0]):
                  D = s.W[i].reshape([n.w,n.l])
                  D[i//(n.l-fl+1):i//(n.l-fl+1)+fw, i%(n.l-fl+1):i%(n.l-fl+1)+fl] = n.filter
                  s.W[i] = D.flatten()
          n.I = n.get_neuron_vec()
        
    def new_iteration(self, n):
        n.I = n.get_neuron_vec()
        print(n.iteration)
        for s in n.afferent_synapses['All']:
            n.I += np.sum(s.W[:, s.src.fired], axis=1)    #/((n.filter.shape[0]*n.filter.shape[1])*0.2)
        if n.iteration == n.stop :
          for s in n.afferent_synapses['All']:
            del s.W

class LIF_pooling(Behaviour):

    def set_variables(self, n):
          for s in n.afferent_synapses['All']:
              #n.l, n.w
              s.W = s.get_synapse_mat()
              n.filter = np.ones(shape=[2,2])

              for i in range(s.W.shape[0]):
                    D = s.W[i].reshape([n.w,n.l])
                    lp = (n.l//2)
                    D[2*(i//lp):2*(i//lp+1), 2*(i%lp):2*(i%lp+1)] = n.filter
                    s.W[i] = D.flatten()

          n.I = n.get_neuron_vec()
        
    def new_iteration(self, n):
        n.I = n.get_neuron_vec()
        for s in n.afferent_synapses['All']:
                n.I += np.sum(s.W[:, s.src.fired], axis=1)
        
        if n.iteration == n.stop :
          for s in n.afferent_synapses['All']:
            del s.W

class LIF_input(Behaviour):

    def set_variables(self, n):
          n.I = n.get_neuron_vec()
        
    def new_iteration(self, n):
        n.I = n.get_neuron_vec()
        n.I[n.spike_pattern[-2:]] = 16
        del n.spike_pattern[-2:]

class LIF_output(Behaviour):

    def set_variables(self, n):
          itr = 0
          for s in n.afferent_synapses['All']:
            s.W = W[itr]
            itr+=1
          n.I = n.get_neuron_vec()
        
    def new_iteration(self, n):
        n.I = n.get_neuron_vec()
        for s in n.afferent_synapses['All']:
                n.I += np.sum(s.W[:, s.src.fired], axis=1)
        if n.iteration == n.stop :
          for s in n.afferent_synapses['All']:
            del s.W

model = {}
model['layer1'] = ['input',1]
model['layer2'] = ['conv',8,(11,11)]
model['output'] = ['output', 1]

W =[]
untrainable = []
trainable =['layer2']
for layer in model:
  if model[layer][0] == 'conv':
    if layer not in untrainable:
      for _ in range(model[layer][1]):
          model[layer].append(np.random.uniform(0,1,model[layer][2]))

def Simulate(Input, model ,num_steps = 5):
    global dic_m
    global W
    
    spike_pattern = np.argsort(Input.reshape([-1]))
    spike_pattern = spike_pattern.tolist()
    dic_m = {}
    My_Network = Network()
    
    w=Input.shape[0]
    l=Input.shape[1]
    prelayer =[]
    Object = {}
    count = 0

    for layer in model:
        count+=1
        Object[layer] = [None for i in range(model[layer][1])]

        if count == 1:
              for obj in range(model[layer][1]):
                  Object[layer][obj] = NeuronGroup(net=My_Network, tag=str(count)+'a,'+layer+','+str(obj), size=get_squared_dim(w*l), behaviour={
                          1: LIF_main(v_rest=0, v_reset=0, v_threshold=1 , tau_m=1 , R=1, spike_pattern = spike_pattern, stop=num_steps),
                          2: LIF_input(),
                          9: Recorder(tag='my_recorder', variables=['n.fired'])})
            
              prelayer= layer
                  
        elif model[layer][0] == 'conv':
              w1= w- model[layer][2][0]+ 1
              l1= l- model[layer][2][1]+ 1
              for obj in range(model[layer][1]):
                  filter = model[layer][3+obj]

                  Object[layer][obj] = NeuronGroup(net=My_Network, tag=str(count)+'b,'+layer+','+str(obj), size=get_squared_dim(w1*l1), behaviour={
                          1: LIF_main(v_rest=0, v_reset=0, v_threshold=filter.shape[0]*filter.shape[1]*0.3 , tau_m=10 , R=10, l=l, w=w, filter=filter, stop=num_steps),
                          2: LIF_convolution(),
                          9: Recorder(tag='my_recorder', variables=['n.fired' , 'n.v'])})

                  for preobj in range(model[prelayer][1]):
                      SynapseGroup(net=My_Network, src=Object[prelayer][preobj], dst=Object[layer][obj], tag='GLUTAMATE')
              w= w1
              l= l1
              prelayer= layer
        
        elif model[layer][0] == 'pooling':
            w1 = w//2
            l1= l//2
            for obj in range(model[layer][1]):
                Object[layer][obj] = NeuronGroup(net=My_Network, tag=str(count)+'c,'+layer+','+str(obj), size=get_squared_dim(w1*l1), behaviour={
                  1: LIF_main(v_rest=0, v_reset=0, v_threshold=1 , tau_m=10 , R=10, l=l, w=w, stop=num_steps),
                  2: LIF_pooling(),
                  9: Recorder(tag='my_recorder', variables=['n.fired'])})
                
                SynapseGroup(net=My_Network, src=Object[prelayer][obj], dst=Object[layer][obj], tag='GLUTAMATE')
                
            w= w1
            l= l1
            prelayer= layer
            
        
        elif model[layer][0]=='output':
            if W == [] :
              for n in range(model[prelayer][1]):
                  W.append(np.random.normal(0.8,0.02,(10, w*l)))
            
            Object['output'][0] = NeuronGroup(net=My_Network, tag=str(count)+'c,output'+','+str(0), size=get_squared_dim(10), behaviour={
                          1: LIF_main(v_rest=0, v_reset=0, v_threshold=w*l*0.4 , tau_m=10 , R=10, stop=num_steps),
                          2: LIF_output(),
                          9: Recorder(tag='my_recorder', variables=['n.fired','n.v'])})
            
            for obj in range(model[prelayer][1]):
                SynapseGroup(net=My_Network, src=Object[prelayer][obj], dst=Object['output'][0], tag='GLUTAMATE')

    My_Network.initialize()
    My_Network.simulate_iterations(num_steps, measure_block_time=True)
    return My_Network

def cal_C(feacture):
  Ci = 0
  for W in feacture:
    W = W.flatten()
    Ci+=np.sum((W)*(1-W))
  nw = len(feacture)*W.shape[0]
  return Ci/nw

prelayer= []

Ci={}
for layer in trainable:
    Ci[layer]=[cal_C(model[layer][3:])]

for layer in untrainable:
    Ci[layer]=[cal_C(model[layer][3:])]

i=0
while trainable != [] and i<3000:
    i+=1
    X = train_X[i]
    my_Network = Simulate(X, model ,num_steps = 50)

    layer = trainable[0]
    prelayer = layer[:5] + str(int(layer[5:])-1)

    Shape = dic_m[prelayer][0].shape[0]
    pre_layer= np.array(dic_m[prelayer]).reshape(3,int(Shape**0.5),int(Shape**0.5))

    Shape = dic_m[layer][0].shape[0]
    post_layer= np.array(dic_m[layer]).reshape(3,int(Shape**0.5),int(Shape**0.5))

    model[layer][3:] = STDP(pre_layer, post_layer, model[layer][3:])

    for l in trainable:
        Ci[l].append(cal_C(model[l][3:]))
    for l in untrainable:
        Ci[l].append(cal_C(model[l][3:]))

    if cal_C(model[layer][3:])<0.01:
      untrainable.append(trainable[0])
      del trainable[0]

for i in range(5000):
    X = train_X[i]
    my_Network = Simulate(X, model ,num_steps = 50)

    layer = 'output'
    prelayer= 'layer2'

    Shape = dic_m[prelayer][0].shape[0]
    pre_layer= np.array(dic_m[prelayer]).reshape(3,int(Shape**0.5),int(Shape**0.5))

    Shape = dic_m[layer][0].shape[0]
    post_layer= np.array(dic_m[layer]).reshape(3,int(Shape**0.5),int(Shape**0.5))

    W = R_STDP(pre_layer, post_layer, W, train_y[i])