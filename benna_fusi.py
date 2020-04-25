# Implementation of the Benna Fusi model for DQN
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from dqn import DQN

# This class stores the values of the visible and hidden variables in the Benna Fusi model
class BennaFusi:
    def __init__(self, weights, depth=3, base_flow=1, base_c=1, flow_decay=2, slow_update=False, leak=True, beaker_init='noisy', mod_inflow=False):
        # Initialise hidden beakers
        # number of beakers
        self.depth = depth
        self.base_flow = base_flow
        self.flow_decay = flow_decay
        # list of tube widths
        self.flows = base_flow * 1.0 / np.power(self.flow_decay, range(depth))
        self.vars = np.empty((depth, len(weights)), dtype=object)
        self.dvars = np.empty((depth, len(weights)), dtype=object)
        # list of beaker areas
        self.cs = base_c * np.power(self.flow_decay, range(depth))
        # operations for applying flow to each beaker
        self.flow_ops = np.empty((depth, len(weights)), dtype=object)
        self.weights_to_beaker_ops = [] 
        self.beaker_to_weights_ops = []
        # determines whether to apply flow after each update or after each minibatch
        self.slow_update = slow_update
        if leak:
            self.leak = tf.constant(1.0)
        else:
            self.leak = tf.constant(0.0)

        self.beaker_init = beaker_init
        self.mod_inflow = mod_inflow
        
        # Define beaker variables
        for i in range(self.depth):
            for j in range(len(weights)):
                if self.beaker_init == 'random':
                    self.vars[i, j] = tf.Variable(tf.random_normal(weights[j].get_shape().as_list(), stddev=1.0/np.sqrt(weights[j].get_shape().as_list()[0])),name="beaker_"+str(i)+"_"+str(j))
                elif self.beaker_init == 'equilibrium':
                    self.vars[i, j] = tf.Variable(weights[j].initialized_value(),name="beaker_"+str(i)+"_"+str(j),dtype=tf.float32)
                elif self.beaker_init == 'zeros':
                    if i == 0:
                        self.vars[i,j]=tf.Variable(weights[j].initialized_value(),name="beaker_"+str(i)+"_"+str(j),dtype=tf.float32)
                    else:
                        self.vars[i,j]=tf.Variable(tf.zeros(weights[j].get_shape().as_list()),name="beaker_"+str(i)+"_"+str(j),dtype=tf.float32)
                elif self.beaker_init=='eq_distribution':
                    if i==0:
                        self.vars[i,j]=tf.Variable(weights[j].initialized_value(),name="beaker_"+str(i)+"_"+str(j),dtype=tf.float32)
                    else:
                        self.vars[i,j]=tf.Variable(self.vars[i-1,j].initialized_value()+tf.random_normal(weights[j].get_shape().as_list(),stddev=np.power(0.5,i)*1.0/np.sqrt(weights[j].get_shape().as_list()[0])),name="beaker_"+str(i)+"_"+str(j))
                elif self.beaker_init=='scaled_random':
                    if i==0:
                        self.vars[i,j]=tf.Variable(weights[j].initialized_value(),name="beaker_"+str(i)+"_"+str(j),dtype=tf.float32)
                    else:
                        self.vars[i,j]=tf.Variable(tf.random_normal(weights[j].get_shape().as_list(),stddev=(1.0-(i+1)/self.depth)/np.sqrt(weights[j].get_shape().as_list()[0])),name="beaker_"+str(i)+"_"+str(j))
                elif self.beaker_init == 'linear_var':
                    if i == 0:
                        self.vars[i, j] = tf.Variable(weights[j].initialized_value(), name="beaker_"+str(i)+"_"+str(j), dtype=tf.float32)
                    else:
                        self.vars[i, j] = tf.Variable(tf.random_normal(weights[j].get_shape().as_list(), stddev=np.sqrt(1.0 - (i + 1) / self.depth) / np.sqrt(weights[j].get_shape().as_list()[0])), name="beaker_"+str(i)+"_"+str(j))
                elif self.beaker_init=='noisy':
                    self.vars[i, j]=tf.Variable(weights[j].initialized_value()+tf.random_normal(weights[j].get_shape().as_list(),stddev=0.1/np.sqrt(weights[j].get_shape().as_list()[0])),name="beaker_"+str(i)+"_"+str(j))
                    
        # Define operations for transferring values from first beaker to actual weights and vice versa
        for j in range(len(weights)):
            self.weights_to_beaker_ops.append(self.vars[0][j].assign(weights[j]))
            self.beaker_to_weights_ops.append(weights[j].assign(self.vars[0][j]))

        self.weights_to_beaker_op_group = tf.group(*(self.weights_to_beaker_ops))
        self.beaker_to_weights_op_group = tf.group(*(self.beaker_to_weights_ops))

        # Define variable that determines whether there is flow into the first beaker
        # This can be used for stabilising the beaker values before allowing the metaplasticity to kick in
        self.flow_switches = tf.placeholder(shape=(self.depth,), dtype=tf.float32)

        # This placeholder is used to modulate flow between beakers
        self.flow_mods = [tf.placeholder(shape=w.get_shape(), dtype=tf.float32) for w in weights]
        self.default_flow_mods = [tf.ones(shape=w.get_shape(), dtype=tf.float32) for w in weights]
        
        # Define dvars which calculates the flow between beakers
        if self.mod_inflow:
            for i in range(self.depth):
                for j in range(len(weights)):
                    if i == 0:
                        self.dvars[i,j]=self.flow_switches[i]*(1.0/self.cs[i])*self.flows[i]*(self.vars[i+1][j]-self.vars[i][j])
                    elif i == (self.depth-1):
                        self.dvars[i,j]=(1.0/self.cs[i])*(tf.minimum(self.flow_mods[j]*self.flows[i-1],1.0)*(self.vars[i-1][j]-self.vars[i][j])
                                                          -self.flow_switches[i]*self.leak*self.flows[i]*self.vars[i][j])
                    else:
                        self.dvars[i,j]=(1.0/self.cs[i])*(tf.minimum(self.flow_mods[j]*self.flows[i-1],1.0)*(self.vars[i-1][j]-self.vars[i][j])
                                                          +self.flow_switches[i]*self.flows[i]*(self.vars[i+1][j]-self.vars[i][j]))
                    self.flow_ops[i,j]=self.vars[i][j].assign_add(self.dvars[i][j])
        else:
            for i in range(self.depth):
                for j in range(len(weights)):
                    if i == 0:
                        self.dvars[i, j] = self.flow_switches[i] * (1.0/self.cs[i])*tf.minimum(self.flows[i]*self.flow_mods[j],1.0)*(self.vars[i+1][j]-self.vars[i][j])
                    elif i == (self.depth-1):
                        self.dvars[i,j] = (1.0/self.cs[i])*(tf.minimum(self.flow_mods[j]*self.flows[i-1],1.0)*(self.vars[i-1][j]-self.vars[i][j])
                                                          -self.flow_switches[i]*self.leak*tf.minimum(self.flow_mods[j]*self.flows[i],1.0)*self.vars[i][j])
                    else:
                        self.dvars[i,j]=(1.0/self.cs[i])*(tf.minimum(self.flow_mods[j]*self.flows[i-1],1.0)*(self.vars[i-1][j]-self.vars[i][j])
                                                          +self.flow_switches[i]*tf.minimum(self.flow_mods[j]*self.flows[i],1.0)*(self.vars[i+1][j]-self.vars[i][j]))
                    self.flow_ops[i,j]=self.vars[i][j].assign_add(self.dvars[i][j])
                
        self.flat_dvars = np.array(self.dvars).flatten().tolist()
        
        # Define flow operations
        self.flow_ops = np.array(self.flow_ops).flatten()
        self.flow_op_group = tf.group(*(self.flow_ops))

        # variables for recording ratio of flow to total update for first beaker
        self.beaker_data = []
    
    def apply_flow(self, sess=tf.get_default_session(), flow_switches=None, flow_mods=None):

        if flow_mods is None:
            flow_mods = sess.run(self.default_flow_mods)

        if flow_switches is None:
            flow_switches = np.ones(self.depth)
            
        feed_dict = {i:j for i,j in zip(self.flow_mods, flow_mods)}
        feed_dict[self.flow_switches]=flow_switches
    
        # First copy weights to first beaker
        sess.run(self.weights_to_beaker_op_group)
        # Then calculate flow (dvars)
        sess.run(self.flat_dvars, feed_dict=feed_dict)
    
        # Then implement flow
        sess.run(self.flow_op_group, feed_dict=feed_dict)
        # Then copy back from first beaker to weights
        sess.run(self.beaker_to_weights_op_group)

if __name__=="__main__":

    num_inputs = 4
    num_actions = 4

    agent = DQN(num_inputs, num_actions)


    opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    vars = tf.trainable_variables()

    print vars

    bf = BennaFusi(vars, base_flow=0.1)

    #print bf.vars

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        grads_and_vars = opt.compute_gradients(agent.loss, tf.trainable_variables())
#        tf.Print(bf.vars[0][0],[bf.vars[0][0]])
#        tf.Print(bf.vars[1][0],[bf.vars[1][0]])
#        tf.Print(bf.vars[2][0],[bf.vars[2][0]])
        for i in range(5):
     #       print i
#            print sess.run(bf.flat_dvars)
            #print sess.run(bf.flat_dvars)
            var1,var2,var3 = sess.run([bf.vars[0][0],bf.vars[1][0],bf.vars[2][0]])
      #      print var1[0], var2[0], var3[0]
            #print sess.run(vars[0])
            #sess.run(bf.weights_to_beaker_op_group)
            bf.apply_flow(sess=sess)
        
       # print bf.beaker_data

    plt.figure()
    plt.plot(bf.beaker_data)
    plt.show()
