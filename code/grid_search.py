from cnn_ff_lab import net_train
from net_tester_lab import test_net
import math
import tensorflow as tf2
tf=tf2.compat.v1
tf.disable_v2_behavior()

grid_param = {
    'conv1_x': [2, 3, 4],
    #'conv1_y': [2, 3, 4],
    'conv1_z': [16, 32, 64],
    'conv1_nfilters': [1, 2, 3],
    'fc1': [26, 39, 52],
    'fc2': [26, 39, 52],
    'fc3': [13, 26, 39],
    'learn_rate': [0.0001],
    'n_iters': [250],
    'cost': [6,12,24,36,48,60,72],
}

grid_param_conv = {
    'conv1_x': [2, 3, 4],
    #'conv1_y': [2, 3, 4],
    'conv1_z': [16, 32, 64],
    'conv1_nfilters': [1, 2, 3],
    'max_pool':[6,12,24]
    }

grid_param_fc = {
    'fc1': [26, 39, 52],
    'fc2': [26, 39, 52],
    'fc3': [13, 26, 39],
}

grid_param_cost = {
    'cost': [6,12,24,36,48,60,72,90,120,150,180],
}

#gd_sr_cost = GridSearchCV(estimator=classifier,
#                     param_grid=grid_param_cost,
 #                    scoring='accuracy',
  #                   cv=5,
   #                  n_jobs=-1)

#gd_sr_conv = GridSearchCV(estimator=classifier,
 #                    param_grid=grid_param_conv,
  #                   scoring='accuracy',
   #                  cv=5,
    #                 n_jobs=-1)

def optimize_cost(iters,cost,step1,step2,direction):
    n_costs=iters
    i = 0
    tf.reset_default_graph()
    while(i<n_costs):
        f = open("scores\\scores_cost", "a")
        net_train([4, 32, 3], [42, 42, 14],cost,[1, 8], [2, math.floor((1280 - 32) / (13 * 8))], tf.orthogonal_initializer, 0.0005,"model_"+str(cost)+"_cost_demo",10, "channels_586202_ordered_batch_3608.mat", "labels_586202_batch_3608.mat")
        tf.reset_default_graph()
        [fpr, fnr, frar, nsrar, fsfpr] =test_net([4, 32, 3], [42, 42, 14],cost,[1, 8], [2, math.floor((1280 - 32) / (13 * 8))],tf.orthogonal_initializer, "model_"+str(cost)+"_cost_demo",11,20,"channels_586202_ordered_batch_3608.mat","labels_586202_batch_3608.mat")
        print("cost: ", cost)
        print("fpr: ", fpr)
        print("fnr: ", fnr)
        print("frar: ", frar)
        f.write("cost: " + str(cost) + "\tfpr: " + str(fpr) + "\tfnr: " + str(fnr) + "\tfrar" + str(frar) + "\tnsrar" + str(
            nsrar) + " \tfsfpr" + str(fsfpr) + "\n")
        f.close()

        #if score>=prev:
         #   direction = direction*-1
        #    step1=step1-step2
        #if step1<=0:
        #    break
        cost = cost + step1 * direction
        tf.reset_default_graph()
        f.close()
        #prev=score
        i=i+1

def layer_search():
    pass

def conv_search(cost):
    tf.reset_default_graph()
    a=grid_param_conv['conv1_x']
    f = open("scores\\scores_convxy", "a")
    for i in range(len(grid_param_conv['conv1_x'])):
        net_train([a[i], 32, 3], [42, 42, 14], cost, [1, 8], [6-a[i], math.floor((1280 - 32) / (13 * 8))],
                  tf.orthogonal_initializer, 0.0005, "model_" + str(cost) + "_convxy_demo", 10,
                  "channels_586202_ordered_batch_3608.mat", "labels_586202_batch_3608.mat")
        tf.reset_default_graph()
        [fpr, fnr, frar, nsrar, fsfpr] = test_net([a[i], 32, 3], [42, 42, 14], cost, [1, 8],
                                                  [6-a[i], math.floor((1280 - 32) / (13 * 8))], tf.orthogonal_initializer,
                                                  "model_" + str(cost) + "_convxy_demo", 11, 20,
                                                  "channels_586202_ordered_batch_3608.mat",
                                                  "labels_586202_batch_3608.mat")
        print("convxy: ", a[i])
        print("fpr: ", fpr)
        print("fnr: ", fnr)
        print("frar: ", frar)
        f.write(
            "convxy: " + str(a[i]) + "\tfpr: " + str(fpr) + "\tfnr: " + str(fnr) + "\tfrar" + str(frar) + "\tnsrar" + str(
                nsrar) + " \tfsfpr" + str(fsfpr) + "\n")
        tf.reset_default_graph()
    f.close()

def conv_search_z(iters,cost,cx,init,step1,step2,direction):
    tf.reset_default_graph()
    i=0
    f = open("scores_convz", "a")
    val=init
    prev=99999
    while i<iters:
        net_train([cx, val, 1], [39, 39, 13], cost, [1, 8], [6-cx, math.floor((1280-val)/(13*8))], "model_" + str(val) + "_convz")
        tf.reset_default_graph()
        [fpr, fnr, score] = test_net([cx, val, 1], [39, 39, 13], cost, [1, 8], [6-cx, math.floor((1280-val)/(13*8))], "model_" + str(val) + "_convz")
        print("conv: ", val)
        print("fpr: ", fpr)
        print("fnr: ", fnr)
        print("score: ", score)
        f.write("convz: " + str(val) + "\tscore: " + str(score) + "\n")
        if score>=prev:
            direction =direction*-1
            step1 = step1 - step2
        if step1 <= 0:
            break
        val = val + step1 * direction
        prev=score
        tf.reset_default_graph()
        i=i+1

def search_width(iters,cost,cx,val,init):
    i=0
    tf.reset_default_graph()
    neurons=init
    prev=9999
    while(i<iters):
        f = open("scores_neuron_width", "a")
        net_train([cx, val, 1], [neurons, neurons, math.floor(neurons/3)], cost, [1, 8], [6 - cx, math.floor((1280 - val) / (13 * 8))],
                  "model_" + str(neurons) + "_neuron_width")
        tf.reset_default_graph()
        [fpr, fnr, score] = test_net([cx, val, 1], [neurons, neurons, math.floor(neurons/3)], cost, [1, 8],
                                     [6 - cx, math.floor((1280 - val) / (13 * 8))], "model_" + str(neurons) + "_neuron_width")
        print("width: ", neurons)
        print("fpr: ", fpr)
        print("fnr: ", fnr)
        print("score: ", score)
        f.write("cost: " + str(neurons) + "\tscore: " + str(score) + "\tfpr: " +str(fpr) + "\tfnr: " + str(fnr) + "\n")
        #if score >= prev:
        #    direction = direction * -1
        #    step1 = step1 - step2
        #if step1 <= 0:
        #    break
        neurons = neurons + 80
        prev = score
        tf.reset_default_graph()
        f.close()
        i = i + 1

def search_initializers(init,stri):
    i=0
    tf.reset_default_graph()
    f = open("scores\\scores_inits", "a")
    net_train([4, 32, 3], [42, 42, 14], 50, [1, 8], [2, math.floor((1280 - 32) / (13 * 8))], init, 0.0005,
              "model_" + stri + "_inits", 10, "channels_586202_ordered_batch_3608.mat", "labels_586202_batch_3608.mat")
    tf.reset_default_graph()
    [fpr, fnr, frar, nsrar, fsfpr] = test_net([4,32,3],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],init,"model_"+stri+"_inits",11,20,"channels_586202_ordered_batch_3608.mat","labels_586202_batch_3608.mat")

    print("init: ", stri)
    print("fpr: ", fpr)
    print("fnr: ", fnr)
    print("frar: ", frar)
    f.write("init: " + stri + "\tfpr: " + str(fpr) + "\tfnr: " + str(fnr) + "\tfrar" + str(frar) + "\tnsrar" + str(nsrar) + " \tfsfpr" + str(fsfpr) + "\n")
    f.close()

def search_square(init,fs,it):
    tf.reset_default_graph()
    f = open("scores_convxy_2", "a")
    net_train([fs, 48, 1], [39, 39, 13], 50, [1, 8], [6-fs, 14], init, 0.0005, "model_" + str(fs) + "_convxy_2_"+str(it))

    tf.reset_default_graph()
    [fpr, fnr, score] = test_net([fs, 48, 1], [39, 39, 13], 50, [1, 8], [6-fs, 14], init, "model_" + str(fs) + "_convxy_2_"+str(it))

    print("convxy: ", str(fs))
    print("fpr: ", fpr)
    print("fnr: ", fnr)
    print("score: ", score)
    f.write("convxy: " + str(fs) + "\tscore: " + str(score) + "\tfpr: " + str(fpr) + "\tfnr: " + str(fnr) + "\n")
    f.close()

def search_z(init,fs,it):
    tf.reset_default_graph()
    f = open("scores_convz_2", "a")
    net_train([4, fs, 1], [39, 39, 13], 50, [1, 8], [2, math.floor((1280-fs)/(13*8))], init, 0.0005, "model_" + str(fs) + "_convz_2_"+str(it))

    tf.reset_default_graph()
    [fpr, fnr, score] = test_net([4, fs, 1], [39, 39, 13], 50, [1, 8], [2, math.floor((1280-fs)/(13*8))], init, "model_" + str(fs) + "_convz_2_"+str(it))

    print("convz: ", str(fs))
    print("fpr: ", fpr)
    print("fnr: ", fnr)
    print("score: ", score)
    f.write("convz: " + str(fs) + "\tscore: " + str(score) + "\tfpr: " + str(fpr) + "\tfnr: " + str(fnr) + "\n")
    f.close()

def search_w(init,fs,w,it):
    tf.reset_default_graph()
    f = open("scores_width_2", "a")
    net_train([4, fs, 1], [w, w, w/3], 50, [1, 8], [2, math.floor((1280-fs)/(13*8))], init, 0.0005, "model_" + str(w) + "_width_2_"+str(it))

    tf.reset_default_graph()
    [fpr, fnr, score] = test_net([4, fs, 1], [w, w, w/3], 50, [1, 8], [2, math.floor((1280-fs)/(13*8))], init, "model_" + str(w) + "_width_2_"+str(it))

    print("width: ", str(w))
    print("fpr: ", fpr)
    print("fnr: ", fnr)
    print("score: ", score)
    f.write("width: " + str(w) + "\tscore: " + str(score) + "\tfpr: " + str(fpr) + "\tfnr: " + str(fnr) + "\n")
    f.close()

def lr_search():
    lrs=[0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001,0.000005,0.000001,0.0000005]

    for i in range(len(lrs)):
        tf.reset_default_graph()
        f = open("scores_lr", "a")
        net_train([3, 48, 1], [39, 39, 13], 50, [1, 8], [3, 14], tf.keras.initializers.glorot_normal,lrs[i], "model_" + str(lrs[i]) + "_lr")

        tf.reset_default_graph()
        [fpr, fnr, score] = test_net([3, 48, 1], [39, 39, 13], 50, [1, 8], [3, 14], tf.keras.initializers.glorot_normal, "model_" + str(lrs[i]) + "_lr")

        print("lr: ", lrs[i])
        print("fpr: ", fpr)
        print("fnr: ", fnr)
        print("score: ", score)
        f.write("lr: " + str(lrs[i]) + "\tscore: " + str(score) + "\tfpr: " + str(fpr) + "\tfnr: " + str(fnr) + "\n")
        f.close()


#optimize_cost(8,25,5,2,1)
conv_search(60) #result:3
#conv_search_z(12,60,3,16,16,4,1)
#search_width(5,48,3,48,300)

#inits1=[tf.ones_initializer,tf.random_normal_initializer(mean=0.0,stddev=5),tf.random_uniform_initializer(minval=-100,maxval=100),tf.glorot_normal_initializer,
 #      tf.glorot_uniform_initializer,tf.keras.initializers.glorot_normal,tf.orthogonal_initializer,tf.truncated_normal_initializer,tf.uniform_unit_scaling_initializer,
  #     tf.variance_scaling_initializer]
#strs1=['ones','random_normal_stddev=5','random_uniform_(-100,100)','glorot_normal','glorot_uniform','xavier','orthogonal','truncated','uniform_unit_scaling','variance_scaling']

inits=[tf.glorot_uniform_initializer,tf.keras.initializers.glorot_normal,tf.orthogonal_initializer,tf.uniform_unit_scaling_initializer,
       tf.variance_scaling_initializer]
strs=['glorot_uniform','glort_normal','orthogonal','uniform_unit_scaling','variance_scaling']

#xy=[1,2,3,4,5]
z=[16,24,32,48,64,88]#32
w=[12,18,24,33,42,54,66,81,99]

#search_w(inits[2],32,81,3)

#for j in range(5):
#for i in range (len(w)):
   # search_w(inits[2],32,w[i],4)

#Search for initializers
#for i in range(len(inits)):
 #   search_initializers(inits[i],strs[i])



