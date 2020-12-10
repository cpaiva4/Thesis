from utils import *
import hdf5storage
import matplotlib.pyplot as plt
import math
from matplotlib import collections as matcoll
import statistics as stat
from matplotlib import colors as mcolors
tf=tf2.compat.v1
tf.disable_v2_behavior()


def conv_net_test(x, weights, biases, batch_size,strides,maxpool):
    conv1 = convimp(x, weights['wc1'], biases['bc1'],strides[0],strides[1],"VALID")
    conv1 = maxpoolimp(conv1,maxpool[0],maxpool[1],"VALID")
    #conv2 = convimp(conv1, weights['wc2'], biases['bc2'], 1,4, "VALID")
    #conv2 = maxpoolimp(conv2, 2, 2, "VALID")
    s = tf.shape(conv1)
    fc1 = tf.reshape(conv1, [batch_size, s[1] * s[2] * s[3] * s[4]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc2=tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3'])
    fc3 = tf.nn.relu(fc3)
    out = tf.add(tf.matmul(fc3, weights['out']), biases['out'])
    return out

def retrieve_batch(channels,batch_size,lab):
    maps=np.zeros((batch_size,5,5,1280))
    #print(channels.shape)
    #print(lab.shape)
    for i in range(len(lab)):

        for j in range(25):
            maps[i,j//5,j%5,:]=channels[j,(i)*256:(i)*256+1280] #256=1 second step
    maps=np.reshape(maps,[batch_size,5,5,1280,1])
    return maps

def build_scatter(alerts,ax,color):
    x = alerts
    y = [0.5 for i in range(len(x))]

    lines = []
    for i in range(len(x)):
        pair = [(x[i], 0), (x[i], y[i])]
        lines.append(pair)

    #colors = [mcolors.to_rgba('r')]

    linecoll = matcoll.LineCollection(lines,colors=color,linewidths=[1.5 for i in range(len(x))])
    ax.add_collection(linecoll)
    #plt.scatter(x, y)

def test_net(convw,ffw,relcost,strides,maxpool,init,savepath,n_seizures,ch_file,label_file):
    labels = hdf5storage.loadmat(label_file)#labels_114902_train.mat')#labels_586202_batch_3608.mat')

    ch = hdf5storage.loadmat(ch_file)#channels_114902_train.mat')#channels_586202_ordered_batch_3608.mat')

   # labels = hdf5storage.loadmat('D:\\docs\\tese\\pacientes\\mat_data\\feature_files\\labels_114902_train.mat')#labels_test_set_58602.mat')
    #inp = hdf5storage.loadmat('D:\\docs\\tese\\pacientes\\mat_data\\feature_files\\inputs_95202_1_hora_80_overlap.mat')
   # ch=hdf5storage.loadmat('D:\\docs\\tese\\pacientes\\mat_data\\feature_files\\channels_114902_train.mat')#channels_test_set_58602.mat')

    l=labels['labels']
    #inp=inp['inputs']#[0]
    ch=ch['ch']


    training_iters = 200
    learning_rate = 0.0005
    batch_size = 3600

    n_input = 5 * 5 * 5 * 256

    n_classes = 2

    x = tf.placeholder("float", [batch_size, 5, 5, 1280, 1])
    y = tf.placeholder("float", [batch_size, n_classes])
    w = tf.placeholder("float", [batch_size])

    train_limit = n_seizures*3604#18040#36040  # 43985#18345 #5 seizures overlap 80%

    weights = {
        'wc1': tf.get_variable('W0', shape=(convw[0], convw[0], convw[1], 1, convw[2]), initializer=init()),
        'wd1': tf.get_variable('W1',
                               shape=(math.floor(math.ceil((1280 - convw[1]) / 8) / maxpool[1]) * convw[2], ffw[0]),
                               initializer=init()),
        'wd2': tf.get_variable('W2', shape=(ffw[0], ffw[1]), initializer=init()),
        'wd3': tf.get_variable('W3', shape=(ffw[1], ffw[2]), initializer=init()),
        'out': tf.get_variable('W4', shape=(ffw[2], n_classes), initializer=init()),
    }

    biases = {
        'bc1': tf.get_variable('B0', shape=(convw[2]), initializer=tf.ones_initializer()),
        'bd1': tf.get_variable('B1', shape=(ffw[0]), initializer=tf.ones_initializer()),
        'bd2': tf.get_variable('B2', shape=(ffw[1]), initializer=tf.ones_initializer()),
        'bd3': tf.get_variable('B3', shape=(ffw[2]), initializer=tf.ones_initializer()),
        'out': tf.get_variable('B4', shape=(n_classes), initializer=tf.ones_initializer()),
    }

    pred = conv_net_test(x, weights, biases, batch_size, strides, maxpool)
    cost = tf.losses.softmax_cross_entropy(logits=pred, onehot_labels=y, weights=w)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True

    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
      # Restore variables from disk.
      saver.restore(sess, ""+savepath+"\\"+savepath+".ckpt")
      #saver.restore(sess, "D:\\docs\\tese\\python\\models_width\\" + savepath + "\\" + savepath + "3.ckpt")
      batch = 0
      start=0
      print(train_limit//batch_size)
      skips=batch*4
      test_loss = []
      test_accuracy = []
      final_res = []
      ltrain = []
      summary_writer = tf.summary.FileWriter('./Output', sess.graph)
      while batch < train_limit// batch_size:
          batch_x = retrieve_batch(ch[:,(batch * batch_size+skips)*256:((batch + 1) * batch_size+skips)*256+256*(batch_size+4)],batch_size,l[(batch * batch_size+skips):((batch + 1) * batch_size+skips),:])
          batch_y = l[batch * batch_size + skips:(batch + 1) * batch_size + skips,:]
          #print(batch_y)
          weights_cost=cost_weights(batch_y,batch_size,relcost)
          test_acc, valid_loss, pp = sess.run([accuracy, cost, pred], feed_dict={x: batch_x, y: batch_y, w: weights_cost})
          # train_loss.append(loss)
          for i in range(len(pp)):
              final_res.append(pp[i])
              ltrain.append(batch_y[i])
          test_loss.append(valid_loss)
          # train_accuracy.append(acc)
          test_accuracy.append(test_acc)
          batch = batch + 1
          skips=skips+4
      ac = np.mean(test_accuracy)
      print("Testing Accuracy:" + "{:.5f}".format(ac))
      # print(final_res)
      false_positives = []
      false_negatives = []
      ltrain = np.array(ltrain)
      seizure_indexes = np.where(ltrain[:, 1] == 1)[0]
      inter_indexes = np.where(ltrain[:, 0] == 1)[0]
      a = np.argmax(ltrain, 1)
      b = np.argmax(final_res, 1)
      misses = a != b
      # pat 11002
      #j = 0
      #while j < len(misses):
       # pat 11002
       #if (j> 3600 and j < 4100) or (j> 12700 and j < 13300):
           # pat 98202
           #if (j > 10200 and j < 10300):
           #pat 114702
           #if (j > 130 and j < 330) or (j > 2100 and j < 2350) or (j > 4200 and j < 4500) or (j > 8550 and j < 8850) or (j > 10350 and j < 10550) :

           #misses[j]=False
      # j = j + 1
      fpr = np.sum(misses[inter_indexes]) / len(inter_indexes)
      fnr = np.sum(misses[seizure_indexes]) / len(seizure_indexes)
      print("fpr: ", fpr)
      print("fnr: ", fnr)
      seizure_starts = []
      for i in range(len(seizure_indexes)):
          if i == 0:
              seizure_starts.append(seizure_indexes[i])
          elif seizure_indexes[i] > 1 + seizure_indexes[i - 1]:
              seizure_starts.append(seizure_indexes[i])
          if len(misses) > seizure_indexes[i]:
              print(seizure_indexes[i], " ### ", misses[seizure_indexes[i]])

      print(seizure_starts)
      red_alerts=[]
      orange_alerts = []
      yellow_alerts = []
      alert_rate=0
      print('\nFalse positives')
      for i in range(len(inter_indexes)):
          if misses[inter_indexes[i]]:
              #print(inter_indexes[i])
              if alert_rate<4:
                alert_rate=alert_rate+1

              if alert_rate == 4:
                red_alerts.append(inter_indexes[i])
          else:
              if alert_rate>0:
                  alert_rate=alert_rate-1
          if alert_rate == 2:
              yellow_alerts.append(inter_indexes[i])
          if alert_rate == 3:
              orange_alerts.append(inter_indexes[i])


      #print("false red alerts")
      #for i in range(len(red_alerts)):
      #    print(red_alerts[i])
      frar = len(red_alerts) / len(inter_indexes)
      print("frar: ", frar)
      #print(red_alerts)
      near_reds=0
      preictals=[]
      reds=np.array(red_alerts)
   #   print(reds)
      prevst=0
      fsfreds=[]#far seizure false reds
      for i in range(n_seizures-start):
          st=seizure_starts[i]

          #print(((reds<st) & (reds>(st-301))))
          nr=np.where(((reds<st) & (reds>(st-301))))[0]
          fr=np.where(((reds<(st-301))& (reds>prevst)))[0]
          #print(np.array(fr))
          for j in range (len(fr)):
              fsfreds.append(red_alerts[fr[j]])
          near_reds=near_reds+len(nr)
          preictals.append(st-301)
          prevst=seizure_starts[i]
      nsrar=near_reds/ (300*n_seizures)#near seizure red alert rate
      fsfpr=(len(red_alerts)-near_reds)/(len(inter_indexes)-300*(n_seizures-start))#far seizure false positive rate
      print("nsrar: ", nsrar)
      print("fsfpr: ", fsfpr)
      #print("nfraph: ", num_alerts/n_seizures)
    #  print(np.array(fsfreds))
      fraavg=0
      frastd=0
      if len(fsfreds)>0:
          prev=fsfreds[0]
          lim_red=[prev]
          for i in range(1,len(fsfreds)):
              if prev<fsfreds[i]-3:
                  lim_red.append(prev)
                  lim_red.append(fsfreds[i])
              if i==len(fsfreds)-1:
                  lim_red.append(fsfreds[i])
                  #if prev<fsfreds[i]-3:
                  #    lim_red.append(fsfreds[i])
              prev=fsfreds[i]
          fra_lengths=[]
         # print(np.array(lim_red))
          print("num red alerts: ", len(lim_red) / 2)
          for i in range(0,len(lim_red),2):
              fra_lengths.append(lim_red[i+1]-lim_red[i]+1)
              #print(i)
          print(fra_lengths)
          fraavg=stat.mean(fra_lengths)
          frastd=stat.stdev(fra_lengths)

      print("fraavg: ", fraavg)
      print("frastd: ", frastd)


      first_reds = []
      seiz_count = 0
      flag = 0
      prev = 0
      alert_rate = 0
      tras=[]
      toas=[]
      for i in range(len(seizure_indexes)):
          a = seizure_indexes[i] - 1
          if prev != a:
              seiz_count = 0
              flag = 0
              if a in yellow_alerts:
                  alert_rate = 2
              elif a in orange_alerts:
                  alert_rate = 3
              elif a in red_alerts:
                  alert_rate = 4
                  first_reds.append(seiz_count)
                  flag = 1
              elif misses[a] == True:
                  alert_rate = 1
              else:
                  alert_rate = 0
          if ~misses[seizure_indexes[i]]:
              if alert_rate < 4:
                  alert_rate = alert_rate + 1

              if alert_rate == 4:
                  if flag == 0:
                      first_reds.append(seiz_count)
                      flag = 1
                  red_alerts.append(seizure_indexes[i])
                  tras.append(seizure_indexes[i])
          else:
              if alert_rate > 0:
                  alert_rate = alert_rate - 1
          if alert_rate == 2:
              yellow_alerts.append(seizure_indexes[i])
          if alert_rate == 3:
              orange_alerts.append(seizure_indexes[i])
              toas.append(seizure_indexes[i])
          seiz_count = seiz_count + 1
          prev = seizure_indexes[i]

     # prev = tras[0]
     # lim_red = [prev]
     # for i in range(1, len(tras)):
        #  if prev < tras[i] - 3:
        #      lim_red.append(prev)
        #      lim_red.append(tras[i])
       #   if i == len(tras) - 1:
        #      lim_red.append(tras[i])
              # if prev<fsfreds[i]-3:
              #    lim_red.append(fsfreds[i])
         # prev = tras[i]
     # #tra_lengths = []
      # print(np.array(lim_red))
      #print("num red alerts: ", len(lim_red) / 2)
      #for i in range(0, len(lim_red), 2):
        #  tra_lengths.append(lim_red[i + 1] - lim_red[i] + 1)
          # print(i)
      #print(tra_lengths)
      print("tra_length per seizure length", (len(tras)/n_seizures)/(len(seizure_indexes)/n_seizures))
      print((len(toas)/n_seizures)/(len(seizure_indexes)/n_seizures))

      print('\nRed alerts')
      print(first_reds)
      print(sum(first_reds)/len(first_reds))
      red_alerts=np.array(red_alerts)
      #for i in range(len(red_alerts)):
       #   print(red_alerts[i])
      orange_alerts = np.array(orange_alerts)
      #for i in range(len(orange_alerts)):
       #   print(orange_alerts[i])
      yellow_alerts = np.array(yellow_alerts)
      #for i in range(len(yellow_alerts)):
       #   print(yellow_alerts[i])
      i = 0
      while (1):
          print(i)
          fig, ax = plt.subplots()
          red_alerts_now = red_alerts[red_alerts > i * 3600]
          red_alerts_now = red_alerts_now[red_alerts_now < (i + 1) * 3600]
          orange_alerts_now = orange_alerts[orange_alerts > i * 3600]
          orange_alerts_now = orange_alerts_now[orange_alerts_now < (i + 1) * 3600]
          yellow_alerts_now = yellow_alerts[yellow_alerts > i * 3600]
          yellow_alerts_now = yellow_alerts_now[yellow_alerts_now < (i + 1) * 3600]
          plt.yticks([0,1])
          plt.ylabel('Not ictal/ Ictal.')
          plt.xlabel('Time (s). Purple line = seizure onset')
          plt.plot(range(i * 3600, (i + 1) * 3600), b[i * 3600:(i + 1) * 3600], linewidth=0.15,linestyle='--')
          plt.plot([seizure_starts[i], seizure_starts[i]], [0, 1], linewidth=1, color='m')  # ,facecolors = 'red'
          plt.scatter(yellow_alerts_now, np.ones(len(yellow_alerts_now)) * 0.5, s=1.5, color='gold')
          build_scatter(yellow_alerts_now, ax, 'gold')
          plt.scatter(orange_alerts_now, np.ones(len(orange_alerts_now)) * 0.5, s=1.5, color='darkorange')
          build_scatter(orange_alerts_now, ax, 'darkorange')
          plt.scatter(red_alerts_now,np.ones(len(red_alerts_now))*0.5, s=1.5, color='r')
          build_scatter(red_alerts_now, ax, 'r')
          plt.show()
          a = input('where do you want to move the window? Ex: 1 moves 1 forward, -2 moves 2 back, string ends')
          if a.isdigit():
              print(a)
              i = i + int(a)
          else:
              break
      summary_writer.close()
      score=fpr*12+fnr
      return [fpr,fnr,score]

#test_net([3,48,1],[18,18,6],50,[1,8],[3,14],"model_"+str(50)+"_cost_2_18_neurons_50")
#test_net([3,48,1],[13,13,4],60,[1,7],[3,12],"model_13_neuron_width")

#test_net([4,32,3],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"first_model_pat_114902_100_iters")

#test_net([4,32,1],[21,21,7],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"first_model_pat_11002_nfilters_1_width_21",4,"channels_11002_test.mat","labels_11002_test_p4.mat")

#test_net([4,32,3],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"first_model_pat_11002",4,"channels_11002_test.mat","labels_11002_test_p4.mat")

#test_net([4,16,1],[21,21,7],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"model_pat_30802_n_filters_1_cost_70_width_21_z_16_clinical_high_4",4,"channels_30802_test_high.mat","labels_30802_test_p4.mat")

#test_net([4,32,3],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"first_model_pat_114902",5,"channels_114902_test.mat","labels_114902_test_p4.mat")

#test_net([4,32,3],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"pat_58602_no_clinical",10,"channels_58602_test.mat","labels_58602_test.mat")

#test_net([4,32,3],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"pat_81102_8_seizures_cost_100",4,"channels_81102_test_4_seizures.mat","labels_81102_test.mat")

#test_net([4,32,3],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"pat_85202_6_seizures",4,"channels_85202_test_final.mat","labels_85202_test_final.mat")

#test_net([4,32,1],[21,21,7],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"pat_109502_cost_40_nfilters_1_z_21",4,"channels_109502_test.mat","labels_109502_test.mat")

#test_net([4,32,3],[42,42,14],40,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"pat_98202",4,"channels_98202_test.mat","labels_98202_test.mat")

#test_net([4,16,1],[42,42,14],40,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"pat_98202_nfilters_1_cost_12_z_16",4,"channels_98202_test.mat","labels_98202_test_p4.mat")

#test_net([4,32,1],[21,21,7],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"pat_109502_cost_40_nfilters_1_z_21",4,"channels_109502_test_2.mat","labels_109502_test_p4.mat")

#test_net([4,16,1],[42,42,14],40,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"pat_55202_2_cost_70_z_16_nfilters_1",4,"channels_55202_test_2.mat","labels_55202_test_2.mat")

#test_net([4,32,3],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"pat_109502_2_cost_12",3,"channels_109502_test_2.mat","labels_109502_test_p4.mat")

#test_net([4,16,1],[21,21,7],20,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"pat_94402_z_16_nfilters_1_cost_180",4,"channels_94402_test.mat","labels_94402_test.mat")

#test_net([4,32,1],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"pat_96002_nfilters_1",4,"channels_96002_test.mat","labels_96002_test.mat")

#test_net([4,32,1],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"pat_94402_6_seizures_cost_180_nfilters_1",3,"channels_94402_test_6_seizures.mat","labels_94402_test_6_seizures.mat")

#test_net([4,32,3],[42,42,14],40,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"pat_55202_6_seizures_cost_150",3,"channels_55202_test_6_seizures.mat","labels_55202_test_p4.mat")

#test_net([4,32,3],[42,42,14],40,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"pat_102202",10,"channels_102202_test","labels_102202_test.mat")

#test_net([4,32,3],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"pat_113902_cost_40",6,"channels_113902_test_6_seizures.mat","labels_113902_test_6_seizures.mat")

#test_net([4,32,3],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"pat_114702",3,"channels_114702_test.mat","labels_114702_test.mat")

#test_net([4,32,3],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"pat_30802_5",4,"channels_30802_test_2.mat","labels_30802_test_2.mat")

#test_net([4,32,3],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"mixed",22,"channels_mixed_test","labels_mixed_test.mat")

test_net([4,32,1],[39,39,13],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"model_32_convz_2_3",10,"channels_58602_test.mat","labels_58602_test.mat")

