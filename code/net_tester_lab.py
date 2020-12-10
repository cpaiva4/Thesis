from utils import *
import hdf5storage
import math
import statistics as stat
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

def test_net(convw,ffw,relcost,strides,maxpool,init,savepath,start,end,ch_file,label_file):
    labels = hdf5storage.loadmat(label_file)#labels_114902_train.mat')#labels_586202_batch_3608.mat')

    ch = hdf5storage.loadmat(ch_file)#channels_114902_train.mat')#channels_586202_ordered_batch_3608.mat')
    l=labels['labels']
    ch=ch['ch']

    training_iters = 400
    learning_rate = 0.0005
    batch_size = 3600

    n_input = 5 * 5 * 5 * 256

    n_classes = 2

    x = tf.placeholder("float", [batch_size, 5, 5, 1280, 1])
    y = tf.placeholder("float", [batch_size, n_classes])
    w = tf.placeholder("float", [batch_size])

    train_limit = end*3608#36080  # 43985#18345 #5 seizures overlap 80%

    weights = {
        'wc1': tf.get_variable('W0', shape=(convw[0], convw[0], convw[1], 1, convw[2]), initializer=init()),
        'wd1': tf.get_variable('W1', shape=(math.floor(math.ceil((1280-convw[1])/8)/maxpool[1])*convw[2], ffw[0]), initializer=init()),
        'wd2': tf.get_variable('W2', shape=(ffw[0], ffw[1]), initializer=init()),
        'wd3': tf.get_variable('W3', shape=(ffw[1], ffw[2]), initializer=init()),
        'out': tf.get_variable('W4', shape=(ffw[2], n_classes), initializer=init()),
    }

    if init!=tf.orthogonal_initializer:
        biases = {
            'bc1': tf.get_variable('B0', shape=(convw[2]), initializer=init()),
            'bd1': tf.get_variable('B1', shape=(ffw[0]), initializer=init()),
            'bd2': tf.get_variable('B2', shape=(ffw[1]), initializer=init()),
            'bd3': tf.get_variable('B3', shape=(ffw[2]), initializer=init()),
            'out': tf.get_variable('B4', shape=(n_classes), initializer=init()),
        }
    else:
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
      saver.restore(sess, savepath+"\\"+savepath+".ckpt")

      #batch = train_limit // batch_size
      batch=start

      print(batch)
      test_loss = []
      test_accuracy = []
      final_res = []
      ltrain = []
      prev_loss = 999
      #skips=80
      skips=8*batch
      summary_writer = tf.summary.FileWriter('./Output', sess.graph)
      while batch < train_limit // batch_size:
          # batch_x = inp[batch * batch_size:min((batch + 1) * batch_size, len(l) - 1)]
          [batch_x, la, cut] = retrieve_batch(ch[:, (batch * batch_size + skips) * 256:(
                                                                                               batch * batch_size + skips) * 256 + 1280 + 256 * (
                                                                                               batch_size + 4)],
                                              batch_size,
                                              l[batch * batch_size + skips:batch * batch_size + skips + (batch_size + 4),
                                              :])
          # batch_y = l[batch * batch_size:min((batch + 1) * batch_size, train_limit)]
          if cut == -1:
              batch_y = l[batch * batch_size + skips:batch * batch_size + skips + batch_size, :]
          else:
              batch_y = np.vstack((l[batch * batch_size + skips:batch * batch_size + skips + cut, :],
                                   l[batch * batch_size + skips + cut + 4:batch * batch_size + skips + (batch_size + 4),
                                   :]))
          # print(batch)
          # print(skips)
          # print(batch * batch_size + skips)
          # print(batch_y)
          # print(batch_y)
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
          skips = skips + la
          skips = skips + 4  # for full batch training
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
      fpr = np.sum(misses[inter_indexes]) / len(inter_indexes)
      fnr = np.sum(misses[seizure_indexes]) / len(seizure_indexes)
      print("fpr: ", fpr)
      print("fnr: ", fnr)
      score = fpr * 12 + fnr
      seizure_starts = []
      for i in range(len(seizure_indexes)):
          if i == 0:
              seizure_starts.append(seizure_indexes[i])
          elif seizure_indexes[i] > 1 + seizure_indexes[i - 1]:
              seizure_starts.append(seizure_indexes[i])
          if len(misses) > seizure_indexes[i]:
              print(seizure_indexes[i], " ### ", misses[seizure_indexes[i]])
      red_alerts = []
      orange_alerts = []
      yellow_alerts = []
      alert_rate = 0
      print('\nFalse positives')
      for i in range(len(inter_indexes)):
          if misses[inter_indexes[i]]:
              # print(inter_indexes[i])
              if alert_rate < 4:
                  alert_rate = alert_rate + 1

              if alert_rate == 4:
                  red_alerts.append(inter_indexes[i])
          else:
              if alert_rate > 0:
                  alert_rate = alert_rate - 1
          if alert_rate == 2:
              yellow_alerts.append(inter_indexes[i])
          if alert_rate == 3:
              orange_alerts.append(inter_indexes[i])
      # print("false red alerts")
      # for i in range(len(red_alerts)):
      #    print(red_alerts[i])
      frar = len(red_alerts) / len(inter_indexes)
      #print("frar: ", frar)
      # print(red_alerts)
      near_reds = 0
      preictals = []
      reds = np.array(red_alerts)
      #   print(reds)
      prevst = 0
      fsfreds = []  # far seizure false reds
      for i in range(1):
          st = seizure_starts[i]

          # print(((reds<st) & (reds>(st-301))))
          nr = np.where(((reds < st) & (reds > (st - 301))))[0]
          fr = np.where(((reds < (st - 301)) & (reds > prevst)))[0]
          # print(np.array(fr))
          for j in range(len(fr)):
              fsfreds.append(red_alerts[fr[j]])
          near_reds = near_reds + len(nr)
          preictals.append(st - 301)
          prevst = seizure_starts[i]
      nsrar = near_reds / (300)  # near seizure red alert rate
      fsfpr = (len(red_alerts) - near_reds) / len(inter_indexes)  # far seizure false positive rate
      # print("nfraph: ", num_alerts/n_seizures)
      #  print(np.array(fsfreds))

      prev = fsfreds[0]
      lim_red = [prev]
      for i in range(1, len(fsfreds)):
          if prev < fsfreds[i] - 3:
              lim_red.append(prev)
              lim_red.append(fsfreds[i])
          if i == len(fsfreds) - 1:
              lim_red.append(fsfreds[i])
              # if prev<fsfreds[i]-3:
              #    lim_red.append(fsfreds[i])
          prev = fsfreds[i]
      fra_lengths = []
      # print(np.array(lim_red))
      print("num red alerts: ", len(lim_red) / 2)
      #for i in range(0, len(lim_red), 2):
       #   fra_lengths.append(lim_red[i + 1] - lim_red[i] + 1)
          # print(i)
      print(fra_lengths)
      #fraavg = stat.mean(fra_lengths)
      #frastd = stat.stdev(fra_lengths)


      first_reds = []
      seiz_count = 0
      flag = 0
      prev = 0
      alert_rate = 0
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
          else:
              if alert_rate > 0:
                  alert_rate = alert_rate - 1
          if alert_rate == 2:
              yellow_alerts.append(seizure_indexes[i])
          if alert_rate == 3:
              orange_alerts.append(seizure_indexes[i])
          seiz_count = seiz_count + 1
          prev = seizure_indexes[i]

    return [fpr, fnr, frar, nsrar, fsfpr]

#test_net([4,32,3],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,"first_model_pat_114902_200 iters")