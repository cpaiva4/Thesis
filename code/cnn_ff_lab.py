from utils import *
import hdf5storage
import math
tf=tf2.compat.v1
tf.disable_v2_behavior()

def conv_net_test(x, weights, biases, batch_size,strides,maxpool):
    conv1 = convimp(x, weights['wc1'], biases['bc1'],strides[0],strides[1],"VALID")
    conv1 = maxpoolimp(conv1,maxpool[0],maxpool[1],"VALID")
    #conv2 = convimp(conv1, weights['wc2'], biases['bc2'], 1,4, "VALID")
    #conv2 = maxpoolimp(conv2, 2, 2, "VALID")
    s=tf.shape(conv1)
    fc1 = tf.reshape(conv1, [batch_size, s[1]*s[2]*s[3]*s[4]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc2=tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3'])
    fc3 = tf.nn.relu(fc3)
    out = tf.add(tf.matmul(fc3, weights['out']), biases['out'])
    return out

def net_train(convw,ffw,relcost,strides,maxpool,init,lr,savepath,n_seizures,ch_file,label_file):
    labels = hdf5storage.loadmat(label_file)#labels_114902_train.mat')#labels_586202_batch_3608.mat')

    ch = hdf5storage.loadmat(ch_file)#channels_114902_train.mat')#channels_586202_ordered_batch_3608.mat')
    l=labels['labels']
    ch=ch['ch']

    training_iters = 1
    learning_rate = lr
    batch_size = 3600

    n_input = 5 * 5 * 5 * 256

    n_classes = 2

    x = tf.placeholder("float", [batch_size, 5, 5, 1280, 1])
    y = tf.placeholder("float", [batch_size, n_classes])
    w = tf.placeholder("float", [batch_size])

    train_limit = n_seizures*3608#21648#36080  # 43985#18345 #5 seizures overlap 80%

    weights = {
        'wc1': tf.get_variable('W0', shape=(convw[0], convw[0], convw[1], 1, convw[2]), initializer=init()),
        'wd1': tf.get_variable('W1', shape=(math.floor(math.ceil((1280-convw[1])/8)/maxpool[1])*convw[2], ffw[0]), initializer=init()),
        'wd2': tf.get_variable('W2', shape=(ffw[0], ffw[1]), initializer=init()),
        'wd3': tf.get_variable('W3', shape=(ffw[1], ffw[2]), initializer=init),
        'out': tf.get_variable('W4', shape=(ffw[2], n_classes), initializer=init),
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

    pred = conv_net_test(x, weights, biases, batch_size,strides,maxpool)
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
        sess.run(init)
        #saver.restore(sess, "D:\\docs\\tese\\python\\models_pat_2\\" + savepath + "\\" + savepath + ".ckpt")
        summary_writer = tf.summary.FileWriter('./Output', sess.graph)
        for i in range(training_iters):
            totalacc = 0
            totalloss = 0
            batch = 0
            skips = 0

            while batch < (train_limit - skips) // batch_size:
                [batch_x, la, cut] = retrieve_batch(ch[:, (batch * batch_size + skips) * 256:(
                                                                                                         batch * batch_size + skips) * 256 + 1280 + 256 * (
                                                                                                         batch_size + 4)],
                                                    batch_size, l[
                                                                batch * batch_size + skips:batch * batch_size + skips + (
                                                                            batch_size + 4), :])
                if cut == -1:
                    batch_y = l[batch * batch_size + skips:batch * batch_size + skips + batch_size, :]
                else:
                    batch_y = np.vstack((l[batch * batch_size + skips:batch * batch_size + skips + cut, :], l[
                                                                                                            batch * batch_size + skips + cut + 4:batch * batch_size + skips + (
                                                                                                                        batch_size + 4),
                                                                                                            :]))
                weights_cost = cost_weights(batch_y, batch_size,relcost)
                opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, w: weights_cost})
                loss, acc, p, c = sess.run([cost, accuracy, pred, correct_prediction],
                                           feed_dict={x: batch_x, y: batch_y, w: weights_cost})
                totalacc += acc * (1 / (train_limit // batch_size))
                totalloss += loss * (1 / (train_limit // batch_size))
                batch = batch + 1
                skips = skips + la
                skips = skips + 4  # for full batch training

            print("Iter " + str(i) + ", Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            print(p)
            print("iteration Finished!")
            print(c)
            print("Total accuracy:")
            print(totalacc)
            print("Total loss:")
            print(totalloss)
            if totalloss < 0.05:
                break
            # print (batch_y)

            if (i+1) % 50 == 0:
                print("saving...")
                save_path = saver.save(sess, ""+savepath+"\\"+savepath+".ckpt")
                print("saved")
            # Calculate accuracy for all 10000 mnist test images
        print("saving...")
        save_path = saver.save(sess, ""+savepath+"\\"+savepath+".ckpt")
        print("saved")
        print("train finished")
        summary_writer.close()

#net_train([4,32,3],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0001,"first_model_pat_114902_100_iters")

#net_train([4,32,3],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0001,"first_model_pat_114902_other",6,"channels_114902_train.mat","labels_114902_train.mat")

#net_train([4,16,1],[21,21,7],70,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0005,"model_pat_30802_n_filters_1_cost_70_width_21_z_16_clinical_high_4",4,"channels_30802_train_high.mat","labels_30802_train_clinical.mat")

net_train([4,32,3],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0005,"pat_58602_demo",10,"channels_586202_ordered_batch_3608.mat","labels_586202_batch_3608.mat")

#net_train([4,32,1],[21,21,7],60,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0005,"first_model_pat_11002_nfilters_1_width_21_cost_60",4,"channels_11002_train.mat","labels_11002_train.mat")

#net_train([4,32,3],[42,42,14],225,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0005,"pat_81102_8_seizures_cost_225",8,"channels_81102_train_8_seizures.mat","labels_81102_train_8_seizures.mat")

#net_train([4,32,3],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0005,"pat_85202_6_seizures",6,"channels_85202_train_6_seizures.mat","labels_85202_train_6_seizures.mat")

#net_train([4,32,1],[21,21,7],40,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0005,"pat_109502_cost_40_nfilters_1_z_21",6,"channels_109502_train.mat","labels_109502_train.mat")

#net_train([4,32,3],[42,42,14],12,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0005,"pat_98202_cost_12",5,"channels_98202_train.mat","labels_98202_train.mat")

#net_train([4,16,1],[42,42,14],70,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0005,"pat_55202_2_cost_70_z_16_nfilters_1",5,"channels_55202_train_2.mat","labels_55202_train_2.mat")

#net_train([4,32,3],[42,42,14],9,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0005,"pat_109502_2_cost_9",6,"channels_109502_train_2.mat","labels_109502_train_2.mat")

#net_train([4,16,1],[21,21,7],180,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0005,"pat_94402_z_16_nfilters_1_cost_180",5,"channels_94402_train.mat","labels_94402_train.mat")

#net_train([4,16,1],[21,21,7],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0005,"pat_96002_nfilters_1_z_16_width_21",4,"channels_96002_train.mat","labels_96002_train.mat")

#net_train([4,32,1],[42,42,14],180,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0005,"pat_94402_6_seizures_cost_180_nfilters_1",6,"channels_94402_train_6_seizures.mat","labels_94402_train_6_seizures.mat")

#net_train([4,32,3],[42,42,14],100,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0005,"pat_55202_6_seizures_cost_150",6,"channels_55202_train_6_seizures.mat","labels_55202_train_6_seizures.mat")

#net_train([4,32,3],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0005,"pat_102202_5_seizures",8,"channels_102202_train.mat","labels_102202_train.mat")

#net_train([4,32,3],[42,42,14],40,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0005,"pat_113902_cost_40",9,"channels_113902_train.mat","labels_113902_train.mat")

#net_train([4,32,3],[42,42,14],30,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0005,"pat_114702_cost_30",11,"channels_114702_train.mat","labels_114702_train.mat")

#net_train([4,32,3],[42,42,14],225,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0005,"pat_30802_5_cost_225",5,"channels_30802_train_2.mat","labels_30802_train_2.mat")

#net_train([4,32,3],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0005,"mixed",46,"channels_mixed_train.mat","labels_mixed_train.mat")

#net_train([4,32,3],[42,42,14],50,[1,8],[2,math.floor((1280-32)/(13*8))],tf.orthogonal_initializer,0.0005,"pat_85202_5_seizures",5,"channels_85202_train_final.mat","labels_85202_train_final.mat")

