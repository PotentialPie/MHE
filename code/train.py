import os
import argparse
import pickle
import numpy as np
import tensorflow as tf
import cifar100_input_python as ip
from loss import loss2
from architecture import VGG, CNNSimple
import time

# # python 2
# def unpickle(file):
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo)
#     return dict

# python 3
def unpickle(file):
    with open(file, 'rb') as fo:
        u = pickle._Unpickler(fo)
        u.encoding = 'latin1'
        dict = u.load()
    return dict

train_data = unpickle('../train')
test_data = unpickle('../test')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path

def Get_Average(list):
   sum = 0.
   for item in list:
      sum += item
   return sum/len(list)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

acc_list = []
time_list = []
n = 5

def train(base_lr, batch_sz, gpu_no, model_name, power_s, architecture, width, used_in_H, used_in_O, visual):

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_no

    cnn2depth = {'CNN6': 2, 'CNN9': 3, 'CNN15': 5}
    cnn2width = {'0': [16,32,64], '1': [32,64,128], '2': [64,128,254736s6], '3': [128,256,512], '4': [256,512,1024]}

    used_in_H = str2bool(used_in_H)
    used_in_O = str2bool(used_in_O)
    visual = str2bool(visual)

    root_path = os.path.dirname(os.path.realpath(__file__))
    exp_name = '%s_%s_power%s_width%s'%(architecture, model_name, power_s, width)
    if used_in_H:
        exp_name = exp_name + '_H'
    if used_in_O:
        exp_name = exp_name + '_O'
    log_path = create_dir(os.path.join(root_path, 'log', exp_name))
    save_path = create_dir(os.path.join(root_path, 'weights', exp_name))

    acc_count = 0
    while acc_count < 100:
        if os.path.exists(os.path.join(log_path, 'log_test_%02d.txt' % acc_count)):
            acc_count += 1
        else:
            break
    assert acc_count < 100

    log_train_fname = 'log_train_%02d.txt' % acc_count
    log_test_fname = 'log_test_%02d.txt' % acc_count
    log_test_all_fname = 'log_test_all.txt'

    n_class = 100
    batch_sz = batch_sz
    batch_test = 100
    max_epoch = 42500
    lr = base_lr
    momentum = 0.9
    is_training = tf.placeholder("bool")

    images = tf.placeholder(tf.float32, (None, 32, 32, 3))
    labels = tf.placeholder(tf.int32, (None))

    n_layer = cnn2depth[architecture]
    width = cnn2width[width]

    cnn = CNNSimple()
    cnn.build(images, n_class, is_training, model_name, power_s, n_layer, width, used_in_H, used_in_O, visual)

    fit_loss = loss2(cnn.score, labels, n_class, 'c_entropy')
    loss_op = fit_loss
    reg_loss_list = tf.losses.get_regularization_losses()
    if len(reg_loss_list) != 0:
        reg_loss = tf.add_n(reg_loss_list)
        loss_op += reg_loss

    thom_loss_list = tf.get_collection('thomson_loss')
    if len(thom_loss_list) != 0:
        thom_loss = tf.add_n(thom_loss_list)
        loss_op += thom_loss

    thom_final_list = tf.get_collection('thomson_final')
    if len(thom_final_list) != 0:
        thom_final = tf.add_n(thom_final_list)
        loss_op += thom_final

    lr_ = tf.placeholder("float")
    update_op = tf.train.MomentumOptimizer(lr_, 0.9).minimize(loss_op)
    predc = cnn.pred
    acc_op = tf.reduce_mean(tf.to_float(tf.equal(labels, tf.to_int32(cnn.pred))))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        tf.summary.scalar('fit loss', fit_loss)
        if len(reg_loss_list) != 0:
            tf.summary.scalar('reg loss', reg_loss)
        if len(thom_loss_list) != 0:
            tf.summary.scalar('thomson loss', thom_loss)
        if len(thom_final_list) != 0:
            tf.summary.scalar('thomson final loss', thom_final)
        tf.summary.scalar('learning rate', lr)
        tf.summary.scalar('accuracy', acc_op)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(root_path + '/tf_log/%s' % exp_name, sess.graph)

        print ("====================")
        print ("Log will be saved to: " + log_path)

        with open(os.path.join(log_path, log_train_fname), 'w'):
            pass
        with open(os.path.join(log_path, log_test_fname), 'w'):
            pass

        with open(os.path.join(log_path, log_train_fname), 'a') as train_acc_file:
            train_acc_file.write('model_name: %s, power_s: %s\n' %(model_name, power_s))
        with open(os.path.join(log_path, log_test_fname), 'a') as test_acc_file:
            test_acc_file.write('model_name: %s, power_s: %s\n' %(model_name, power_s))

        print('starting training........')

        start = time.time()

        for i in range(max_epoch):
            t = i % 390
            if t == 0:
                idx = np.arange(0, 50000)
                np.random.shuffle(idx)
                train_data['data'] = train_data['data'][idx]
                train_data['fine_labels'] = np.reshape(train_data['fine_labels'], [50000])
                train_data['fine_labels'] = train_data['fine_labels'][idx]
            tr_images, tr_labels = ip.load_train(train_data, batch_sz, t)

            if i == 20000:
                lr *= 0.1
            elif i == 30000:
                lr *= 0.1
            elif i == 37500:
                lr *= 0.1

            if len(thom_loss_list) != 0:
                if len(thom_final_list) != 0:
                    summary, fit, reg, thom, thomf, acc, _ = sess.run([merged,  fit_loss, reg_loss, thom_loss, thom_final, acc_op, update_op],
                                                        {lr_: lr, is_training: True, images: tr_images, labels: tr_labels})

                    if i % 100 == 0 and i != 0:
                        print('====iter_%d: fit=%.4f, reg=%.4f, thom=%.4f, thomf=%.4f, acc=%.4f'
                            % (i, fit, reg, thom, thomf, acc))
                        with open(os.path.join(log_path, log_train_fname), 'a') as train_acc_file:
                            train_acc_file.write('====iter_%d: fit=%.4f, reg=%.4f, thom=%.4f, thomf=%.4f, acc=%.4f\n'
                            % (i, fit, reg, thom, thomf, acc))
                    train_writer.add_summary(summary, i)
                else:
                    summary, fit, reg, thom, acc, _ = sess.run(
                        [merged, fit_loss, reg_loss, thom_loss, acc_op, update_op],
                        {lr_: lr, is_training: True, images: tr_images, labels: tr_labels})

                    if i % 100 == 0 and i != 0:
                        print('====iter_%d: fit=%.4f, reg=%.4f, thom=%.4f, acc=%.4f'
                              % (i, fit, reg, thom, acc))
                        with open(os.path.join(log_path, log_train_fname), 'a') as train_acc_file:
                            train_acc_file.write('====iter_%d: fit=%.4f, reg=%.4f, thom=%.4f, acc=%.4f\n'
                                                 % (i, fit, reg, thom, acc))
                    train_writer.add_summary(summary, i)
            else:
                summary, fit, reg, acc, _ = sess.run([merged, fit_loss, reg_loss, acc_op, update_op],
                                                    {lr_: lr, is_training: True, images: tr_images, labels: tr_labels})

                if i % 100 == 0 and i != 0:
                    print('====iter_%d: fit=%.4f, reg=%.4f, acc=%.4f'
                        % (i, fit, reg, acc))
                    with open(os.path.join(log_path, log_train_fname), 'a') as train_acc_file:
                        train_acc_file.write('====iter_%d: fit=%.4f, reg=%.4f, acc=%.4f\n'
                        % (i, fit, reg, acc))
                train_writer.add_summary(summary, i)


            if i % 500 == 0 and i != 0:
                n_test = 10000
                acc = 0.0
                for j in range(int(n_test/batch_test)):
                    te_images, te_labels = ip.load_test(test_data, batch_test, j)
                    acc = acc + sess.run(acc_op, {is_training: False, images: te_images, labels: te_labels})
                acc = acc * batch_test / float(n_test)
                print('++++iter_%d: test acc=%.4f' % (i, acc))
                with open(os.path.join(log_path, log_test_fname), 'a') as test_acc_file:
                    test_acc_file.write('++++iter_%d: test acc=%.4f\n' % (i, acc))

            if i%10000==0 and i!=0:
                tf.train.Saver().save(sess, os.path.join(save_path, str(i)))
        tf.train.Saver().save(sess, os.path.join(save_path, str(i)))


        n_test = 10000
        acc = 0.0
        for j in range(int(n_test/batch_test)):
            te_images, te_labels = ip.load_test(test_data, batch_test, j)
            acc = acc + sess.run(acc_op, {is_training: False, images: te_images, labels: te_labels})
        acc = acc * batch_test / float(n_test)
        print('++++iter_%d: test acc=%.4f' % (i, acc))
        with open(os.path.join(log_path, log_test_fname), 'a') as test_acc_file:
            test_acc_file.write('++++iter_%d: test acc=%.4f\n' % (i, acc))

        end = time.time()
        total_cost = end - start
        print('====total time: %.4f\n' % total_cost)

        with open(os.path.join(log_path, log_test_all_fname), 'a') as test_acc_file:
            test_acc_file.write('this time, test acc=%.4f, cost time=%.4f\n' % (acc, total_cost))

        acc_list.append(round(100.-acc*100., 4))
        time_list.append(total_cost)
        if len(acc_list) == n:
            with open(os.path.join(log_path, log_test_all_fname), 'a') as test_acc_file:
                test_acc_file.write('Totally, the average test error=%.2f, the average cost time=%.4f\n' %\
                                    (Get_Average(acc_list), Get_Average(time_list)))
                test_acc_file.write(','.join(list(map(str, acc_list))))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='take parameters')
    parser.add_argument('--base_lr', type=float, default=1e-1,
                    help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
    parser.add_argument('--gpu_no', type=str, default='0',
                    help='gpu no')
    parser.add_argument('--model_name', type=str, default='mhe',
                    help='model name: [baseline, mhe, half_mhe]')
    parser.add_argument('--power_s', type=str, default='0',
                    help='power s: [0, 1, 2, a0, a1, a2]')
    parser.add_argument('--architecture', type=str, default='CNN9',
                        help='architecture: [CNN6, CNN9, CNN15, ResNet18, ResNet34]')
    parser.add_argument('--width', type=str, default='2',
                        help="width: {'0': [16,32,64], '1': [32,64,128], '2': [64,128,256], '3': [128,256,512],\
                         '4': [256,512,1024]}")
    parser.add_argument('--used_in_H', type=str, default='True',
                        help='True if use the MHE in hidden layer')
    parser.add_argument('--used_in_O', type=str, default='True',
                        help='True if use the MHE in output layer')
    parser.add_argument('--visual', type=str, default='False',
                        help='True if to visualization the embedding features')

    args = parser.parse_args()

    for i in range(n):
        tf.reset_default_graph()
        train(args.base_lr, args.batch_size, args.gpu_no, args.model_name, args.power_s, args.architecture, args.width,\
              args.used_in_H, args.used_in_O, args.visual)



