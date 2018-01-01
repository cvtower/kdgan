import tensorflow as tf

from os import path

temp_dir = 'temp'
alp_ckpt = path.join(temp_dir, 'alp.ckpt')
bet_ckpt = path.join(temp_dir, 'bet.ckpt')

def save_var():
    alp = tf.get_variable('alp', shape=[3], initializer = tf.zeros_initializer)
    bet = tf.get_variable('bet', shape=[5], initializer = tf.zeros_initializer)
    inc_alp = alp.assign(alp + 1)
    dec_bet = bet.assign(bet - 1)
    init_op = tf.global_variables_initializer()
    saver_alp = tf.train.Saver({'alp':alp})
    saver_bet = tf.train.Saver({'bet':bet})
    with tf.Session() as sess:
        sess.run(init_op)
        inc_alp.op.run()
        dec_bet.op.run()
        path_alp = saver_alp.save(sess, alp_ckpt)
        print('save alp in %s' % path_alp)
        path_bet = saver_bet.save(sess, bet_ckpt)
        print('save bet in %s' % path_bet)

def restore_var():
    alp = tf.get_variable('alp', shape=[3])
    # bet = tf.get_variable('bet', shape=[5])
    bet = tf.get_variable('bet', shape=[5], initializer = tf.zeros_initializer)
    saver_alp = tf.train.Saver({'alp':alp})
    # saver_bet = tf.train.Saver({'bet':bet})
    with tf.Session() as sess:
        # saver_alp.restore(sess, alp_ckpt)
        # saver_bet.restore(sess, bet_ckpt)
        saver_alp.restore(sess, alp_ckpt)
        bet.initializer.run()
        print('alp: %s' % alp.eval())
        print('bet: %s' % bet.eval())

def main():
    # save_var()
    restore_var()

if __name__ == '__main__':
    main()