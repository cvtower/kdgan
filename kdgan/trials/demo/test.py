import tensorflow as tf

from os import path

# temp_dir = 'temp'
# alp_ckpt = path.join(temp_dir, 'alp.ckpt')
# bet_ckpt = path.join(temp_dir, 'bet.ckpt')

# def save_var():
#     alp = tf.get_variable('alp', shape=[3], initializer = tf.zeros_initializer)
#     bet = tf.get_variable('bet', shape=[5], initializer = tf.zeros_initializer)
#     inc_alp = alp.assign(alp + 1)
#     dec_bet = bet.assign(bet - 1)
#     init_op = tf.global_variables_initializer()
#     saver_alp = tf.train.Saver({'alp':alp})
#     saver_bet = tf.train.Saver({'bet':bet})
#     with tf.Session() as sess:
#         sess.run(init_op)
#         inc_alp.op.run()
#         dec_bet.op.run()
#         path_alp = saver_alp.save(sess, alp_ckpt)
#         print('save alp in %s' % path_alp)
#         path_bet = saver_bet.save(sess, bet_ckpt)
#         print('save bet in %s' % path_bet)

# def restore_var():
#     alp = tf.get_variable('alp', shape=[3])
#     # bet = tf.get_variable('bet', shape=[5])
#     bet = tf.get_variable('bet', shape=[5], initializer = tf.zeros_initializer)
#     saver_alp = tf.train.Saver({'alp':alp})
#     # saver_bet = tf.train.Saver({'bet':bet})
#     with tf.Session() as sess:
#         # saver_alp.restore(sess, alp_ckpt)
#         # saver_bet.restore(sess, bet_ckpt)
#         saver_alp.restore(sess, alp_ckpt)
#         bet.initializer.run()
#         print('alp: %s' % alp.eval())
#         print('bet: %s' % bet.eval())

# def main():
#     # save_var()
#     restore_var()

# if __name__ == '__main__':
#     main()


# [0, 1, 2, 3, 4 ,...]
x = tf.range(1, 10, name="x")
x = ['0', '1', '2', '3', '4', '5']

# A queue that outputs 0,1,2,3,...
range_q = tf.train.range_input_producer(limit=5, shuffle=False)
slice_end = range_q.dequeue()

# Slice x to variable length, i.e. [0], [0, 1], [0, 1, 2], ....
y = tf.slice(x, [0], [slice_end], name="y")

batched_data = tf.train.batch(
    tensors=[y],
    batch_size=5,
    dynamic_pad=True,
    name="y_batch"
)

# Run the graph
# tf.contrib.learn takes care of starting the queues for us
res = tf.contrib.learn.run_n({"y": batched_data}, n=1, feed_dict=None)

# Print the result
print("Batch shape: {}".format(res[0]["y"].shape))
print(res[0]["y"])