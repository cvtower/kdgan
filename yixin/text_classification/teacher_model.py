import tensorflow as tf

class Teacher():
	def __init__(self, wordsNum, tagNum, emb_dim, sample_num, lamda, max_length, param=None, initdelta=0.05, learning_rate=0.01):
		self.wordsNum = wordsNum
		self.tagNum = tagNum
		self.emb_dim = emb_dim
		self.sample_num = sample_num
		self.lamda = lamda  # regularization parameters
		self.param = param
		self.initdelta = initdelta
		self.learning_rate = learning_rate
		self.t_params = []

		with tf.variable_scope('teacher'):
			if self.param == None:
				self.word_embeddings = tf.Variable(
					tf.random_uniform([self.wordsNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
										dtype=tf.float32))
			else:
				#self.word_embeddings = tf.Variable(self.param)
				self.word_embeddings = self.param
				

		# sents are inputted as one-hot form
		self.sents = tf.placeholder(tf.int32, shape=(None, max_length),name='sents')
		self.seqlen = tf.placeholder(tf.int32, shape=[None], name='seqLen') # record the length of each sentence
		self.samples = tf.placeholder(tf.int32, shape=(None, self.sample_num), name='samples')
		self.reward = tf.placeholder(tf.float32, shape=(None, self.sample_num), name = 'rewward')
		# for test
		self.true_tag = tf.placeholder(tf.float32, shape=(None, tagNum))

		self.sent_embedding = tf.nn.embedding_lookup(self.word_embeddings, self.sents)
		#print("sent_embedding*******************:", self.sent_embedding)
		# tag_prob has the form (None, tagNum), where None is the batch_size
		self.tag_prob = self.LSTM()
		#self.tag_prob = self.simpleNN()

		# for test
		self.test_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.tag_prob, labels=self.true_tag)
		#self.test_loss = tf.nn.l2_loss(self.tag_prob - self.true_tag)
		
		self.test_opt = tf.train.GradientDescentOptimizer(self.learning_rate) 
			

		#self.teacher_loss = -tf.reduce_mean(tf.log(self.tag_prob) * self.reward) + self.lamda * (
			#tf.nn.l2_loss(self.sent_embedding))


		t_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
		with tf.variable_scope("teacher") as vs:
			self.t_params = [v for v in tf.global_variables() if v.name.startswith(vs.name)]
			#self.teacher_updates = t_opt.minimize(self.teacher_loss, var_list=self.t_params)
			self.test_updates = self.test_opt.minimize(self.test_loss)


	def LSTM(self):
		print('using LSTM...')
		with tf.variable_scope('teacher'):
			cell = tf.nn.rnn_cell.GRUCell(self.emb_dim)
			cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
			#cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2, state_is_tuple=False)
			batch_size = tf.shape(self.sents)[0]
			inital_state = cell.zero_state(batch_size, tf.float32)
			#inital_state = tf.identity(inital_state, name="initial_state")
			
			rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, self.sent_embedding, sequence_length=self.seqlen,
							initial_state=inital_state)
			#print("LSTM********************************:", final_state)
			logits = tf.contrib.layers.fully_connected(final_state, self.tagNum, activation_fn = None,
                                              weights_initializer = tf.truncated_normal_initializer(mean = 0, stddev = 0.001),
                                              biases_initializer = tf.zeros_initializer())  

		#print("LSTM********************************:", final_state)	
		#prob = tf.nn.softmax(logits)	
		return logits

	def simpleNN(self):
		print("using simpleNN...")
		with tf.variable_scope('teacher'):
			#print(self.sent_embedding)
			#print(self.seqlen)
			sum_sents = tf.reduce_sum(self.sent_embedding, 1)
			d_seqlen = tf.tile(tf.expand_dims(self.seqlen, 1), [1,self.emb_dim])
			#print(sum_sents)
			
			#ave_sents = tf.div(sum_sents, tf.cast(d_seqlen, tf.float32))
			ave_sents = sum_sents
			layer1 = tf.contrib.layers.fully_connected(ave_sents, 1000,
                                              weights_initializer = tf.truncated_normal_initializer(mean = 0, stddev = 0.001),
                                              biases_initializer = tf.zeros_initializer()) 
			layer2 = tf.contrib.layers.fully_connected(layer1, 700,
                                              weights_initializer = tf.truncated_normal_initializer(mean = 0, stddev = 0.001),
                                              biases_initializer = tf.zeros_initializer()) 
			layer3 = tf.contrib.layers.fully_connected(layer2, 500,
                                              weights_initializer = tf.truncated_normal_initializer(mean = 0, stddev = 0.001),
                                              biases_initializer = tf.zeros_initializer()) 
			logits = tf.contrib.layers.fully_connected(layer3, self.tagNum, activation_fn = None,
                                              weights_initializer = tf.truncated_normal_initializer(mean = 0, stddev = 0.001),
                                              biases_initializer = tf.zeros_initializer())
			#prob =tf.nn.softmax(logits)
		return logits 

	def save_model(self, sess, filename):
		param = sess.run(self.g_params)
		cPickle.dump(param, open(filename, 'w'))