import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

nodes_hl1 = 500
nodes_hl2 = 500
nodes_hl3 = 500

n_classes = 10
batch_size = 100

# height X width

x = tf.placeholder('float')
y = tf.placeholder('float')




def neural_network_model(data):
	hidden_layer1 = { 'weights': tf.Variable(tf.random_normal([784, nodes_hl1])),
					  'biases': tf.Variable(tf.random_normal([nodes_hl1])) }

	hidden_layer2 = { 'weights': tf.Variable(tf.random_normal([nodes_hl1, nodes_hl2])),
					  'biases': tf.Variable(tf.random_normal([nodes_hl2])) }

	hidden_layer3 = { 'weights': tf.Variable(tf.random_normal([nodes_hl2, nodes_hl3])),
					  'biases': tf.Variable(tf.random_normal([nodes_hl3])) }

	output_layer = { 'weights': tf.Variable(tf.random_normal([nodes_hl3, n_classes])),
					  'biases': tf.Variable(tf.random_normal([n_classes])) }


	l1 = tf.add(tf.matmul(data, hidden_layer1['weights']), hidden_layer1['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_layer2['weights']), hidden_layer2['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_layer3['weights']), hidden_layer3['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

	return output



def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y) )	
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	epochs = 10

	with tf.Session() as sess:
		#sess.run(tf.initialize_all_variables())
		tf.global_variables_initializer().run()
		
		for epoch in range(epochs):
			epoch_loss = 0
			for i in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				i, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss +=c
			print('Epoch', epoch, 'completed out of 10  loss:', epoch_loss)
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)



























