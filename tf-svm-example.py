import tensorflow as tf

batch_size = 10
num_features = 123
num_examples = 100

def input_fn():
  example_ids = tf.random_uniform(
      [batch_size], maxval=num_examples, dtype=tf.int64)
  # Construct a SparseTensor with features
  dense_features = (example_ids[:, None]
                    + tf.range(num_features, dtype=tf.int64)[None, :]) % 2
  non_zeros = tf.where(tf.not_equal(dense_features, 0))
  sparse_features = tf.SparseTensor(
      indices=non_zeros,
      values=tf.gather_nd(dense_features, non_zeros),
      dense_shape=[batch_size, num_features])
  features = {
      'some_sparse_features': tf.sparse_tensor_to_dense(sparse_features),
      'example_id': tf.as_string(example_ids)}
  labels = tf.equal(dense_features[:, 0], 1)
  return features, labels
svm = tf.contrib.learn.SVM(
    example_id_column='example_id',
    feature_columns=[
      tf.contrib.layers.real_valued_column(
          'some_sparse_features')],
    l2_regularization=0.1, l1_regularization=0.5)
svm.fit(input_fn=input_fn, steps=1000)
positive_example = lambda: {
    'some_sparse_features': tf.SparseTensor([[0, 0]], [1], [1, num_features]),
    'example_id': ['a']}
print(svm.evaluate(input_fn=input_fn, steps=20))
print(next(svm.predict(input_fn=positive_example)))
negative_example = lambda: {
    'some_sparse_features': tf.SparseTensor([[0, 0]], [0], [1, num_features]),
    'example_id': ['b']}
print(next(svm.predict(input_fn=negative_example)))

#outputs:
#{'loss': 1.0728836e-06, 'accuracy': 1.0, 'global_step': 1000}
#{'logits': array([ 0.01612902], dtype=float32), 'classes': 1}
#{'logits': array([ 0.], dtype=float32), 'classes': 0}
