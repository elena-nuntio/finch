import tensorflow as tf
import paddle.v2 as paddle


def model_fn(features, labels, mode, params):
    predicted_score = forward(features)
    loss_op = tf.reduce_mean(tf.squared_difference(predicted_score, tf.to_float(labels)))

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss_op,
            eval_metric_ops={'mse': tf.metrics.mean_squared_error(tf.to_float(labels), predicted_score)})
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer(params['lr']).minimize(loss_op, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_op, train_op=train_op)


def forward(features):
    with tf.variable_scope('user_id'):
        user_id_embed = tf.contrib.layers.embed_sequence(
            ids = features['user_id'],
            vocab_size = paddle.dataset.movielens.max_user_id()+1,
            embed_dim = 32)
        user_id_fc = tf.layers.dense(user_id_embed, 32)

    with tf.variable_scope('gender_id'):
        gender_id_embed = tf.contrib.layers.embed_sequence(
            ids = features['gender_id'],
            vocab_size = 2,
            embed_dim = 16)
        gender_id_fc = tf.layers.dense(gender_id_embed, 16)
    
    with tf.variable_scope('age_id'):
        age_id_embed = tf.contrib.layers.embed_sequence(
            ids = features['age_id'],
            vocab_size = len(paddle.dataset.movielens.age_table),
            embed_dim = 16)
        age_id_fc = tf.layers.dense(age_id_embed, 16)

    with tf.variable_scope('job_id'):
        job_id_embed = tf.contrib.layers.embed_sequence(
            ids = features['job_id'],
            vocab_size = paddle.dataset.movielens.max_job_id()+1,
            embed_dim = 16)
        job_id_fc = tf.layers.dense(job_id_embed, 16)

    user_feats = tf.concat([user_id_fc, gender_id_fc, age_id_fc, job_id_fc], -1)
    user_feats = tf.layers.dense(user_feats, 200, tf.tanh)

    with tf.variable_scope('movie_id'):
        movie_id_embed = tf.contrib.layers.embed_sequence(
            ids = features['movie_id'],
            vocab_size = paddle.dataset.movielens.max_movie_id()+1,
            embed_dim = 32)
        movie_id_fc = tf.layers.dense(movie_id_embed, 32)

    with tf.variable_scope('category_ids'):
        category_fc = tf.layers.dense(tf.to_float(features['category_ids']), 32)

    with tf.variable_scope('movie_title'):
        movie_title_embed = tf.contrib.layers.embed_sequence(
            ids = features['movie_title'],
            vocab_size = 5175,
            embed_dim = 32)
        movie_title_conv = tf.layers.conv1d(movie_title_embed, 32, 3)
        movie_title_fc = global_max_pooling(movie_title_conv)
    
    movie_feats = tf.concat([movie_id_fc, category_fc, movie_title_fc], -1)
    movie_feats = tf.layers.dense(movie_feats, 200, tf.tanh)

    user_norm = tf.nn.l2_normalize(user_feats, -1)
    movie_norm = tf.nn.l2_normalize(movie_feats, -1)
    cos_sim = tf.reduce_sum(tf.multiply(user_norm, movie_norm), -1)
    return 5 * cos_sim


def global_max_pooling(x):
    batch_size = tf.shape(x)[0]
    num_units = x.get_shape().as_list()[-1]
    x = tf.layers.max_pooling1d(x, x.get_shape().as_list()[1], 1)
    x = tf.reshape(x, [batch_size, num_units])
    return x
