import tensorflow as tf


# Encodes the question (a sequence of embedding vectors) to its representation
# as re-weighted sum of the input vectors.
# Input question vector : (batch size, sequence length, input_dims)
# Output question vector: (batch size, 2 * hidden_size)
class QuestionEncoder(tf.keras.Model):
    def __init__(self, input_dims, hidden_units, batch_size=None):
        super(QuestionEncoder, self).__init__()
        self.batch_size = batch_size
        self.layer1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units,
                                                                         return_sequences=True),
                                                    input_shape=(batch_size, input_dims))
        self.layer2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units, return_sequences=True))
        self.layer3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units, return_sequences=True))
        self.W = tf.keras.layers.Dense(1)

    # Encodes the question input sequence to the weighted vector q.
    # Input question vector : (batch size, sequence length, input_dims)
    # Output question vector: (batch size, 2 * hidden_size)
    def call(self, inputs):
        output1 = self.layer1(inputs)
        output2 = self.layer2(output1)
        output3 = self.layer3(output2)
        q_j = output3
        # (batch size, q sequence length, hidden units * 2)
        print("q_j: {}".format(q_j.shape))
        # w * q_j : (batch size, q sequence length, 1)
        w_qj = self.W(q_j)
        print("W q_j shape: {}".format(w_qj.shape))
        # Coefficients b_j : (batch_size, q sequence length, 1)
        b_j = tf.nn.softmax(w_qj, axis=1)
        # Weighted terms : (batch size, q sequence length, hidden size * 2)
        bj_qj = b_j * q_j
        print("bj_qj : {}".format(bj_qj.shape))
        # Weighted sums : (batch_size, hidden size * 2)
        weighted_output = tf.reduce_sum(bj_qj, axis=1)
        print("weighted_output : {}".format(weighted_output.shape))
        return weighted_output


# Full network implementation. TODO: Find a different name for this and maybe modularise more.
# Input :
#  - Paragraph as a sequence of word IDs : (batch size, p sequence length)
#  - Question as a sequence of word IDs : (batch size, q sequence length)
# Output :
#  - index_start * (p sequence length) + index_end
class FullNetwork(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units, batch_size):
        super(FullNetwork, self).__init__()
        self.batch_size = batch_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.layer1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units,
                                                                         return_sequences=True),
                                                    input_shape=(batch_size, embedding_dim))
        self.layer2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units, return_sequences=True))
        self.layer3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units, return_sequences=True))
        self.question_encoder = QuestionEncoder(
            input_dims=embedding_dim,
            hidden_units=hidden_units,
            batch_size=batch_size)
        self.alpha = tf.keras.layers.Dense(1, activation=tf.nn.relu)
        self.p_feature_vec_total_length = embedding_dim * 2
        self.W_s = tf.keras.layers.Dense(self.p_feature_vec_total_length)
        self.W_e = tf.keras.layers.Dense(self.p_feature_vec_total_length)

    # Input :
    #  - Paragraph as a sequence of word IDs : (batch size, p sequence length)
    #  - Question as a sequence of word IDs : (batch size, q sequence length)
    # Output :
    #  - index_start * (p sequence length) + index_end
    def call(self, paragraph_inputs, question_inputs):
        q_embedded_inputs = self.embedding(question_inputs)
        print("q_embedded_inputs : {}".format(q_embedded_inputs.shape))
        p_embedded_inputs = self.embedding(paragraph_inputs)
        print("Embedded inputs : {}".format(p_embedded_inputs.shape))
        output1 = self.layer1(p_embedded_inputs)
        output2 = self.layer2(output1)
        output3 = self.layer3(output2)
        p_j = output3
        # p_j : (batch size, p sequence length, hidden units * 2)
        print("p_j : {}".format(p_j.shape))
        # a(E(p)) : (batch size, p sequence length, embedding dims)
        a_e_p = tf.expand_dims(tf.squeeze(self.alpha(p_embedded_inputs)), 2)
        print("a(E(p)) shape: {}".format(a_e_p.shape))
        # a(E(q)) : (batch size, q sequence length, embedding dims)
        a_e_q = tf.expand_dims(tf.squeeze(self.alpha(q_embedded_inputs)), 1)
        print("a(E(q)) shape: {}".format(a_e_q.shape))
        score = a_e_p * a_e_q
        print("score : {}".format(score.shape))
        a_ij = tf.nn.softmax(score, axis=1)
        print("a_ij : {}".format(a_ij.shape))
        # Weighted sums : (batch_size, hidden size)
        f_align = tf.matmul(a_ij, q_embedded_inputs)
        print("f_align : {}".format(f_align.shape))

        # Concatenate the feature vectors.
        # p_feature_vec : (batch size, p sequence length, 2 * embedding dim)
        # TODO: Add more feature vectors for the inputs (and update p_feature_vec_total_length).
        p_feature_vec = tf.concat([p_embedded_inputs, f_align], 2)
        print("paragraph feature vector : {}".format(p_feature_vec.shape))
        # p_end : (batch size, p sequence length, 2 * hidden units)
        q_feature_vec = tf.expand_dims(self.question_encoder(q_embedded_inputs), 1)
        print("question feature vector : {}".format(q_feature_vec.shape))
        # p_start : (batch size, p sequence length)
        p_start = tf.matmul(p_feature_vec, tf.expand_dims(tf.squeeze(self.W_s(q_feature_vec)), 2))
        print("p_start : {}".format(p_start.shape))
        # p_end : (batch size, p sequence length)
        p_end = tf.matmul(p_feature_vec, tf.expand_dims(tf.squeeze(self.W_e(q_feature_vec)), 2))
        print("p_end : {}".format(p_end.shape))

        # p_start(i) = exp(p_j W_s q) : (batch size, p sequence length, 1)
        soft_p_start = tf.nn.softmax(p_start, axis=1)
        print("soft_p_start : {}".format(soft_p_start))
        # p_end(i) = exp(p_j W_e q) : (batch size, p sequence length, 1)
        soft_p_end = tf.nn.softmax(p_end, axis=1)
        print("soft_p_end : {}".format(soft_p_end))

        # Change p_end to (batch size, 1, p sequence length)
        # This makes the matrix multiplication feasible.
        # With dimensions (batch size, p sequence length, p sequence length)
        p_start_end = tf.matmul(soft_p_start, tf.expand_dims(tf.squeeze(p_end), 1))
        print("p_start_end : {}".format(p_start_end.shape))

        # Flattens p_start_end to (batch size, (p sequence length) * (p sequence length))
        # and return the maximum index (this should be interpreted as index_start * (p sequence length) + index_end ).
        p_start_end_flattened = tf.keras.layers.Flatten()(p_start_end)
        print("p_start_end_flattened : {}".format(p_start_end_flattened.shape))
        ans = tf.arg_max(p_start_end_flattened, 1)
        print("ans : {}".format(ans.shape))
        return ans


# The following code is used just for testing the dimensions of input and output.
# encoder = QuestionEncoder(input_dims=100, hidden_units=300, batch_size=None)
# max_sequence_length = 16
# encoder(tf.keras.layers.Embedding(100, 200)(tf.zeros(shape=(400, max_sequence_length))))
p_encoder = FullNetwork(vocab_size=100, embedding_dim=200, hidden_units=300, batch_size=None)
max_p_length = 42
max_q_length = 16
paragraph = tf.zeros(shape=(500, max_p_length))
question = tf.zeros(shape=(500, max_q_length))
p_encoder(paragraph, question)
