# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from reco_utils.recommender.deeprec.models.sequential.sli_rec import (
    SLI_RECModel,
)
from tensorflow.nn import dynamic_rnn
from reco_utils.recommender.deeprec.deeprec_utils import load_dict
from .utils import sparse_matrix_to_tensor, sparse_dropout, matrix_to_tensor
sparse_dot = tf.sparse_tensor_dense_matmul
import scipy.sparse as sp
# from scipy import sparse

import numpy as np
__all__ = ["PPRModel"]


class PPRModel(SLI_RECModel):

    def _build_seq_graph(self):
        """The main function to create din model.
        
        Returns:
            obj:the output of din section.
        """
        with tf.name_scope('diffu'):
            hist_input = self.item_history_embedding
            hist_input = hist_input + self.position_embedding
            self.hidden_units = self.item_embedding_dim
            self.mask = self.iterator.mask
            # w = 5
            import numpy as np
            w = 5
            LEN = 257
            self.window_mask = np.zeros((LEN,LEN))


            alp = 0.1
            self.dropout_rate = 0.0
            self.sequence_length = tf.reduce_sum(self.mask, 1)

            self.num_blocks = 2
            self.num_heads = 1
            self.is_training = True
            #  self.recent_k = 5
            self.recent_k = 1

            for i in range(LEN - 1):
                # for j in range(i-w, i+w+1):
                    # if (j == LEN - 1 or i == LEN - 1) or (i-w<=j and j<=i+w):
                low_bound = max(i-w, 0)
                upper_bound = min(i+w+1, LEN)
                alp_low = max(0-(i-w), 0)
                alp_upper = min(2*w+1, LEN+w-i)
                self.window_mask[i,low_bound:upper_bound] = (1) 
                # import pdb 
                # pdb.set_trace()
                # self.window_mask[i,low_bound:upper_bound] = (diff_alp[alp_low:alp_upper])
                if self.hparams.is_random:
                    rand_val = np.random.randint(0, upper, size=2)
                    for r_val in rand_val:
                        if r_val >= low_bound and r_val <= upper_bound:
                            r_val = r_val + 2 * w + 1
                        self.window_mask[i,r_val] = 1
                # self.window_mask[i,low_bound:upper_bound].assign(1)
            for i in range(LEN):
                self.window_mask[i,LEN - 1] = 1
                self.window_mask[LEN - 1,i] = 1
            self.window_mask_tensor = tf.cast(tf.convert_to_tensor(self.window_mask), tf.float32)

            self.real_mask = tf.cast(self.mask, tf.float32)
            # self.hist_embedding_sum = tf.reduce_sum(hist_input*tf.expand_dims(self.real_mask, -1), 1)
            hist_input = hist_input*tf.expand_dims(self.real_mask, -1)
            self.seq = tf.concat([hist_input, tf.expand_dims(self.target_item_embedding, 1)], axis=1)
            # self.seq, key_masks = self.multihead_attention(his_tar_input, his_tar_input, num_units=self.hidden_units)

            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):

                    # Self-attention
                    self.seq = self.multihead_attention(queries=self.normalize(self.seq),
                                                    keys=self.seq,
                                                    num_units=self.hidden_units,
                                                    num_heads=self.num_heads,
                                                    causality=True,
                                                    #  causality=False,
                                                    scope="self_attention")


                    # Feed forward
                    self.seq = self.feedforward(self.normalize(self.seq), num_units=[self.hidden_units, self.hidden_units],
                                           dropout_rate=self.dropout_rate, is_training=self.is_training)
                    self.seq = tf.concat([self.seq[:, :-1, :] * tf.expand_dims(self.real_mask, -1), tf.expand_dims(self.seq[:, -1, :], 1)], axis=1)

            self.seq = self.normalize(self.seq)
            # att_fea = tf.reduce_sum(attention_output, 1)
            # tf.summary.histogram('att_fea', att_fea)
            self.hist_embedding_mean = (tf.reduce_sum(self.seq[:,:-1,:]*tf.expand_dims(self.real_mask, -1), 1) + self.seq[:,-1,:])/ (tf.reduce_sum(self.real_mask, 1, keepdims=True) + 1)
            # self.hist_embedding_mean = (self.seq[:,-1,:] + self.hist_embedding_mean)/2
        model_output = tf.concat([self.target_item_embedding, self.hist_embedding_mean], -1)
        # tf.summary.histogram("model_output", model_output)
        return model_output
        
    def multihead_attention(self,
                            queries, 
                            keys, 
                            num_units=None, 
                            num_heads=1, 
                            causality=True,
                            scope="multihead_attention", 
                            reuse=None):
        '''Applies multihead attention.
        
        Args:
        queries: A 3d tensor with shape of [N, T_q, C_q].
        keys: A 3d tensor with shape of [N, T_k, C_k].
        num_units: A scalar. Attention size.
        num_heads: An int. Number of heads.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
            
        Returns
        A 3d tensor with shape of (N, T_q, C)  
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]
            
            # Linear projections
            # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
            # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
            # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
            Q = tf.layers.dense(queries, num_units, activation=None) # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
            
            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
            
            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            
            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
            # w = 5
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
            # import pdb
            # pdb.set_trace()
            window_mask_tensor = tf.tile(tf.expand_dims(self.window_mask_tensor, 0), [tf.shape(key_masks)[0], 1, 1])
                    # if :
            self.initial_key_mask = key_masks
            key_masks = tf.minimum(key_masks, window_mask_tensor)
            self.combine_mask = key_masks
            paddings = tf.ones_like(outputs)*(-2**32+1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

            # Activation
            if causality:
                # 构建下三角为1的tensor
                diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
       
                paddings = tf.ones_like(masks)*(-2**32+1)
                # 下三角置为无穷小负值（原因是下一步要进行softmax）
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
       
            outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
            
            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
            self.query_mask = query_masks
            query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
            # self.query_mask = 
            outputs *= query_masks # broadcasting. (N, T_q, C)
            
            # Dropouts
            # outputs = tf.nn.dropout(outputs, self.keep_prob)
            # attn_outputs = tf.matmul(outputs, V_)
            ppriter = PPRPowerIteration(outputs, 0.1, 2)
            outputs = ppriter.build_model(V_, 1.0)
            # Weighted sum
             # ( h*N, T_q, C/h)
            
            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
                
            # Residual connection
            outputs += queries
                
            # Normalize
            #outputs = normalize(outputs) # (N, T_q, C)
    
        return outputs
    def feedforward(self, inputs, 
                    num_units=[2048, 512],
                    scope="multihead_attention", 
                    dropout_rate=0.2,
                    is_training=True,
                    reuse=None):
        '''Point-wise feed forward net.
        
        Args:
          inputs: A 3d tensor with shape of [N, T, C].
          num_units: A list of two integers.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
            
        Returns:
          A 3d tensor with the same shape and dtype as inputs
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            #  outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            #  outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
            
            # Residual connection
            outputs += inputs
            
            # Normalize
            #outputs = normalize(outputs)
        
        return outputs
    def normalize(self, inputs, 
                  epsilon = 1e-8,
                  scope="ln",
                  reuse=None):
        '''Applies layer normalization.
        
        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
          
        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
        
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta= tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
            outputs = gamma * normalized + beta
            
        return outputs


class PPRPowerIteration:
    def __init__(self, adj_matrix: tf.Tensor, alpha: float, niter: int):
        self.alpha = alpha
        self.niter = niter
        # adj_matrix = sp.csr_matrix(adj_matrix)
        M = adj_matrix
        self.A_hat = (1 - alpha) * M

    def build_model(self, Z: tf.Tensor, keep_prob: float) -> tf.Tensor:
        with tf.variable_scope(f'Propagation'):
            A_hat_tf = (self.A_hat)
            Zs_prop = Z
            for _ in range(self.niter):
                # A_drop = sparse_dropout(A_hat_tf, keep_prob)
                # import pdb
                # pdb.set_trace()
                Zs_prop = tf.matmul(A_hat_tf, Zs_prop) + self.alpha * Z
                # Zs_prop = sparse_dot(tf.tile(tf.expand_dims(A_drop, 0), [Zs_prop.shape[0], 1, 1]), Zs_prop) + self.alpha * Z
            return Zs_prop

