import keras.backend as Keras
import tensorflow as tf

P_param = 4
K_param = 4
input_output_dim = 128
input_margin = 0.5

def log1p(x):
    return Keras.log(1 + Keras.exp(x))

def dist(x1, x2, axis=1, norm=1):
    return Keras.sum(Keras.abs(x1 - x2), axis=axis)

def norm(x1, x2, axis=1, norm=1):
    return Keras.pow(Keras.sum(Keras.pow(Keras.abs(x1 - x2), norm), axis=axis), 1.0 / norm)

def triplet_loss(y_true, y_pred, margin=input_margin, P=P_param, K=K_param, output_dim=input_output_dim):
    embeddings = Keras.reshape(y_pred, (-1, output_dim))
    loss = Keras.variable(0, dtype='float32')

    for i in range(P):
        for a in range(K):
            pred_anchor = embeddings[i*K + a]
            hard_pos = Keras.max(dist(pred_anchor, embeddings[i*K:(i + 1)*K]))
            hard_neg = Keras.min(dist(pred_anchor, Keras.concatenate([embeddings[0:i*K],
                                                                    embeddings[(i + 1)*K:]], 0)))
            if margin == None:
                loss += log1p(hard_pos - hard_neg)
            else:
                loss += Keras.maximum(margin + hard_pos - hard_neg, 0.0)
    return loss

def cam_loss(y_true, y_pred):
#     return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
#     return Keras.sum(triplet.norm(y_true, y_pred, axis=(1,2), norm=1))
#     return tf.losses.hinge_loss(labels=y_true, logits=y_pred)
    return Keras.sum(Keras.flatten(tf.multiply(y_true, y_pred)))
