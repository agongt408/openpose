def L(a,b, e=2):
    return K.exp(-(a**2 + b**2)/(2*e**2))

# https://github.com/tensorflow/tensorflow/issues/6503
def svd(A, full_matrices=False, compute_uv=True, name=None):
  # since dA = dUSVt + UdSVt + USdVt
  # we can simply recompute each matrix using A = USVt
  # while blocking gradients to the original op.
  _, M, N = A.get_shape().as_list()
  P = min(M, N)
  S0, U0, V0 = map(tf.stop_gradient, tf.svd(A, full_matrices=True, name=name))
  Ui, Vti = map(tf.matrix_inverse, [U0, tf.transpose(V0, (0, 2, 1))])
  # A = USVt
  # S = UiAVti
  S = tf.matmul(Ui, tf.matmul(A, Vti))
  S = tf.matrix_diag_part(S)
  if not compute_uv:
    return S
  Si = tf.pad(tf.matrix_diag(1/S0), [[0,0], [0,N-P], [0,M-P]])
  # U = AVtiSi
  U = tf.matmul(A, tf.matmul(Vti, Si))
  U = U if full_matrices else U[:, :M, :P]
  # Vt = SiUiA
  V = tf.transpose(tf.matmul(Si, tf.matmul(Ui, A)), (0, 2, 1))
  V = V if full_matrices else V[:, :N, :P]
  return S, U, V

def_cam = """

def cam(pool, cam_dim=(8,4), P=6, K=4, reduce_dim=%r, dim=%d):
    batch_size = P * K
    height = cam_dim[0]
    width = cam_dim[1]

    if reduce_dim:
        s, u, _ = svd(tf.reshape(pool, (batch_size,height*width, pool.get_shape()[3].value)))
        pool = tf.reshape(tf.matmul(u[:, :dim], tf.matrix_diag(s[:, :dim])), (batch_size, height, width, -1))
        print pool.shape

    channels = pool.get_shape()[3].value

    norms = []
    for b in range(batch_size):
        for i in range(height):
            for j in range(width):
                norms.append(triplet.norm(pool[b][i][j], tf.reshape(pool[b], (height*width, channels)), axis=1))
#     triplet.norm(tf.reshape(pool, (-1, height * width, 1, channels)),
#                          tf.tile(tf.reshape(pool,(-1, 1, height * width, channels)),
#                                  [1, height * width, 1, 1]), axis=3)

    L_x = tf.reshape(tf.tile(tf.range(0, width, delta=1, dtype=tf.float32), [height]), (1,1, height*width))
    L_x_0 = tf.tile(tf.reshape(L_x, (1, height * width, 1)), [batch_size,1,1])
    L_x_1 = tf.tile(L_x, [batch_size, height * width, 1])

    L_y = tf.reshape(tf.tile(tf.reshape(tf.range(0, height, delta=1, dtype=tf.float32),
                                        (1,height,1)), [1,1,width]), (1,height*width,1))
    L_y_0 = tf.tile(L_y, [batch_size,1, 1])
    L_y_1 = tf.tile(tf.reshape(L_y, (1,1,height*width)), [batch_size, height * width, 1])

    L_maps = L(L_x_0 - L_x_1, L_y_0-L_y_1)

    D = tf.multiply(tf.reshape(tf.stack(norms), (batch_size, height*width, height*width)), L_maps)
    D = tf.divide(D, tf.tile(tf.reshape(tf.reduce_sum(D, axis=2), (-1, height * width, 1)), [1,1, height * width]))

    D = tf.transpose(D, (0,2,1))
    M = tf.convert_to_tensor(np.ones((batch_size, height * width,), dtype=np.float32) / (height * width))
    M = tf.expand_dims(M,2)

    for i in range(1):
         M = tf.matmul(D, M)

    return tf.reshape(M, (batch_size, height, width))

"""

def_loss = """

def triplet_loss(y_true, y_pred, margin=1.0, P_param=%d, K_param=%d, output_dim=%d):
    embeddings = K.reshape(y_pred, (-1, output_dim))
    loss = K.variable(0, dtype='float32')

    for i in range(P_param):
        for a in range(K_param):
            pred_anchor = embeddings[i*K_param + a]
            hard_pos = K.max(triplet.dist(pred_anchor, embeddings[i*K_param:(i + 1)*K_param]))
            hard_neg = K.min(triplet.dist(pred_anchor, K.concatenate([embeddings[0:i*K_param],
                                                                    embeddings[(i + 1)*K_param:]], 0)))
            if margin == 'soft':
                loss += triplet.log1p(hard_pos - hard_neg)
            else:
                loss += K.maximum(margin + hard_pos - hard_neg, 0.0)
    return loss

def cam_loss(y_true, y_pred):
    # return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return K.sum(triplet.norm(y_true, y_pred, axis=(1,2), norm=1))
    # return tf.losses.hinge_loss(labels=y_true, logits=y_pred)
    # return K.sum(K.flatten(tf.multiply(y_true, y_pred)))
    # return y_pred

"""

def SPN(P_param, K_param, weights=None, shape=(256,128), cam_dim=(8,4), reduce_dim=True, dim=32):
    exec(def_cam % (reduce_dim, dim))
    exec(def_loss % (P_param, K_param, 128))

    if weights is not 'imagenet':
        trinet = DenseNetImageNet121(input_shape=(shape[0],shape[1],3), weights=None)
    else:
        trinet = DenseNetImageNet121(input_shape=(shape[0],shape[1],3), weights=weights)
    cam_output = Lambda(cam, name='cam_output', arguments={'P' : P_param, 'K' : K_param})(trinet.layers[-6].output)
    camnet = Model(inputs=trinet.input, outputs=[trinet.output, cam_output])
    camnet.compile(loss=[triplet_loss, cam_loss], loss_weights=[1.0, float(P_param*K_param)],
               optimizer=Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))
    if weights is not None and weights is not 'imagenet':
        camnet.set_weights(np.load(weights))
    return camnet
