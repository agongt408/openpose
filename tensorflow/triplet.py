import tensorflow as tf
import keras.backend as Keras

import cv2
import numpy as np
import json
from sklearn import metrics
from scipy.spatial import distance
import time

import sys
sys.path.append('/home/albert/github/DenseNet/')
import densenet

P_param = 4
K_param = 4
input_margin = 0.5
input_output_dim = 128
input_shape = (256,128)
input_overlay = False
input_preprocess = False

DATA_ROOT = '/home/albert/caffe/'

# Data generation

def imread(img_path):
    """
    returns RGB image
    misc.imread is deprecated
    """
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

def imread_scale(img_path, shape=input_shape, preprocess=input_preprocess):
    if preprocess:
        return densenet.preprocess_input(cv2.resize(imread(img_path),
                (shape[1], shape[0])).astype(np.float64))
    else:
        return cv2.resize(imread(img_path),(shape[1], shape[0])).astype(np.float64)

def makeGaussian(shape, fwhm=20, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.

    shape = shape of numpy array (y-axis, x-axis)
    """

    x = np.arange(0, shape[1], 1, float)
    y = np.arange(0, shape[0], 1, float)[:,np.newaxis]

    if center is None:
        x0 = y0 = np.array(shape).min() // 2
    else:
        x0 = center[0]
        y0 = center[1]

    result = np.exp(-4*np.log(2) * (np.power(x-x0,2) + np.power(y-y0,2)) / fwhm**2)[0:shape[0], 0:shape[1]]
    return result

def overlay_keypoints(img_path, shape=input_shape, train=False, preprocess=input_preprocess, fwhm=40):
    if img_path.find('/') > -1:
        root = img_path[len(img_path) - img_path[::-1].index('/'):-4:]
    else:
        root = img_path[0:img_path.index('.')]

    img = imread_scale(img_path, shape=shape, preprocess=preprocess)
    htmp = np.zeros(shape).astype(np.float64)

    if train:
        keypoint_path = DATA_ROOT + \
                        'data/market-1501/train_openpose/train_keypoints/%s_keypoints.json' % root
    else:
        keypoint_path = DATA_ROOT + \
                        'data/market-1501/test_openpose/test_keypoints/%s_keypoints.json' % root
    with open(keypoint_path) as data_file:
        data = json.load(data_file)
    for person in range(len(data['people'])):
        for i in range(0, len(data['people'][person]['pose_keypoints']), 3):
            x_key = data['people'][person]['pose_keypoints'][i] * shape[1] / 64.0
            y_key = data['people'][person]['pose_keypoints'][i + 1] * shape[0] / 128.0
            c_key = data['people'][person]['pose_keypoints'][i + 2]
            if not (x_key == 0 and y_key == 0):
                htmp = np.maximum(htmp, makeGaussian(shape, fwhm=fwhm, center=(x_key, y_key)))

    htmp = np.repeat(htmp, 3).reshape((shape[0], shape[1], 3))
    if htmp.max() > 0.0:
        htmp /= htmp.max()

    # Make sure type is float
    return img*htmp

def output_batch_generator(files_dict, labels, P=P_param, K=K_param, shape=input_shape,
                            preprocess=input_preprocess, overlay=input_overlay):
    print preprocess
    print overlay
    while True:
        batch = []
        idt_choice = np.random.choice(labels, P, replace=False)
        for p in range(len(idt_choice)):
            if K > len(files_dict[idt_choice[p]]):
                k_choice = np.random.choice(range(len(files_dict[idt_choice[p]])), K, replace=True)
            else:
                k_choice = np.random.choice(range(len(files_dict[idt_choice[p]])), K, replace=False)
            for k in k_choice:
                path = files_dict[idt_choice[p]][k]
                if overlay:
                    batch.append(overlay_keypoints(path, shape=shape, train=True,
                                    preprocess=preprocess).tolist())
                else:
                    batch.append(imread_scale(path, shape=shape, preprocess=preprocess).tolist())
        output = np.array(batch) # Make sure type is float
        yield(output, np.zeros((P*K, 128)))

# Triplet loss

def log1p(x):
    return Keras.log(1 + Keras.exp(x))

def dist(x1, x2):
    return Keras.sum(Keras.abs(x1 - x2), axis=1)

def triplet_loss(y_true, y_pred, margin=input_margin, P=P_param, K=K_param, output_dim=input_output_dim):
    embeddings = Keras.reshape(y_pred, (-1, output_dim))
    loss = tf.Variable(1, dtype=tf.float32)

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

# Evaluation

def evaluate_rank(model, files_dict, files_arr, rank=[1,5,20], train=True,
                    test_iter=1000, runs_per_rank=3, shape=input_shape,
                    overlay=input_overlay, preprocess=input_preprocess):
    print 'preprocess' , preprocess
    print 'overlay' , overlay
    print 'train' , train

    all_embeddings = []
    all_identities = []

    s = time.time()
    z = 0

    for idt in files_dict.keys():
        for f in files_dict[idt]:
            if overlay:
                img = overlay_keypoints(f, train=train, preprocess=preprocess, shape=shape)
            else:
                img = imread_scale(f, shape=shape, preprocess=preprocess)
            predict = model.predict(img.reshape(1, shape[0], shape[1], 3))
            all_embeddings.append(predict)
            all_identities.append(idt)
            z += 1
            if z % 1000 == 0:
                print z, time.time() - s

    rank_dict = {}
    for r in rank:
        rank_arr = []
        for x in range(runs_per_rank):
            correct = 0
            f_choice = np.random.choice(range(len(files_arr)),
                                        np.minimum(test_iter, len(files_arr)), replace=False)
            for f in f_choice:
                if overlay:
                    query_img = overlay_keypoints(files_arr[f][0], train=train, preprocess=preprocess, shape=shape)
                else:
                    query_img = imread_scale(files_arr[f][0], shape=shape, preprocess=preprocess)
                query_embedding = model.predict(query_img.reshape(1, shape[0], shape[1], 3))
                distance_vectors = np.squeeze(np.abs(all_embeddings - query_embedding))
                distance = np.sum(distance_vectors, axis=1)
                top_inds = distance.argsort()[:r+1]
                output_classes = np.array(all_identities)[top_inds].astype(np.uint16)

                if np.where(output_classes == int(files_arr[f][1]))[0].shape[0] > 1:
                    correct += 1

            rank_arr.append(float(correct)/test_iter)
        rank_dict[r] = np.mean(np.array(rank_arr))
    return rank_dict

def pos_pair_generator(files_dict):
    while True:
        idt = np.random.choice(files_dict.keys(), 1, replace=False)[0]
        sample_choice = np.random.choice(range(len(files_dict[idt])), np.minimum(2, len(files_dict[idt])),
                                        replace=False)
        pair = [files_dict[idt][p] for p in sample_choice]
        yield pair

def neg_pair_generator(files_dict):
    while True:
        idt_choice = np.random.choice(files_dict.keys(), 2, replace=False)
        pair = []
        for idt in idt_choice:
            sample = np.random.choice(range(len(files_dict[idt])), 1, replace=False)[0]
            pair.append(files_dict[idt][sample])
        yield pair

def evaluate_dist(model, generator, n_pairs=1000, train=False, shape=input_shape,
                    preprocess=input_preprocess, overlay=input_overlay):
    distances = []
    for t in range(n_pairs):
        pair = generator.next()
        predict = []
        for i in range(2):
            if overlay:
                img = overlay_keypoints(pair[i], shape=shape, train=train, preprocess=preprocess)
            else:
                img = imread_scale(pair[i], shape=shape, preprocess=preprocess)
            predict.append(model.predict(img.reshape(1,shape[0],shape[1],3)))
        dist = distance.cosine(predict[0], predict[1])
        distances.append(dist)
    return distances

def get_score(model, files_dict, files_arr, rank_iter=1000, runs_per_rank=3,
                hist=None, overlay=input_overlay, train=True,
                preprocess=input_preprocess, shape=input_shape):
    score = {
        'rank' : {},
        'pos_distance' : [],
        'neg_distance' : [],
        'precision' : [],
        'recall' : [],
        'matt_coef' : [],
        'loss' : []
    }

    score['rank'] = evaluate_rank(model, files_dict, files_arr, test_iter=rank_iter, rank=[1,5,20], train=train,
                                    runs_per_rank=runs_per_rank, shape=shape, overlay=overlay,
                                    preprocess=preprocess)

    pos_generator = pos_pair_generator(files_dict)
    neg_generator = neg_pair_generator(files_dict)

    pos_distance = evaluate_dist(model, pos_generator, train=train,
                                preprocess=preprocess, overlay=overlay, shape=shape)
    score['pos_distance'] = pos_distance

    neg_distance = evaluate_dist(model, neg_generator, train=train,
                                preprocess=preprocess, overlay=overlay, shape=shape)
    score['neg_distance'] = neg_distance

    for thresh in np.linspace(0, np.max(np.array(pos_distance + neg_distance)), 101)[1:]:
        true = np.ones(len(pos_distance)).tolist() + np.zeros(len(neg_distance)).tolist()
        pred_pos = [int(d < thresh) for d in pos_distance]
        pred_neg = [int(d < thresh) for d in neg_distance]
        pred = pred_pos + pred_neg

        score['precision'].append(metrics.precision_score(true, pred))
        score['recall'].append(metrics.recall_score(true, pred))
        score['matt_coef'].append(metrics.matthews_corrcoef(true, pred))

    try:
        score['loss'] += hist.history['loss']
    except AttributeError:
        pass

    return score
