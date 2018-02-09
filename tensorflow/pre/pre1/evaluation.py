import numpy as np
# from sklearn import metrics
# from scipy.spatial import distance
import time
import matplotlib.pyplot as plt

from keras.models import model_from_json

import os
import sys
sys.path.append('/home/albert/github/tensorflow/src/')
import data

input_shape = (256,128)
input_preprocess = True

def get_data(split, DATA_ROOT='/home/albert/github/tensorflow/'):
    files_dict = {}
    files_arr = []

    if split == 'query':
        img_dir = os.listdir(DATA_ROOT + 'data/market-1501/query')
    elif split == 'train':
        img_dir = os.listdir(DATA_ROOT + 'data/market-1501/bounding_box_train')
    elif split == 'test':
        img_dir = os.listdir(DATA_ROOT + 'data/market-1501/bounding_box_test')
    else:
        raise ValueError, 'split must be either query, train, or test'

    for f in img_dir:
        if f[-4:] == '.jpg':
            idt = int(f[0:f.index('_')])
            if idt != 0 and idt != -1:
                if not any(idt == l for l in files_dict.keys()):
                    files_dict[idt] = {}

                if split == 'query':
                    path = DATA_ROOT + 'data/market-1501/query/' + f
                elif split == 'train':
                    path = DATA_ROOT + 'data/market-1501/bounding_box_train/' + f
                elif split == 'test':
                    path = DATA_ROOT + 'data/market-1501/bounding_box_test/' + f

                camera = f[f.index('_') + 2 : f.index('_') + 3]
                files_arr.append([path, idt, int(camera)])
                try:
                    files_dict[idt][int(camera)].append(path)
                except:
                    files_dict[idt][int(camera)] = []
                    files_dict[idt][int(camera)].append(path)

    return files_dict, files_arr

def evaluate_rank(model, overlay, rank=[1,5,20],
                    shape=input_shape,
                    preprocess=input_preprocess,
                    DATA_ROOT='/home/albert/github/tensorflow/'):
    print 'preprocess' , preprocess
    print 'overlay' , overlay

    _, query_files = get_data('query', DATA_ROOT)
    test_dict, test_files = get_data('test', DATA_ROOT)

    s = time.time()
    z = 0

    all_embeddings = []
    all_identities = []
    all_camera = []

    for f, idt, camera in test_files:
        img = data.imread_scale(f, shape, preprocess)

        #if overlay:
        input_dict = {}
        for layer in model.layers:
            if layer.name.find('input_im') > -1:
                input_dict[layer.name] = img.reshape(1,shape[0],shape[1],3)
            elif layer.name.find('input_cam') > -1:
                input_dict[layer.name] = np.ones((1,128,64))
        # print input_dict
        predict = model.predict(input_dict)
        #else:
        #    predict = model.predict(img.reshape(1,shape[0],shape[1],3))

        all_embeddings.append(predict.tolist())
        all_identities.append(idt)
        all_camera.append(camera)

        z += 1
        if z % 1000 == 0:
            print z, time.time() - s

    all_embeddings = np.array(all_embeddings)
    all_identities = np.array(all_identities)
    all_camera = np.array(all_camera)

    rank_dict = {}
    for r in rank:
        rank_arr = []

        correct = 0
        test_iter = 0

        for f, idt, camera in query_files:
            b = np.logical_or(all_camera != camera, all_identities != idt)

            if len(test_dict[idt].keys()) > 1:
                query_img = data.imread_scale(f, shape=shape, preprocess=preprocess)
                #if overlay:
                input_dict = {}
                for layer in model.layers:
                    if layer.name.find('input_im') > -1:
                        input_dict[layer.name] = query_img.reshape(1,shape[0],shape[1],3)
                    elif layer.name.find('input_cam') > -1:
                        input_dict[layer.name] = np.ones((1,128,64))
                query_embedding = model.predict(input_dict)
                #else:
                #    query_embedding = model.predict(query_img.reshape(1,shape[0],shape[1],3))

                distance_vectors = np.squeeze(np.abs(all_embeddings[b] - query_embedding))
                distance = np.sum(distance_vectors, axis=1)
                top_inds = distance.argsort()[:r]
                output_classes = all_identities[b][top_inds]

                if np.where(output_classes == idt)[0].shape[0] > 0:
                    correct += 1
                test_iter += 1

            rank_arr.append(float(correct) / test_iter)
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

def evaluate_dist(model, generator, overlay, n_pairs=1000, train=False,
                    shape=input_shape,
                    preprocess=input_preprocess):
    distances = []
    for t in range(n_pairs):
        pair = generator.next()
        predict = []
        for i in range(2):
            img = data.imread_scale(pair[i], shape=shape, preprocess=preprocess)
            #if overlay:
            input_dict = {}
            for layer in model.layers:
                if layer.name.find('input_im') > -1:
                    input_dict[layer.name] = img.reshape(1,shape[0],shape[1],3)
                elif layer.name.find('input_cam') > -1:
                    input_dict[layer.name] = np.ones((1,128,64))
            predict.append(model.predict(input_dict))
            #else:
            #    predict.append(model.predict(img.reshape(1,shape[0],shape[1],3)))
        dist = distance.cosine(predict[0], predict[1])
        distances.append(dist)
    return distances

def get_score(model, hist=None,
                preprocess=input_preprocess, shape=input_shape,
                DATA_ROOT='/home/albert/github/tensorflow/'):
    score = {
        'rank' : {},
        'pos_distance' : [],
        'neg_distance' : [],
        'precision' : [],
        'recall' : [],
        'matt_coef' : [],
        'loss' : []
    }

    overlay = False
    for layer in model.layers:
        if layer.name.find('input_cam') > -1:
            overlay = True

    score['rank'] = evaluate_rank(model, overlay, [1,5,20], shape, preprocess, DATA_ROOT)

    """
    files_dict, _ = get_data('test')

    pos_generator = pos_pair_generator(files_dict)
    neg_generator = neg_pair_generator(files_dict)

    pos_distance = evaluate_dist(model, pos_generator, overlay, train=False,
                                preprocess=preprocess,
                                overlay_eval=overlay_eval,
                                shape=shape, r=r)
    score['pos_distance'] = pos_distance

    neg_distance = evaluate_dist(model, neg_generator, overlay, train=False,
                                preprocess=preprocess,
                                overlay_eval=overlay_eval,
                                shape=shape, r=r)
    score['neg_distance'] = neg_distance

    for thresh in np.linspace(0, np.max(np.array(pos_distance + neg_distance)), 101)[1:]:
        true = np.ones(len(pos_distance)).tolist() + np.zeros(len(neg_distance)).tolist()
        pred_pos = [int(d < thresh) for d in pos_distance]
        pred_neg = [int(d < thresh) for d in neg_distance]
        pred = pred_pos + pred_neg

        score['precision'].append(metrics.precision_score(true, pred))
        score['recall'].append(metrics.recall_score(true, pred))
        score['matt_coef'].append(metrics.matthews_corrcoef(true, pred))
    """

    try:
        score['loss'] += hist.history['loss']
    except AttributeError:
        pass

    return score

def plot_rank(model_root, ylim_0=0.4, ylim_1=1.0, end_2=False, file_root='/home/albert/github/tensorflow/models/'):
    if end_2:
        train_score = np.load(file_root + '%s/%s_score_2.npz' % (model_root, model_root))['arr_0'].item()
    else:
        train_score = np.load(file_root + '%s/%s_score.npz' % (model_root, model_root))['arr_0'].item()

    train_rank1_avg = []
    train_rank5_avg = []
    train_rank20_avg = []

    min_iter = np.array(train_score.keys()).min()
    max_iter = np.array(train_score.keys()).max()

    for it in range(min_iter, max_iter + 1000,1000):
        train_rank1_avg.append(np.average(np.array(train_score[it]['rank'][1])))
        train_rank5_avg.append(np.average(np.array(train_score[it]['rank'][5])))
        train_rank20_avg.append(np.average(np.array(train_score[it]['rank'][20])))

    plt.figure(figsize=(10,8))
    plt.plot(np.arange(min_iter,max_iter+1000,1000), train_rank1_avg, label='train_rank1_avg', )
    plt.plot(np.arange(min_iter,max_iter+1000,1000), train_rank5_avg, label='train_rank5_avg', )
    plt.plot(np.arange(min_iter,max_iter+1000,1000), train_rank20_avg, label='train_rank20_avg')

    plt.legend(loc='lower right')
    plt.title('rank: %s' % model_root)
    plt.xlabel('Iteration')
    plt.ylabel('Rank')
    plt.ylim(ylim_0,ylim_1)
    plt.show()

    m = np.array(train_rank1_avg).argmax()
    best_train = {1:train_rank1_avg[m], 5:train_rank5_avg[m], 20:train_rank20_avg[m]}

    print 'iterations' , 1000 * m + min_iter
    print 'best_train' , best_train

def plot_loss(model_root, ylim_0=0.4, ylim_1=1.0, end_2=False, file_root='/home/albert/github/tensorflow/models/'):
    if end_2:
        train_score = np.load(file_root + '%s/%s_score_2.npz' % (model_root, model_root))['arr_0'].item()
    else:
        train_score = np.load(file_root + '%s/%s_score.npz' % (model_root, model_root))['arr_0'].item()

    min_iter = np.array(train_score.keys()).min()
    max_iter = np.array(train_score.keys()).max()

    loss = []
    for it in range(min_iter,max_iter + 1000,1000):
        loss += train_score[it]['loss']
    plt.plot(loss)
    plt.title('training loss: %s' % model_root)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()

def load_model(json_file, weights_file=None):
    json_file = open(json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    if weights_file is not None:
        model.set_weights(np.load(weights_file))
    return model
