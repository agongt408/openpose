import numpy as np
# from sklearn import metrics
# from scipy.spatial import distance
import time
import matplotlib.pyplot as plt

import os
import sys
sys.path.append('/home/albert/github/tensorflow/src/')
import data

input_shape = (256,128)
input_preprocess = True

def get_data(split, keypoints=None, DATA_ROOT='/home/albert/github/tensorflow/'):
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
            # if idt != 0 and idt != -1:
            if not any(idt == l for l in files_dict.keys()):
                files_dict[idt] = {}

            if split == 'query':
                path = DATA_ROOT + 'data/market-1501/query/' + f
            elif split == 'train':
                path = DATA_ROOT + 'data/market-1501/bounding_box_train/' + f
            elif split == 'test':
                path = DATA_ROOT + 'data/market-1501/bounding_box_test/' + f

            camera = f[f.index('_') + 2 : f.index('_') + 3]

            if keypoints is None:
                files_arr.append([path, idt, int(camera)])
                try:
                    files_dict[idt][int(camera)].append(path)
                except:
                    files_dict[idt][int(camera)] = []
                    files_dict[idt][int(camera)].append(path)
            else:
                if data.exist_any_keypoints(path, keypoints, DATA_ROOT):
                    files_arr.append([path, idt, int(camera)])
                    try:
                        files_dict[idt][int(camera)].append(path)
                    except:
                        files_dict[idt][int(camera)] = []
                        files_dict[idt][int(camera)].append(path)

    return files_dict, files_arr

def get_embeddings(model, test_files=None, query_files=None,
                    shape=input_shape, preprocess=input_preprocess,
                    DATA_ROOT='/home/albert/github/tensorflow/', cover=False):
    if test_files is None or query_files is None:
        _, test_files = get_data('test', DATA_ROOT)
        _, query_files = get_data('query', DATA_ROOT)

    all_embeddings = []
    query_embeddings = []

    for files_arr, embeddings_arr in [(test_files, all_embeddings), (query_files, query_embeddings)]:
        s, z = time.time(), 0

        for f, _, _ in files_arr:
            img = data.imread_scale(f, shape, preprocess)

            if cover:
                img *= np.tile(data.create_keypoints(f, shape, preprocess, 0.3, ['Neck']).reshape((shape[0], shape[1], 1)), [1,1,3])
                # plt.imshow(img)
                # plt.show()

            # img = img[16:16+shape[0], 8:8+shape[1]]

            input_dict = {}
            for layer in model.layers:
                if layer.name.find('input') > -1:
                    input_dict[layer.name] = img.reshape(1,shape[0],shape[1],3)
                if layer.name.find('input_cam') > -1:
                    input_dict[layer.name] = np.ones((1,128,64))
            # print input_dict.keys()
            if len(model.outputs) == 1:
                predict = model.predict(input_dict)
            else:
                predict = model.predict(input_dict)[0]
            # print predict.shape
            embeddings_arr.append(predict.tolist())

            #embeddings_arr.append(np.random.normal(loc=0, scale=1, size=(128)).tolist())

            z += 1
            if z % 1000 == 0:
                print z, time.time() - s

    return np.array(all_embeddings), np.array(query_embeddings)

def evaluate_rank(all_embeddings, query_embeddings,
                    test_dict=None, test_files=None, query_files=None,
                    rank=[1,5], DATA_ROOT='/home/albert/github/tensorflow/'):
    if test_dict is None or test_files is None or query_files is None:
        test_dict, test_files = get_data('test', DATA_ROOT)
        _, query_files = get_data('query', DATA_ROOT)

    all_identities = np.array([p[1] for p in test_files])
    all_camera = np.array([p[2] for p in test_files])

    correct = np.array([0] * len(rank))
    test_iter = np.array([0] * len(rank))

    for q in range(len(query_files)):
        idt, camera = int(query_files[q][1]), int(query_files[q][2])
        b = np.logical_or(all_camera != camera, all_identities != idt)

        if len(test_dict[idt].keys()) > 1:
            query_embedding = query_embeddings[q]
            distance_vectors = np.squeeze(np.abs(all_embeddings[b] - query_embedding))
            distance = np.sum(distance_vectors, axis=1)

            for r in range(len(rank)):
                top_inds = distance.argsort()[:rank[r]]
                output_classes = all_identities[b][top_inds]
                # print output_classes

                if np.where(output_classes == idt)[0].shape[0] > 0:
                    correct[r] += 1
                test_iter[r] += 1

    return np.divide(correct.astype(np.float32), test_iter).tolist()

def evaluate_mAP(all_embeddings, query_embeddings,
                test_dict=None, test_files=None, query_files=None,
                DATA_ROOT='/home/albert/github/tensorflow/'):
    if test_dict is None or test_files is None or query_files is None:
        test_dict, test_files = get_data('test', DATA_ROOT)
        _, query_files = get_data('query', DATA_ROOT)

    all_identities = np.array([p[1] for p in test_files])
    all_camera = np.array([p[2] for p in test_files])

    AP = []

    for q in range(len(query_files)):
        idt, camera = query_files[q][1], query_files[q][2]
        b = np.logical_or(all_camera != camera, all_identities != idt)

        if len(test_dict[idt].keys()) > 1:
            query_embedding = query_embeddings[q]

            distance_vectors = np.squeeze(np.abs(all_embeddings[b] - query_embedding))
            distance = np.sum(distance_vectors, axis=1)

            top_inds = distance.argsort()
            output_classes = all_identities[b][top_inds]

            precision = []
            correct_old = 0

            for t in range(distance.shape[0]):
                if idt == output_classes[t]:
                    precision.append(float(correct_old + 1) / (t + 1))
                    correct_old += 1

            AP.append(np.mean(np.array(precision)))

    return np.mean(np.array(AP))

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
            input_dict = {}
            for layer in model.layers:
                if layer.name.find('input_im') > -1:
                    input_dict[layer.name] = img.reshape(1,shape[0],shape[1],3)
                if layer.name.find('input_cam') > -1:
                    input_dict[layer.name] = np.ones((1,128,64))
            predict.append(model.predict(input_dict))
        dist = distance.cosine(predict[0], predict[1])
        distances.append(dist)
    return distances

def get_score(model, hist=None, keypoints=None,
                preprocess=input_preprocess, shape=input_shape,
                DATA_ROOT='/home/albert/github/tensorflow/', cover=False):
    score = {
        'rank' : {},
        'mAP' : 0,
        'loss' : []
    }

    test_dict, test_files = get_data('test', keypoints, DATA_ROOT)
    _, query_files = get_data('query', keypoints, DATA_ROOT)

    all_embeddings, query_embeddings = get_embeddings(model, test_files, query_files,
                                                        input_shape, input_preprocess, DATA_ROOT, cover)
    score['rank'] = evaluate_rank(all_embeddings, query_embeddings, test_dict,
                                    test_files, query_files, [1,5,20], DATA_ROOT)
    score['mAP'] = evaluate_mAP(all_embeddings, query_embeddings, test_dict,
                                test_files, query_files, DATA_ROOT)

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
        if type(train_score[it]['rank']) is dict:
            train_rank1_avg.append(np.average(np.array(train_score[it]['rank'][1])))
            train_rank5_avg.append(np.average(np.array(train_score[it]['rank'][5])))
            train_rank20_avg.append(np.average(np.array(train_score[it]['rank'][20])))
        elif type(train_score[it]['rank']) is list:
            train_rank1_avg.append(np.average(np.array(train_score[it]['rank'][0])))
            train_rank5_avg.append(np.average(np.array(train_score[it]['rank'][1])))
            train_rank20_avg.append(np.average(np.array(train_score[it]['rank'][2])))
        else:
            raise TypeError, 'rank container must be list or dict'

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
