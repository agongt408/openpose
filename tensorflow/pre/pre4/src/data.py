import os
import json
import numpy as np
import cv2

import sys
sys.path.append('/home/albert/github/DenseNet/')
import densenet

P_param = 4
K_param = 4
input_output_dim = 128
input_shape = (256,128)
input_overlay = False
input_overlay_eval = False
input_preprocess = True
input_r = 0.6

BODY_PARTS = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar"
]

def exist_any_keypoints(img_path, keypoints='all', DATA_ROOT='/home/albert/github/tensorflow/'):
    if img_path.find('/') > -1:
        root = img_path[len(img_path) - img_path[::-1].index('/'):-4:]
    else:
        root = img_path[0:img_path.index('.')]

    try:
        x = img_path.index('train')
        keypoint_path = DATA_ROOT + 'data/market-1501/train_openpose/train_keypoints/%s_keypoints.json' % root
    except:
        try:
            x = img_path.index('test')
            keypoint_path = DATA_ROOT + 'data/market-1501/test_openpose/test_keypoints/%s_keypoints.json' % root
        except:
            keypoint_path = DATA_ROOT + 'data/market-1501/query_openpose/query_keypoints/%s_keypoints.json' % root

    with open(keypoint_path) as data_file:
        data = json.load(data_file)

    if keypoints == 'front':
        for person in data['people']:
            RShoulder_index = 3 * BODY_PARTS.index('RShoulder')
            RShoulder_key = person['pose_keypoints'][RShoulder_index : RShoulder_index + 3]
            LShoulder_index = 3 * BODY_PARTS.index('LShoulder')
            LShoulder_key = person['pose_keypoints'][LShoulder_index : LShoulder_index + 3]
            if RShoulder_key[0] < LShoulder_key[0]:
                return True
        return False
    elif keypoints == 'back':
        for person in data['people']:
            RShoulder_index = 3 * BODY_PARTS.index('RShoulder')
            RShoulder_key = person['pose_keypoints'][RShoulder_index : RShoulder_index + 3]
            LShoulder_index = 3 * BODY_PARTS.index('LShoulder')
            LShoulder_key = person['pose_keypoints'][LShoulder_index : LShoulder_index + 3]
            if RShoulder_key[0] > LShoulder_key[0]:
                return True
        return False
    else:
        if keypoints == 'all':
            keypoints = BODY_PARTS
        elif type(keypoints) == list:
            pass
        else:
            raise ValueError, 'invalid keypoints argument, must be "all" or list of strings'

        exist_any_keypoints = False
        keypoints_map = np.zeros((3 * len(BODY_PARTS)), dtype=np.int64)
        for k in keypoints:
            n = BODY_PARTS.index(k)
            keypoints_map[3 * n] = keypoints_map[3 * n + 1] = keypoints_map[3 * n + 2] = 1

        for person in range(len(data['people'])):
            if np.multiply(keypoints_map, np.array(data['people'][person]['pose_keypoints'])).max() > 0.0:
                return True
            else:
                return False

def get_data(split, DATA_ROOT='/home/albert/github/tensorflow/', keypoints=None):
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
                files_dict[idt] = []

            if split == 'query':
                path = DATA_ROOT + 'data/market-1501/query/' + f
            elif split == 'train':
                path = DATA_ROOT + 'data/market-1501/bounding_box_train/' + f
            elif split == 'test':
                path = DATA_ROOT + 'data/market-1501/bounding_box_test/' + f

            if keypoints == None:
                files_arr.append([path, idt])
                files_dict[idt].append(path)
            else:
                if exist_any_keypoints(path, keypoints, DATA_ROOT):
                    files_arr.append([path, idt])
                    files_dict[idt].append(path)

    for idt in files_dict.keys():
        if len(files_dict[idt]) == 0:
            files_dict.pop(idt)

    return files_dict, files_arr

def imread(img_path):
    """
    returns RGB image
    misc.imread is deprecated
    """
    im = cv2.imread(img_path)
    if len(im.shape) == 3:
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        return im

def imread_scale(img_path, shape=input_shape, preprocess=input_preprocess):
    if preprocess:
        return densenet.preprocess_input(cv2.resize(imread(img_path),
                (shape[1], shape[0])).astype(np.float64))
    else:
        return cv2.resize(imread(img_path),(shape[1], shape[0])).astype(np.float64)

def makeGaussian(shape, r=input_r, center=None):
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

    radius = np.array(shape).min() * r
    result = np.exp(-4*np.log(2) * (np.power(x-x0,2) + np.power(y-y0,2)) / radius**2)[0:shape[0], 0:shape[1]]
    return result

def create_keypoints(img_path, shape=input_shape, preprocess=input_preprocess,
                    r=input_r, keypoints=None, DATA_ROOT='/home/albert/github/tensorflow/'):
    if img_path.find('/') > -1:
        root = img_path[len(img_path) - img_path[::-1].index('/'):-4:]
    else:
        root = img_path[0:img_path.index('.')]

    htmp = np.zeros(shape).astype(np.float64)

    try:
        x = img_path.index('train')
        keypoint_path = DATA_ROOT + 'data/market-1501/train_openpose/train_keypoints/%s_keypoints.json' % root
    except:
        keypoint_path = DATA_ROOT + 'data/market-1501/test_openpose/test_keypoints/%s_keypoints.json' % root

    if keypoints == 'all':
        keypoints = BODY_PARTS
    elif type(keypoints) == list:
        pass
    elif keypoints == None:
        pass
    else:
        raise ValueError, 'invalid keypoints argument, must be "all", list or None'

    with open(keypoint_path) as data_file:
        data = json.load(data_file)
    if keypoints is None:
        for person in range(len(data['people'])):
            for i in range(0, len(data['people'][person]['pose_keypoints']), 3):
                x_key = data['people'][person]['pose_keypoints'][i] * shape[1] / 64.0
                y_key = data['people'][person]['pose_keypoints'][i + 1] * shape[0] / 128.0
                c_key = data['people'][person]['pose_keypoints'][i + 2]
                if not (x_key == 0 and y_key == 0):
                    htmp = np.maximum(htmp, makeGaussian(shape, r, (x_key, y_key)))
    else:
        for person in range(len(data['people'])):
            for k in keypoints:
                x_key = data['people'][person]['pose_keypoints'][BODY_PARTS.index(k) * 3] * shape[1] / 64.0
                y_key = data['people'][person]['pose_keypoints'][BODY_PARTS.index(k) * 3 + 1] * shape[0] / 128.0
                c_key = data['people'][person]['pose_keypoints'][BODY_PARTS.index(k) * 3 + 2]
                if not (x_key == 0 and y_key == 0):
                    htmp = np.maximum(htmp, makeGaussian(shape, r, (x_key, y_key)))

    # Make sure type is float
    return htmp

def output_batch_generator(files_dict, model=None, P=P_param, K=K_param, shape=input_shape,
                            preprocess=input_preprocess,
                            r=input_r, keypoints=None, cam_output_dim=None, cam_wide=False,
                            crop=True, flip=True, DATA_ROOT='/home/albert/github/tensorflow/'):
    print 'preprocess', preprocess
    print 'keypoints', keypoints

    while True:
        im_batch = []
        cam_batch = []

        orig_batch = []

        input_cam_dict = {'input_cam_fr' : [], 'input_cam_bk' : []}
        output_dict = {'final_output' : np.zeros((P*K, input_output_dim))}

        if cam_output_dim is not None:
            output_dict['cam_output'] = []
        idt_choice = np.random.choice(files_dict.keys(), P, replace=False)
        for p in range(len(idt_choice)):
            if K > len(files_dict[idt_choice[p]]):
                k_choice = np.random.choice(range(len(files_dict[idt_choice[p]])), K, replace=True)
            else:
                k_choice = np.random.choice(range(len(files_dict[idt_choice[p]])), K, replace=False)
            for k in k_choice:
                path = files_dict[idt_choice[p]][k]

                crop_x = np.random.randint(0.125 * shape[1])
                crop_y = np.random.randint(0.125 * shape[0])

                if keypoints == 'front_back':
                    if exist_any_keypoints(path, 'front', DATA_ROOT):
                        input_cam_dict['input_cam_fr'].append(np.ones((128, 64)).tolist())
                        input_cam_dict['input_cam_bk'].append(np.zeros((128, 64)).tolist())
                    else:
                        input_cam_dict['input_cam_fr'].append(np.zeros((128, 64)).tolist())
                        input_cam_dict['input_cam_bk'].append(np.ones((128, 64)).tolist())
                elif keypoints is not None and keypoints != 'ones':
                    cam_batch.append(create_keypoints(path, ((128,64)), preprocess,
                                    r, keypoints, DATA_ROOT).tolist())

                if cam_output_dim is not None:
                    if path.find('/') > -1:
                        root = path[len(path) - path[::-1].index('/'):-4:]
                    else:
                        root = path[0:path.index('.')]
                    if keypoints is None:
                        if cam_wide == True:
                            htmp = imread_scale(DATA_ROOT + 'data/market-1501/train_openpose/train_cams_wide/%s_cam_wide.png' \
                                            % root, cam_output_dim, False)
                        elif cam_wide == False:
                            htmp = imread_scale(DATA_ROOT + 'data/market-1501/train_openpose/train_cams/%s_cam.png' \
                                            % root, cam_output_dim, False)
                            htmp = np.mean(htmp, axis=2)
                        elif cam_wide == 'ones':
                            htmp = np.zeros(cam_output_dim)
                    else:
                        htmp = create_keypoints(path, cam_output_dim, preprocess, r, keypoints, DATA_ROOT)
                    if htmp.max() != 0.0:
                        htmp = htmp / htmp.max()
                    output_dict['cam_output'].append((np.ones(htmp.shape) - htmp).tolist())

                # orig_batch.append(imread_scale(path, shape, preprocess).tolist())

                im = None
                if crop:
                    # print crop_y, crop_y+shape[0], crop_x, crop_x+shape[1]
                    im = imread_scale(path, (int(1.125 * shape[0]), int(1.125 * shape[1])),
                                            preprocess)[crop_y:crop_y+shape[0], crop_x:crop_x+shape[1]]
                else:
                    im = imread_scale(path, shape, preprocess)

                if flip:
                    if np.random.randint(2) == 1:
                        im = np.flip(im, axis=1)

                im_batch.append(im.tolist())

        input_dict = {}
        if keypoints == 'front_back':
            input_dict = {'input_im_b' : np.array(im_batch),
                            'input_im_fr' : np.array(im_batch),
                            'input_im_bk' : np.array(im_batch),
                            'input_cam_fr' : np.array(input_cam_dict['input_cam_fr']),
                            'input_cam_bk' : np.array(input_cam_dict['input_cam_bk'])}
        elif keypoints is not None:
            if model is not None:
                for layer in model.layers:
                    if layer.name.find('input_im') > -1:
                        input_dict[layer.name] = np.array(im_batch)
                    elif layer.name.find('input_cam') > -1:
                        if keypoints == 'ones':
                            input_dict[layer.name] = np.ones((P*K, 128,64))
                        else:
                            input_dict[layer.name] = np.array(cam_batch)
            else:
                input_dict = {'input_im' : np.array(im_batch), 'input_cam' : np.array(cam_batch)}
        else:
            if model is not None:
                for layer in model.layers:
                    if layer.name.find('input_im') > -1:
                        input_dict[layer.name] = np.array(im_batch)
            else:
                input_dict = {'input_im' : np.array(im_batch)}
        if cam_output_dim is not None:
            output_dict['cam_output'] = np.array(output_dict['cam_output'])
        yield input_dict, output_dict #, np.array(orig_batch)
