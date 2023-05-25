import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os
import tensorflow as tf
from tensorflow.python.client import timeline
import io
from scipy.io import savemat, loadmat
from tensorflow.keras.callbacks import Callback
import time
from tensorflow.keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from tensorflow.python.eager.context import context, EAGER_MODE, GRAPH_MODE, eager_mode, graph_mode

from model import HDC_tf_preproc, HDC_hdcc_preproc

def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
    
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 

    return batch_s

def one_hot(y_,n_classes=-1):
    # Function to encode output labels from number indexes 
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    if n_classes==-1:
        n_values = np.max(y_) + 1
    else:
        n_values = n_classes
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

def load_motorway_dataset(data_path='data'):
    # Function to load the motorway dataset only 

    with open(os.path.join(data_path, 'motorway_dataset_window_64_proc_veh_DtA.pkl'), 'rb') as f:
        save = pickle.load(f, encoding='latin1')
        motorway_dataset = save['dataset']
        motorway_labels = save['labels']
        del save
        print('Motorway set', motorway_dataset.shape, motorway_labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(motorway_dataset, motorway_labels, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

def load_secondary_dataset(data_path='data'):
    # Function to load the secondary dataset only 

    with open(os.path.join(data_path,'secondary_dataset_window_64_proc_veh_DtA.pkl'), 'rb') as f:
        save = pickle.load(f, encoding='latin1')
        secondary_dataset = save['dataset']
        secondary_labels = save['labels']
        del save
        print('Secondary set', secondary_dataset.shape, secondary_labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(secondary_dataset, secondary_labels, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

def load_full_dataset(data_path='data'):
    # Function to load the full dataset (motorway+secondary roads)

    with open(os.path.join(data_path, 'motorway_dataset_window_64_proc_veh_DtA.pkl'), 'rb') as f:
        save = pickle.load(f, encoding='latin1')
        motorway_dataset = save['dataset']
        motorway_labels = save['labels']
        del save
        print('Motorway set', motorway_dataset.shape, motorway_labels.shape)

    with open(os.path.join(data_path,'secondary_dataset_window_64_proc_veh_DtA.pkl'), 'rb') as f:
        save = pickle.load(f, encoding='latin1')
        secondary_dataset = save['dataset']
        secondary_labels = save['labels']
        del save
        print('Secondary set', secondary_dataset.shape, secondary_labels.shape)

    dataset = np.concatenate((motorway_dataset,secondary_dataset), axis=0)
    labels = np.concatenate((motorway_labels,secondary_labels), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

class TimingCallback(Callback):
    def __init__(self):
        self.logs=[]
    def on_epoch_begin(self,epoch, logs={}):
        self.starttime=time.time()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(time.time()-self.starttime)


def load_dataset(dataset,config):
    """
    load the specific data set (from the data/ folder)
    @param dataset: specifies the data set [string]
    @param config: configure struct with necessary parameters [struct]
    @param hdc_encoded: if the data set to be loaded is to be HDC-coded [Bool]
    @return: set of training and test data [list]
    """
    # load preprocessed data
    if dataset == "full":
        X_train, X_test, y_train, y_test = load_full_dataset()
    elif dataset == "motorway":
        X_train, X_test, y_train, y_test = load_motorway_dataset()
    elif dataset == "secondary":
        X_train, X_test, y_train, y_test = load_secondary_dataset()
    elif dataset == "full_crossval":
        data = loadmat('data/uah_dataset.mat')
        mot_data = data['motorway_dataset']
        sec_data = data['secondary_dataset']
        mot_label = data['motorway_labels']
        sec_label = data['secondary_labels']

        # creat cross-validation split (3-fold cross-validation) on the full dataset
        k = 3
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        #info = data['data_info']

        kf = KFold(n_splits=k)
        #motorway
        for train_idx, test_idx in kf.split(mot_data):
            X_train.append(mot_data[train_idx])
            y_train.append(mot_label[train_idx])
            X_test.append(mot_data[test_idx])
            y_test.append(mot_label[test_idx])
        k_idx = 0
        # secondary
        for train_idx, test_idx in kf.split(sec_data):
            X_train[k_idx]=np.concatenate((X_train[k_idx],sec_data[train_idx]),0)
            y_train[k_idx]=np.concatenate((y_train[k_idx],sec_label[train_idx]), 0)
            X_test[k_idx]=np.concatenate((X_test[k_idx],sec_data[test_idx]),0)
            y_test[k_idx]=np.concatenate((y_test[k_idx],sec_label[test_idx]), 0)
            k_idx+=1
    else:
        print("No valid dataset argument was set!")

    return X_train, X_test, y_train, y_test, config


def create_HDC_vectors(config, input):
    """
    create the HDC vectors from given input
    @param config: config struct
    @param input: inputs tensor with size m x t x v (m... number of samples, t... number of timesteps, v... number of variables)
    """
    with graph_mode():
        tf.config.optimizer.set_jit(True)
        # pre initialize vectors
        init_vec = tf.random.uniform(shape=(config.input_dim, 1), minval=-np.pi, maxval=np.pi, seed=1,
                                     dtype="float32")
        sensor_ids = tf.random.uniform(shape=(config.input_dim, config.n_inputs), minval=-np.pi, maxval=np.pi,
                                       seed=2,
                                       dtype="float32")
        timestamps = tf.random.uniform(shape=(config.input_dim, config.n_steps), minval=-np.pi, maxval=np.pi,
                                       seed=3,
                                       dtype="float32")
        init_vecs = {'init_vec': init_vec, 'sensor_ids': sensor_ids, 'timestamps': timestamps, 'scale':config.scale}

        X = tf.compat.v1.placeholder(tf.float32, [None, config.n_steps, config.n_inputs], name="X")
        preproc = HDC_tf_preproc(X, init_vecs)
        print("preproc", preproc)
        t_proc = []
        traces = []

        for i in range(config.n_time_measures):
            sess = tf.compat.v1.Session()
            options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            run_metadata = tf.compat.v1.RunMetadata()
            t = time.perf_counter()
            output = sess.run(preproc, feed_dict={X: input}, options=options, run_metadata=run_metadata)
            t_proc.append((time.perf_counter() - t))
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            traces.append(chrome_trace)
        preprocessing_time = np.median(t_proc)
    print("output", output)
    return preprocessing_time, output, traces, init_vecs


import hdcc.HDProg as hdcc
prog = None


def create_HDC_vectors_comp(config, input):
    """
    create the HDC vectors from given input
    @param config: config struct
    @param input: inputs tensor with size m x t x v (m... number of samples, t... number of timesteps, v... number of variables)
    """
    # check if preprocessed vectors are already saved
    # hash the input tensor to get a unique identifier
    input_hash = hash((config, str(input)))
    if os.path.exists(f'preproc/{input_hash}.npy'):
        print("Preprocessed vectors already exist. Loading...")
        preproc = np.load(f'preproc/{input_hash}.npy', allow_pickle=True).item()
        return 0, preproc['output'], [], preproc['init_vecs']
    with graph_mode():
        tf.config.optimizer.set_jit(True)
        # pre initialize vectors
        init_vec = tf.random.uniform(shape=(config.input_dim, 1), minval=-np.pi, maxval=np.pi, seed=1,
                                     dtype="float32")
        init_vec_np = init_vec.eval(session=tf.compat.v1.Session())
        sensor_ids = tf.random.uniform(shape=(config.input_dim, config.n_inputs), minval=-np.pi, maxval=np.pi,
                                       seed=2,
                                       dtype="float32")
        sensor_ids_np = sensor_ids.eval(session=tf.compat.v1.Session())
        timestamps = tf.random.uniform(shape=(config.input_dim, config.n_steps), minval=-np.pi, maxval=np.pi,
                                       seed=3,
                                       dtype="float32")
        timestamps_np = timestamps.eval(session=tf.compat.v1.Session())
        init_vecs = {'init_vec': init_vec, 'sensor_ids': sensor_ids, 'timestamps': timestamps, 'scale':config.scale}
        init_vecs_np = {'init_vec': init_vec_np, 'sensor_ids': sensor_ids_np, 'timestamps': timestamps_np, 'scale':config.scale}

        print("=== create HDC vectors ===")
        global prog
        if prog is None:
            print("=== init program ===")
            hdc_prog_init(config, init_vecs_np)
        print("=== run program ===")
        print("input", input)
        print("input shape", input.shape)
        hdcc_output = []
        for j in range(input.shape[0]):
            input_dict = {}
            for k in range(config.n_inputs):
                for l in range(config.n_steps):
                    input_dict['input_' + str(l) + '_' + str(k)] = input[j, l, k] * config.scale
            hdcc_output.append(prog.run(prog.build(), input_dict)[1].data)
            print("  > " + str(j + 1) + "/" + str(input.shape[0]) + " done", end="\r")
        print()
        print("ALL DONE")
        hdcc_output = np.float32(np.array(hdcc_output))
        print()
        print("HDCC output", hdcc_output)

        X = tf.compat.v1.placeholder(tf.float32, [None, config.n_steps, config.n_inputs], name="X")
        preproc = HDC_tf_preproc(X, init_vecs)
        print("preproc", preproc)
        t_proc = []
        traces = []

        for i in range(config.n_time_measures):
            sess = tf.compat.v1.Session()
            options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            run_metadata = tf.compat.v1.RunMetadata()
            t = time.perf_counter()
            output = sess.run(preproc, feed_dict={X: input}, options=options, run_metadata=run_metadata)
            t_proc.append((time.perf_counter() - t))
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            traces.append(chrome_trace)
        preprocessing_time = np.median(t_proc)
    print("Original output", output)

    print("HDC type", type(hdcc_output))
    print("Original type", type(output))
    
    # check if output is the same
    # if np.allclose(hdcc_output[0], output[0]):
    #     print("HDC output is the same as the original output")
    #     exit(0)
    # else:
    #     print("HDC output is NOT the same as the original output")
    #     # print more information and save two outputs for comparison
    #     print("HDC output shape", np.shape(hdcc_output[0]))
    #     print("Original output shape", np.shape(output))
    #     print("Original output[0] shape", np.shape(output[0]))
    #     # save outputs
    #     np.savetxt("hdcc_output.txt", hdcc_output[0])
    #     np.savetxt("original_output.txt", output)
    #     exit(1)

    init_vecs['prog'] = prog

    # save preprocessed vectors
    if not os.path.exists('preproc'):
        os.makedirs('preproc')
    np.save(f'preproc/{input_hash}.npy', {'output': hdcc_output, 'init_vecs': init_vecs})
    return preprocessing_time, hdcc_output, traces, init_vecs


def hdc_prog_init(config, init_vecs=None):
    """
    initialize the hdcc program
    """
    if init_vecs:
        sensor_ids = init_vecs['sensor_ids']
        time_stamps = init_vecs['timestamps']
        init_vec = init_vecs['init_vec']
    else:
        sensor_ids = np.random.uniform(-np.pi, np.pi, (config.input_dim, config.n_inputs))
        time_stamps = np.random.uniform(-np.pi, np.pi, (config.input_dim, config.n_steps))
        init_vec = np.random.uniform(-np.pi, np.pi, (config.input_dim, 1))

    global prog
    prog = hdcc.HDProg()
    prog.add_param(int, 'input_dim', config.input_dim)
    prog.add_output(hdcc.Types.HV_FHRR, 'output', 'input_dim')

    for i in range(config.n_inputs):
        prog.decl_const(hdcc.Types.HV_FHRR, 'sensor_ids_' + str(i), hdcc.Types.HV_FHRR(config.input_dim, sensor_ids[:, i]))
        for j in range(config.n_steps):
            prog.add_input(float, 'input_' + str(j) + '_' + str(i))
    for i in range(config.n_steps):
        prog.decl_const(hdcc.Types.HV_FHRR, 'timestamps_' + str(i), hdcc.Types.HV_FHRR(config.input_dim, time_stamps[:, i]))
    
    prog.decl_const(hdcc.Types.HV_FHRR, 'init_vec', hdcc.Types.HV_FHRR(config.input_dim, np.array(init_vec)[:, 0]))
    prog.add_param(int, 'scale', config.scale)
    prog.add_param(int, 'n_steps', config.n_steps)
    prog.add_param(int, 'n_inputs', config.n_inputs)

    prog.assign('output', prog.bundle(*[
        prog.bind(
            prog.bundle(*[
                prog.bind(
                    'sensor_ids_' + str(i),
                    prog.frac_bind(
                        'init_vec',
                        'input_' + str(t) + '_' + str(i),
                    )
                ) for i in range(config.n_inputs)
            ]),
            'timestamps_' + str(t)
        ) for t in range(config.n_steps)
    ]))
    print("=== build program ===")
    state = prog.build()
    print("=== run program ===")
    print(prog)


def create_HDC_vectors_hdcc(config, input):
    """
    Same as create_HDC_vectors but using the hdcc package
    create the HDC vectors from given input
    @param config: config struct
    @param input: inputs tensor with size m x t x v (m... number of samples, t... number of timesteps, v... number of variables)
    """
    print("=== create HDC vectors ===")
    global prog
    if prog is None:
        print("=== init program ===")
        hdc_prog_init(config)
    t_proc = []

    for i in range(config.n_time_measures):
        output = []
        t = time.perf_counter()
        print("> generating HDC vectors: ")
        for j in range(input.shape[0]):
            input_dict = {}
            for k in range(config.n_inputs):
                for l in range(config.n_steps):
                    input_dict['input_' + str(l) + '_' + str(k)] = input[j, l, k] * config.scale
            state = prog.build()
            output.append(prog.run(state, input_dict)[1].data)
            print("  > " + str(j + 1) + "/" + str(input.shape[0]) + " done", end="\r")
        t_proc.append(time.perf_counter() - t)
    preprocessing_time = np.median(t_proc)
    print("=== done ===")

    return preprocessing_time, output, [], {'init_vec': state.val_table['init_vec'][3], 'sensor_ids': [state.val_table['sensor_ids_' + str(i)][3] for i in range(config.n_inputs)], 'timestamps': [state.val_table['timestamps_' + str(i)][3] for i in range(config.n_steps)], 'scale':config.scale, 'prog': prog}