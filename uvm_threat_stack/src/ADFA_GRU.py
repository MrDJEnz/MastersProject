#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""Implements a simple GRU for syscall prediction on the ADFA-LD dataset
"""
from matplotlib import pyplot

__author__ = "Duncan Enzmann modified from:John H. Ring IV"

import keras.layers
import numpy as np  # type: ignore
from keras import Model, Sequential  # type: ignore
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, GRU, Dense, CuDNNGRU  # type: ignore
from keras.utils import to_categorical, Sequence  # type: ignore
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu
# from keras.utils.vis_utils import plot_model
from keras.models import load_model
import keras.backend as K
from keras.regularizers import l2
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import random



from keras_self_attention import SeqSelfAttention
from ADFA_Processing import load_files, Encoder, generate_sequence_pairs



def create_gru(seq_len: int, vocab_size: int) -> Model:
    """Implements a GRU for sequence to sequence prediction

    Args:
        seq_len: Length of input and prediction sequence
        vocab_size: Vocabulary size of sequence domain

    Returns:
        Keras GPU model for specified sequences

    """
    model = Sequential()
    model.add(keras.layers.GRU(vocab_size, input_shape=(seq_len, vocab_size), activation='relu', return_sequences=True))

    opt = optimizers.adam(clipvalue=5.0)
    model.compile(loss='categorical_crossentropy',
    optimizer = "adam",
    metrics = ['categorical_accuracy'])  # , perplexity])
    model.summary()


    return model


def create_gru_test1(seq_len=8, vocab_size=175, layers=2, lr=0.1):
    '''

    :Hidden layers: 1
    :nodes per layer: 256
    :training batch: 64
    :dropout: 0.5
    :lr: 0.1
    :maximum graident clipping threshold: 5
    :return: model to train
    '''
    model = Sequential()

    model.add(keras.layers.GRU(vocab_size, input_shape=(seq_len, vocab_size), activation='relu', return_sequences=True))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    for i in range(layers-1):
        model.add(keras.layers.GRU(vocab_size, activation='relu', return_sequences=True))
        model.add(keras.layers.Dropout(0.5))


    model.add(keras.layers.GRU(vocab_size, activation='softmax', return_sequences=True))

    # model.add(CuDNNGRU(vocab_size, input_shape=(seq_len, vocab_size), kernel_regularizer=l2(0.001), return_sequences=True))
    # model.add(SeqSelfAttention(attention_activation='tanh'))
    # for i in range(layers - 1):
    #     model.add(CuDNNGRU(vocab_size, return_sequences=True))
    #     model.add(keras.layers.Dropout(0.5))
    #
    # model.add(CuDNNGRU(vocab_size, return_sequences=True))


    opt = optimizers.adam(lr=lr, clipvalue=5.0)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=opt,
    #               metrics=['categorical_accuracy'])#, perplexity])
    # model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['categorical_accuracy', perplexity])
    model.summary()
    #plot_model(model, to_file='model.png', show_shapes=True)





    return model

# Wikipedia, and github forums for information...
# https://en.wikipedia.org/wiki/Perplexity
def perplexity(y_true, y_pred):
    cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    perplexity = K.pow(2.0, cross_entropy)
    return perplexity



def plot_roc(y_true, y_pred):
    from collections import Counter
    print(Counter(y_true)) # Expects 50% 0's and 50% 1's
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=0)
    print("fpr: " + str(fpr))
    print("tpr: " + str(tpr))
    print("thresh: " + str(thresholds))
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel('True Positive Rate')
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.show()


def get_real_probs(pred, trgt):
    seq_len = len(trgt[0])
    probs = []
    y_pred = pred
    y = trgt
    i = 0
    for y_elm, y_pred_elm in zip(y, y_pred):
        print("y_elm")
        print(y_elm[i])
        print("\n\ny_pred")
        print(y_pred_elm[0])
        print("\n\ntry to index y_pred_elm[0][0]")
        a = y_elm[i]
        print("y_elm["+str(i)+"]")
        print(a)
        a = int(a)
        print(y_pred_elm[0][a])
        i += 1
        elm_probs = []
        for seq_idx in range(seq_len):
            a = y_elm[seq_idx]
            a = int(a)
            elm_probs.append(y_pred_elm[a][seq_idx])
            print("y_pred_elm["+str(a)+"]["+str(seq_idx)+"]")
        print("Elements:")
        print(elm_probs)
        # elm_probs = np.log(elm_probs)
        # probs.append(-np.sum(elm_probs))
        # probs.append(np.power(np.e, np.sum(np.log(elm_probs))))

        prob = max(elm_probs)
        print("Prob: ")
        print(prob)
        b = -np.log(prob)
        print("Appending: " + str(b) + " to probs2")
        probs.append(-np.log(prob))
        print(probs)
    return np.array(probs)


# Plot a confusion matrix.
# cm is the confusion matrix, names are the names of the classes.
def plot_confusion_matrix(cm, names, title='Confusion matrix',
                            cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def draw_roc(model, val_sequences, test_sequences, atk_sequences):
    # final_scores = np.zeros(len(val_sequences[0][0] + len(atk_sequences[0][0])))
    model = load_model(model)

    # Get the predicted model scores.
    predX = val_sequences[0][0]
    predY = test_sequences[0][1]
    predv = [predX, predY]
    prediction = np.concatenate(predv)
    pred = model.predict(prediction)
    # This will contain a shape of (64, 10, 177) # Batch size, seq len, vocab size


    # Next we need to get our target which we will use as truth values... we can choose valseq[1][0] and testseq[1][0]
    # This is the target
    trgtx = val_sequences[1][0]
    trgty = test_sequences[1][1]
    trgtv = [trgtx, trgty]
    trgt = np.concatenate(trgtv)

    if len(trgt) == len(pred):
        print("Matching lengths of length: " + str(len(pred)))

    # Since they are in sequence format we need to turn them into binary.
    # Score 1 for Not attack and Score 0 for attack
    # First we need to get some attack data to compare against.
    atk = atk_sequences[0][0]
    normal = val_sequences[0][0]

    # Create the truth comparative: This is a mix of the normal and attack values.
    truth = np.concatenate(
        [np.ones(len(normal)), np.zeros(len(atk))]
    )

    # Next we need to create scoring for the prediction to be a 0 or 1.
    x_data = pred
    y_data = trgt
    num_subsequences = [len(x) for x in x_data]
    file_to_idx = []
    offset = 0
    for num in num_subsequences:
        file_to_idx.append(np.arange(num) + offset)
        offset += num
    # x_data = np.concatenate(pred)
    y_data = np.concatenate(y_data)

    probs = get_real_probs(x_data, y_data)

    scores =[]

    # print(probs)
    for n in range(len(file_to_idx)):
        a = np.max(probs[n])
        scores.append(a)
    print(scores)
    # scores = np.array([np.max(probs[n]) for n in file_to_idx])
    # Let's compare the truth value to the predicted value


    # pred_concat = np.concatenate(pred)

    print(len(truth))
    plot_roc(truth, scores)




def pad(array, reference, offsets):
    """
    array: Array to be padded
    reference: Reference array with the desired shape
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    """
    # Create an array of zeros with the reference shape
    result = np.zeros(reference.shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim]) for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result

def train_gru_test(train_data, val_data, test_to_run,  model: Model, epochs=int, batch_size=int, patience=int) -> None:
    """Trains model of ADFA-LD data
    Model: First GRU test
    Args:
        model: Keras model to train
        skip: see generate_sequence_pairs()

    Returns:

    """
    # setting early stopping for the network...
    # paper does not have patience listed so patience of 0
    # Create a checkpoint of the best model.
    callbacks = [EarlyStopping(monitor='val_loss', patience=patience),
                 # ModelCheckpoint(filepath='tests/'+str(test_to_run) + '/trainAttk_VS_trainVal_'+str(test_to_run)+'_best_model_acc{categorical_accuracy:02f}.hdf5', monitor='val_loss', save_best_only=True)]
                 ModelCheckpoint(filepath='tests/'+str(test_to_run) + '/PICS_best_model_acc{categorical_accuracy:02f}.hdf5', monitor='val_loss', save_best_only=True)]

    history = model.fit(
        x=train_data,   # train features
        # y=Y,   # Target vector
        # batch_size=batch_size,  # num of observations per batch
        epochs=epochs,  # num epochs
        callbacks=callbacks,  # Early stopping
        verbose=1,  # Verbose on
        validation_data=val_data, # Data for eval
        shuffle=True    # Shuffle the data
    )

    figure, axes = pyplot.subplots(nrows=2, ncols=2)
    axes[0,0].plot(history.history['categorical_accuracy'], label='train')
    axes[0,0].plot(history.history['val_categorical_accuracy'], label='test')
    axes[0,0].legend()
    # axes[0,0].show()

    axes[0,1].plot(history.history['perplexity'], label='perplex_train')
    axes[0,1].plot(history.history['val_perplexity'], label='perplex_test')
    axes[0,1].legend()
    # axes[0,1].show()

    figure.tight_layout()
    figure.show()

class Generator(Sequence):
    def __init__(self, x_set, y_set, batch_size=64, vocab_size=176):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])
        self.vocab_size = vocab_size

    def __len__(self):
        return np.int(np.ceil(self.x.shape[0] / self.batch_size))

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y = self.y[inds]
        return to_categorical(batch_x, num_classes=self.vocab_size), to_categorical(batch_y, num_classes=self.vocab_size), [None]

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def run(test_to_run, seq_len, vocab_size, train_data_gen, val_data_gen, epochs, batch_size, trials):
    if test_to_run == 1:
        # model1
        '''

            :Hidden layers: 1
            :nodes per layer: 256
            :training batch: 64
            :dropout: 0.5
            :lr: 0.1
            :maximum graident clipping threshold: 5
            :return: model to train
            '''
        for idx in range(trials):
            model = create_gru_test1(seq_len, vocab_size, layers=1, lr=0.1)
            train_gru_test(train_data_gen, val_data_gen, test_to_run, model=model, epochs=epochs, batch_size=batch_size, patience=0)

    elif test_to_run == 2:
        # # model 2
        '''
        GRU-2
        :Hidden layers: 2
        :nodes per layer: 256
        :training batch: 64
        :dropout: 0.5
        :lr: 0.1
        :maximum graident clipping threshold: 5
        :return: model to train
        '''
        for idx in range(trials):
            model = create_gru_test1(seq_len, vocab_size, layers=2, lr=0.1)
            train_gru_test(train_data_gen, val_data_gen, test_to_run, model=model, epochs=epochs, batch_size=batch_size, patience=0)
        #
        # model = create_gru_test1(seq_len, vocab_size, layers=2, lr=0.1)
        # train_gru_test(model=model, epochs=epochs, batch_size=64, patience=3)
        #
    elif test_to_run == 3:
        # # model 3
        '''
        GRU-3
        :Hidden layers: 3
        :nodes per layer: 256
        :training batch: 64
        :dropout: 0.5
        :lr: 0.1
        :maximum graident clipping threshold: 5
        :return: model to train
        '''

        for idx in range(trials):
            model = create_gru_test1(seq_len, vocab_size, layers=3, lr=0.1)
            train_gru_test(train_data_gen, val_data_gen, test_to_run, model=model, epochs=epochs, batch_size=batch_size, patience=0)

        # model = create_gru_test1(seq_len, vocab_size, layers=3, lr=0.1)
        # train_gru_test(model=model, epochs=epochs, batch_size=64, patience=3)
    elif test_to_run == 4:
        # model 3.1
        '''
        :Hidden layers: 3
        :nodes per layer: 175
        :training batch: 64
        :dropout: 0.5
        :lr: 0.001
        :maximum graident clipping threshold: 5
        :return: model to train
        '''
        for idx in range(trials):
            model = create_gru_test1(seq_len, vocab_size, layers=3, lr=0.001)
            train_gru_test(train_data_gen, val_data_gen, test_to_run, model=model, epochs=epochs, batch_size=batch_size, patience=0)

            #return
        #
        # model = create_gru_test1(seq_len, vocab_size, layers=3, lr=0.001)
        # train_gru_test(model=model, epochs=epochs, batch_size=64, patience=3)
    elif test_to_run == 5:
        for idx in range(trials):
            model = create_gru(seq_len, vocab_size)
            train_gru_test(train_data_gen, val_data_gen, test_to_run, model=model, epochs=epochs, batch_size=batch_size, patience=0)

    else:
        print("Invalid test.... please pass in 1, 2, 3, or 4")



def BLEU_TESTING(test, val):
    # https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    if test == 0:
        reference = [['this', 'is', 'small', 'test']]
        candidate = ['this', 'is', 'a', 'test']
        score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        print(score)
    elif test == 1:
        # Not important information gathered...
        reference = [['this', 'is', 'a', 'test']]
        candidate = ['this', 'is', 'a', 'test']
        print('Individual 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
        print('Individual 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))
        print('Individual 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 1, 0)))
        print('Individual 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 0, 1)))
    elif test ==2 :
        # BLEU n-gram cumulative scores
        reference = [['this', 'is', 'small', 'test']]
        candidate = ['this', 'is', 'a', 'test']
        print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
        print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
        print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
        print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))
    elif test == 3:
        # Testing a workable model
        reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
        candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
        score = sentence_bleu(reference, candidate)
        print(score)
    elif test == 4:
        # See a big drop in the score of ~25% from missing 1 word
        reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
        candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
        score = sentence_bleu(reference, candidate)
        print(score)
    elif test == 5:
        # See a massive drop in the score of ~51% from 2 mismatched words
        reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
        candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'sleepy', 'dog']
        score = sentence_bleu(reference, candidate)
        print(score)
    elif test == 6:
        # Paper Score We train our model on GTX1050Ti GPU devices. The final
        # perplex of the three-layer model is under 0.4 after training
        # 48 hours.

        # Use the target vector in the test set as the reference sequence
        BLEU_SCORE_REF = val[0]
        print("Target shape for BLEU: \n")
        print(BLEU_SCORE_REF.shape)
        print("---\n\n\n")

        # load in out best model for testing BLEU-score
        model = load_model('best_model.h5')
        print(model.shape())

        BLEU_SCORE_CAN = None
    else:
        print("ERROR INVALID BLEU TEST")

def one_one_ratio(train, att):
    numElemsDelete = len(train) - len(att)
    numElemsKeep = len(train) - numElemsDelete
    print("Size of train: " + str(len(train)))
    print("Size of att: " + str(len(att)))
    random.shuffle(train)
    trainResized = train[:numElemsKeep]
    trainLeftOver = train[numElemsKeep:]

    print("Size of att: " + str(len(att)))
    print("Size trainResized: " + str(len(trainResized)))

    return trainResized, trainLeftOver

def get_adfa_sequence_group(data, seq_len, skip, keep_nested):
    encoder = Encoder("../data/encoder.npy")
    vec_encode = np.vectorize(encoder.encode)
    # vec = np.vectorize(data)
    x_val = []
    y_val= []
    for row in data:
        x, y = generate_sequence_pairs(vec_encode(row), seq_len, skip=skip)
        x_val.append(x)
        y_val.append(y)

    if not keep_nested:
        x_val = np.concatenate(x_val)
        y_val = np.concatenate(y_val)

    return x_val, y_val


if __name__ == '__main__':
    input_dims = [len(np.load("../data/encoder.npy", allow_pickle=True).item()) + 1]
    seq_len = 10
    epochs = 25
    batch_size = 64
    vocab_size = input_dims[0]
    trials = 10

    dataPreprocessed = 1
    if dataPreprocessed == 0:
        dataTrain = load_files('train')
        dataAtt = load_files('attack')
        dataVal = load_files('val')

        dataTrainResized = one_one_ratio(dataTrain, dataAtt)
        dataValResized = one_one_ratio(dataVal, dataAtt)

        dataNormal = np.concatenate((dataTrain, dataVal), 0)
        dataNormalResized, dataNormalLeftOver = one_one_ratio(dataNormal, dataAtt)

        dataTest = np.concatenate((dataAtt,dataNormalResized), 0)



        # train = get_adfa_sequence_group(dataTrainResized, seq_len, skip=True, keep_nested=False)
        # val = get_adfa_sequence_group(dataValResized, seq_len, skip=True, keep_nested=False)
        test = get_adfa_sequence_group(dataTest, seq_len, skip=True, keep_nested=False)
        attack = get_adfa_sequence_group(dataAtt, seq_len, skip=True, keep_nested=False)
        normal = get_adfa_sequence_group(dataNormalLeftOver, seq_len, skip=True, keep_nested=False)

        np.savetxt('data_to_use/normalDataX.out', normal[0], delimiter=',')
        np.savetxt('data_to_use/normalDataY.txt', normal[1], delimiter=',')


        xTrain, xval, yTrain, yval = train_test_split(normal[0], normal[1], test_size=.3, random_state=42)

        np.savetxt('data_to_use/trainingDataX.out', xTrain, delimiter=',')
        np.savetxt('data_to_use/trainingDataY.out', yTrain, delimiter=',')
        np.savetxt('data_to_use/valDataX.out', xval, delimiter=',')
        np.savetxt('data_to_use/valDataY.out', yval, delimiter=',')
        np.savetxt('data_to_use/testingDataX.out', test[0], delimiter=',')
        np.savetxt('data_to_use/testingDataY.out', test[1], delimiter=',')
        np.savetxt('data_to_use/attackDataX.out', attack[0], delimiter=',')
        np.savetxt('data_to_use/attackDataY.out', attack[1], delimiter=',')


        print("vSize: " + str(vocab_size))
        normal_data_gen = Generator(normal[0], normal[1], batch_size, vocab_size)
        train_data_gen = Generator(xTrain, yTrain, batch_size, vocab_size)
        testing_data_gen = Generator(test[0], test[1], batch_size, vocab_size)
        # testingVal_data_gen = Generator(testingVal[0], testingVal[1], batch_size, vocab_size)
        val_data_gen = Generator(xval, yval, batch_size, vocab_size)
        atk_data_gen = Generator(attack[0], attack[1], batch_size, vocab_size)


    else:

        normalx = np.loadtxt('data_to_use/normalDataX.out', delimiter=',')
        normaly = np.loadtxt('data_to_use/normalDataY.txt', delimiter=',')
        xTrain = np.loadtxt('data_to_use/trainingDataX.out', delimiter=',')
        yTrain = np.loadtxt('data_to_use/trainingDataY.out', delimiter=',')
        xval = np.loadtxt('data_to_use/valDataX.out', delimiter=',')
        yval = np.loadtxt('data_to_use/valDataY.out', delimiter=',')
        testX = np.loadtxt('data_to_use/testingDataX.out', delimiter=',')
        testY = np.loadtxt('data_to_use/testingDataY.out', delimiter=',')
        attackX = np.loadtxt('data_to_use/attackDataX.out', delimiter=',')
        attackY = np.loadtxt('data_to_use/attackDataY.out', delimiter=',')


        traindX = np.concatenate((xTrain, attackX), 0)
        traindY = np.concatenate((yTrain, attackY), 0)

        valX = np.concatenate((xval, attackX), 0)
        valY = np.concatenate((yval, attackY), 0)

        print("vSize: " + str(vocab_size))
        normal_data_gen = Generator(normalx, normaly, batch_size, vocab_size)
        # train_data_gen = Generator(xTrain, yTrain, batch_size, vocab_size)
        train_data_gen = Generator(traindX, traindY, batch_size, vocab_size)
        testing_data_gen = Generator(testX, testY, batch_size, vocab_size)
        # testingVal_data_gen = Generator(testingVal[0], testingVal[1], batch_size, vocab_size)
        val_data_gen = Generator(valX, valY, batch_size, vocab_size)
        atk_data_gen = Generator(attackX, attackY, batch_size, vocab_size)

    # Change this number to change the neural net test
    # 0 -> nothing
    # 1 -> 1 layer lr = 0.1
    # 2 -> 2 layers lr = 0.1
    # 3 -> 3 layers lr = 0.1
    # 4 -> 4 layers lr = 0.001
    if dataPreprocessed == 1:
        trained = 0
        if trained == 0:
            test_to_run = 4
            print("\n------------Running Neural Network------------\n")
            # run(test_to_run, seq_len, vocab_size, train_data_gen, val_data_gen, epochs, batch_size, trials)
            run(test_to_run, seq_len, vocab_size, train_data_gen, val_data_gen, epochs, batch_size, trials)

            # 6 will be the actual test for the paper, 0-5 are examples
            # bleu_test = 99
            # print("\n------------Running BLEU Tests------------\n")
            # BLEU_TESTING(bleu_test, val, )
            # print("------------Done with BLEU Tests-----------\n\n")
        if trained == 2:
            print(len(traindY))
            print(len(testY))
        else:
            if dataPreprocessed == 1:

                model_load = "tests/4/trainAttk_VS_trainVal_4_best_model_acc0.461834.hdf5" # Model 4 59
                # model_load = "tests/5/trainAttk_VS_trainVal_5_best_model_acc0.394675.hdf5" # Johns

                draw_roc(model_load, val_data_gen, testing_data_gen, atk_data_gen)


