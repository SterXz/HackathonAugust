import keras
from keras.utils.vis_utils import plot_model
import numpy as np
import tensorflow as tf
import copy
import sqlite3
tf.__version__
%load_ext tensorboard


SEQ_LEN = 2
def allru(sentence):
    for string in sentence:
        #print(string)
        for s in string:
            #print(s, (s in 'ёйцукенгшщзхъфывапролджэячсмитьбю-'))
            if s not in 'qwertyuiopasdfghjklzxcvbnmёйцукенгшщзхъфывапролджэячсмитьбю-': 
                print('странный символ', s)
                return False
    return True
 
def readtxt (links):
    if type(links) != list:
        links = [links]
    output =[]
    for link in links:
        with open (link , 'r') as f:    
            s = f.read() 
        some_shit = [',', '\t', '- ', '"', ')', '(', '«', '»', '\ufeff']
        for sim in some_shit:
            s= s.replace(sim, '')
        s = s.replace('!', '.')
        s = s.replace('?', '.')
        s = s.replace('\n', '.') 
        s = s.replace('…', '.') 
        s = s.lower()
        sentl = s.split('.')
        sentl = list(filter(None, sentl))
        for sent in sentl:
            sent = list(filter(None, sent.split(' ')))
            if allru(sent):
                output.append(sent)
            else:
                print('else', sent)
    return output
 
def f_r_gen (data, targets, b_size):
    x_f_set = data[0]
    x_r_set = data[1]
    butch=[]
    i=0
    while 1:
        if i+b_size <= len(x_f_set):
            butch_x_f = x_f_set[i:i+b_size]
            butch_x_r = x_r_set[i:i+b_size]
            butch_y = targets[i:i+b_size]
            i+=b_size
        else:
            butch_x_f=x_f_set[i:]
            butch_x_r=x_r_set[i:]
            butch_y = targets[i:]
            i = i+b_size-len(x_f_set)
            f = x_f_set[:i]
            r = x_r_set[:i]
            y = targets[:i]
            butch_x_f = np.concatenate([butch_x_f, f], axis=0)
            butch_x_r = np.concatenate([butch_x_r, r], axis=0)
            butch_y = np.concatenate([butch_y, y], axis=0)
        yield [butch_x_f, butch_x_r], butch_y
 
 
 
def sent_to_seq(sent_list):
    f_x = []
    r_x = []
    y = []
    for sent in sent_list:
        pad_sent = [0 for _ in range(SEQ_LEN)]+copy.copy(sent)+[0, 0]
 
        for i in range(SEQ_LEN, len(pad_sent)-SEQ_LEN):
            f_seq = np.array(pad_sent[i-SEQ_LEN:i])
            r_seq = np.array(pad_sent[i+1:i+SEQ_LEN+1][::-1])
            f_x.append(f_seq)
            r_x.append(r_seq)
            y.append(pad_sent[i])
    return [np.array(f_x), np.array(r_x)], y #обрезка!!!
 
 
list_str = readtxt(['/content/drive/My Drive/SlangPredl.txt', '/content/drive/My Drive/dtd.txt'])
#list_str = readtxt(['/content/drive/My Drive/SlangPredl.txt', '/content/drive/My Drive/feedback.txt'])
Set = []
for sent in list_str:
   Set+=sent
MAX_WORDS = len(set(Set))
 
tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(list_str)
sequences = tokenizer.texts_to_sequences(list_str)
x, y = sent_to_seq(sequences)
x_val = copy.copy(x)
print('x_val', x_val)
print(len(x), len(y))
y_data = keras.utils.to_categorical(y, num_classes=MAX_WORDS)
y_data_val = copy.copy(y_data)
print(MAX_WORDS)
valid_x = [x[-320:] for x in x_val]
valid_y = y_data_val[-320:]
print(len(valid_y))

SEQ_LEN = 2
EMBD_SIZE = 256
VOC_SIZE = 10
B_SIZE = 32
logdir="logs/fit/" + '111'
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
 
class WeightedSum(keras.layers.Layer):
    def __init__(self, b_size):
        self.b_size = b_size
        super(WeightedSum, self).__init__()
    def build(self, input_shape):
        
        self.w = self.add_weight(
            shape=([input_shape[1]]),
            initializer="ones",
            trainable=True,
        )
        w = self.w
 
        new = tf.expand_dims(w, axis=-1)
        for _ in range(1, input_shape[2]):
            new = tf.concat([new, tf.expand_dims(w, axis=-1)], axis=-1)
 
        newnew = tf.expand_dims(new, axis=-1)
        for _ in range(1, input_shape[3]):
            newnew = tf.concat([newnew, tf.expand_dims(new, axis=-1)], axis=-1)
        self.w = newnew
 
 
    def call(self, inputs):
        print('nu privet', inputs)
        w = tf.broadcast_to(self.w, (self.b_size, inputs.shape[1], inputs.shape[2], inputs.shape[3]))#
        sum = tf.math.reduce_sum(inputs*w, axis=1)
        sum = tf.slice(sum, [0, 1, 0], [B_SIZE, 1, 512])
        return tf.squeeze(sum, axis=1) #tf.math.reduce_sum(inputs*w, axis=1)
 
def stack(inp):
    
    # stack_ = tf.stack(inp, axis=1)
    # slice_ = tf.slice(stack_, [0, 0, stack_.shape[2], 0], [B_SIZE, stack_.shape[1], 1, stack_.shape[3]])
    return tf.stack(inp, axis=1)
 
f_inp = keras.layers.Input(shape=(SEQ_LEN), name='f')
r_inp = keras.layers.Input(shape=(SEQ_LEN), name='r')
 
embd = keras.layers.Embedding(MAX_WORDS, EMBD_SIZE, mask_zero=True)
f_embd = embd(f_inp)
r_embd = embd(r_inp)
 
f_lstm = keras.layers.LSTM(EMBD_SIZE, return_sequences=True)(f_embd)
r_lstm = keras.layers.LSTM(EMBD_SIZE, return_sequences=True)(r_embd)
concatenate_1 = keras.layers.Concatenate(axis=-1)([f_lstm, r_lstm])
 
f_lstm = keras.layers.LSTM(EMBD_SIZE, return_sequences=True)(f_lstm)
r_lstm = keras.layers.LSTM(EMBD_SIZE, return_sequences=True)(r_lstm)
concatenate_2 = keras.layers.Concatenate(axis=-1)([f_lstm, r_lstm])
 
f_lstm = keras.layers.LSTM(EMBD_SIZE, return_sequences=True)(f_lstm)
r_lstm = keras.layers.LSTM(EMBD_SIZE, return_sequences=True)(r_lstm)
concatenate_3 = keras.layers.Concatenate(axis=-1)([f_lstm, r_lstm])
 
stack = keras.layers.Lambda(stack)([concatenate_3, concatenate_2, concatenate_1])
sum = WeightedSum(B_SIZE)(stack)
 
d = keras.layers.Dense(MAX_WORDS, activation='softmax')(sum) 
 
model = keras.models.Model(inputs=[f_inp, r_inp], outputs = d)
 
model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics='acc')
plot_model(model, 'testebani.png', show_shapes=True, show_layer_names=True)

model2 = keras.models.Model(inputs=[f_inp, r_inp], outputs = sum)
# np.set_printoptions(suppress=True) 
p_pool = model2.predict([x[0][:15264], x[1][:15264]])
# for y1, y2, xf, xr in zip(np.argmax(p, axis=1), y[:320], x[0][:320], x[1][:320]): #предсказание слов по контексту
#     print(tokenizer.sequences_to_texts([xf]), tokenizer.sequences_to_texts([[y1]]), '|', tokenizer.sequences_to_texts([[y2]]), tokenizer.sequences_to_texts([xr[::-1]]))
 
# for y1, y2, xf, xr in zip(p, y[:], x[0][:], x[1][:]): #генератор ембединга по контексту 
#     print(tokenizer.sequences_to_texts([[y2]]), '=>', y1)

def encode(string1, i_1, string2, i_2):    
    l = string1.split(' ')
    l = tokenizer.texts_to_sequences(l)
    l_new=[]
    for i1 in l:
        if i1:
            l_new.append(i1[0])
        else:
            l_new.append(0)
    print(l_new)
    l = [0, 0]+l_new+[0, 0]
    x1 = l[i_1:i_1+2]
    x2 = l[i_1+3:i_1+3+2][::-1]
    x1_b=[]
    x2_b = []
    print(x1, x2)
    for _ in range(32):
        x1_b.append(x1)
        x2_b.append(x2)
    x1_b = np.array(x1_b)
    x2_b = np.array(x2_b)
    print(x2_b.shape, x[0].shape)
    
    p = model2.predict([x1_b, x2_b])
    ###
    l = string2.split(' ')
    l = tokenizer.texts_to_sequences(l)
    l_new=[]
    for i1 in l:
        if i1:
            l_new.append(i1[0])
        else:
            l_new.append(0)
    #print(l_new)
    l = [0, 0]+l_new+[0, 0]
    x1 = l[i_2:i_2+2]
    x2 = l[i_2+3:i_2+3+2][::-1]
    x1_b=[]
    x2_b = []
    #print(x1, x2)
    for _ in range(32):
        x1_b.append(x1)
        x2_b.append(x2)
    x1_b = np.array(x1_b)
    x2_b = np.array(x2_b)
    #print(x2_b.shape, x[0].shape)
    
    p_2 = model2.predict([x1_b, x2_b])

    min=1000000000
    list_words=[]
    
    for embd, y_word in zip(p_pool, y):
        distance = np.sum((p[0]-embd)**2)
        if distance<min:
            #print(distance)
            w=y_word
            list_words.append(y_word)
    return np.sum((p[0]-p_2[0])**2)#tokenizer.sequences_to_texts([list_words])[::-1]
    #print(x1_b, len(x2_b))
encode('я очень хочу спать сейчас', 4, 'сегодня я ел вкусный ужин', 3)

model.fit(f_r_gen (x, y_data, 32), steps_per_epoch=1000, epochs=10, callbacks=[tensorboard_callback], validation_data=(valid_x, valid_y),  validation_batch_size=32)

model.predict(valid_x)[0]

%tensorboard --logdir logs

valid_x

conn = sqlite3.connect("DaBa.db") # или :memory: чтобы сохранить в RAM
cursor = conn.cursor()
 
# Создание таблицы
#cursor.execute("""CREATE TABLE slovar
#                  (slovo, embedingi,
#                   sinonimi, sr_embedingi)
#               """)
 
def vstavka (slovo, embedingi, sinonimi, sr_embedingi):
  conn = sqlite3.connect("DaBa.db")
  cursor = conn.cursor()
 
  # Вставляем данные в таблицу
  sqlite_insert_with_param = """INSERT INTO slovar
                            (slovo, embedingi, sinonimi, sr_embedingi) 
                            VALUES (?, ?, ?, ?);"""
 
#data_tuple = ('kul', '12', 'nais', '4')
#cursor.execute(sqlite_insert_with_param, data_tuple)
vstavka('kul', '12', 'nais', '4')
# Сохраняем изменения
conn.commit()
 
cursor.execute("""SELECT * FROM slovar""")
print(cursor.fetchall())