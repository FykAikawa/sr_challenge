import keras
from espcn_keras import ESPCNRGB,ESPCNRGB_Residual,ESPCNDeconv,ESPCNDeconv_Residual
from baseseven import base7,base7_woresdual
import tensorflow as tf
import math
import tensorflow_datasets as tfds
import pathlib
from pathlib import Path
import tqdm
from random import randint
import numpy as np
from PIL import Image
import time
import os
from statistics import mean
import tensorflow_model_optimization as tfmot
import random
from NAdaBelief import Nadabelief
quantize_model = tfmot.quantization.keras.quantize_model
import sys
args = sys.argv
AUTOTUNE = tf.data.experimental.AUTOTUNE
batch_size = 8

class MyDataLoader():
    def __init__(self,scale):
        self.x_path = [f"./D2K/X1/0{p:03d}.png" for p in range(1,801)] 
        self.y_path = [f"./D2K/X3/0{p:03d}x3.png" for p in range(1,801)] 
        self.pack = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(self.x_path),tf.data.Dataset.from_tensor_slices(self.y_path)))
        self.num_examples = len(self.x_path)
        path_ds = self.pack
        self.ds = path_ds.map(self.load_and_preprocess_image,num_parallel_calls=AUTOTUNE).cache()
        pfc=tf.py_function
        self.ds = self.ds.repeat().shuffle(buffer_size=self.num_examples).map(self.get_pair).batch(batch_size)
        self.ds = self.ds.prefetch(buffer_size=AUTOTUNE)
        self.ds=iter(self.ds)

    def get_pair(self,x,y):
        a = tf.shape(y)
        h = a[0]
        w = a[1]
        lr_x = tf.random.uniform(shape=(),minval=0,maxval=w-120,dtype=tf.int32)
        lr_y = tf.random.uniform(shape=(),minval=0,maxval=h-120,dtype=tf.int32)
        return y[lr_y:lr_y+120,lr_x:lr_x+120,:],x[3*lr_y:3*lr_y+360,3*lr_x:3*lr_x+360,:]
    def load_and_preprocess_image(self,path,path2):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image2 = tf.io.read_file(path2)
        image2 = tf.image.decode_jpeg(image2, channels=3)        
        return tf.dtypes.cast(image,tf.float32),tf.dtypes.cast(image2,tf.float32)
    def __iter__(self):
        return self
    def __next__(self):
        data = next(self.ds)
        return data
    def __len__(self):
        return (self.num_examples)

class MyDataLoader_2():
    def __init__(self,scale):
        all_image_paths= [p for p in Path("resizeddiv").glob('**/*') if p.suffix in [".jpg", ".jpeg", ".png", ".bmp"]]
        all_image_paths = [str(path) for path in all_image_paths]
        #print(all_image_paths)
        self.num_examples = len(all_image_paths)
        self.repetit=100
        path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
        self.ds = path_ds.repeat().shuffle(buffer_size=self.num_examples)
        self.ds = self.ds.map(lambda t:self.load_and_preprocess_image(t,scale) ,num_parallel_calls=AUTOTUNE).batch(batch_size)
        self.ds = self.ds.prefetch(buffer_size=AUTOTUNE)
        self.ds=iter(self.ds)
    @tf.function
    def load_and_preprocess_image(self,path,scale):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image=tf.image.random_crop(image, size=(360, 360,3))
        image_resized = tf.image.resize(
        image, [360//scale,360//scale], method='area'
        )  
        return image_resized,tf.dtypes.cast(image,tf.float32)
    def __iter__(self):
        return self
    def __next__(self):
        data = next(self.ds)
        return data
    def __len__(self):
        return (self.num_examples*self.repetit)//batch_size


def test_step(model_R,epoch,name,scale,batch_size):
    if not(os.path.isdir(f'param_{name}_x{scale}_bs{batch_size}')):
        os.mkdir(f'param_{name}_x{scale}_bs{batch_size}')
    model_R.save(f'param_{name}_x{scale}_bs{batch_size}/my_model_{epoch}')

    psnr_list=[]
    for a in range(1,101):
        hr=np.asarray(Image.open(f"DIV2K_valid_HR/pic/0{a+800}.png"))
        h,w,c=hr.shape
        hr = tf.reshape(hr,[1,h,w,c])
        lr=np.asarray(Image.open(f"DIV2K_valid_LR_bicubic/X3/png/0{a+800}x3.png"))
        h,w,c=lr.shape
        lr = tf.reshape(lr,[1,h,w,c])        
        generated_img = model_R(lr)
        psnr_list.append(10*math.log10(65535./np.mean((hr.numpy()-generated_img.numpy())**2)))
    return mean(psnr_list)


epoch=0
start=time.time()
scale=3
zenhan,kouhan = str(args[-1]).split("/")
epoch = 0
name = (zenhan.strip("_x3_bs"))[0].replace('param_','')+"_retrain"
model = tf.keras.models.load_model(args[-1],compile=False)
#model.summary()
train_summary_writer = tf.summary.create_file_writer(f"./logs/train/{name}_{scale}_bs{batch_size}")
valid_summary_writer = tf.summary.create_file_writer(f"./logs/valid/{name}_{scale}_bs{batch_size}")
loss_object = tf.keras.losses.MeanSquaredError()
train_ds=MyDataLoader(scale)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001*(batch_size/8)),
              loss='mean_squared_error')
while time.time()-start<3600*12:
    hist = model.fit(train_ds,epochs=1,steps_per_epoch=50000//batch_size)
    train_loss = mean(hist.history['loss'])
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', float(train_loss), step=epoch)
    psnr = test_step(model,epoch,name,scale,batch_size)
    with valid_summary_writer.as_default():
        tf.summary.scalar('valid_psnr', psnr, step=epoch)
    template = 'Epoch {}, Loss: {}, PSNR: {}'
    print (template.format(epoch+1,train_loss,psnr))
    epoch+=1

