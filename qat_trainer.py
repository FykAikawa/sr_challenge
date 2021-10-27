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
from tensorflow.keras.layers import Conv2D, Input, ReLU, Lambda
quantize_model = tfmot.quantization.keras.quantize_model
import sys

AUTOTUNE = tf.data.experimental.AUTOTUNE
batch_size = 8

def load_crop_resize(path):

    img = Image.open(path)
    img_resize = img.crop((0, 0, 360, 640))

    return img_resize

def representative_dataset():
    n=0
    for i,x in enumerate(Path("resizeddiv").glob("**/*")):
        if os.path.isfile(x) and i % 40 == 0:
            n+=1
            print(n)
            yield [np.expand_dims(np.asarray(load_crop_resize(x)),0).astype(np.float32)]
args = sys.argv

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


def test_step(model_R,epoch,scale):
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

class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return []
    def get_activations_and_quantizers(self, layer):
        return []
    def set_quantize_weights(self, layer, quantize_weights):
        pass
    def set_quantize_activations(self, layer, quantize_anctivations):
        pass
    def get_output_quantizers(self, layer):
        return []
    def get_config(self):
        return {}

def model_quantizer(model):
    annotate_model = tf.keras.models.clone_model(
        model,
        clone_function=lambda x: tfmot.quantization.keras.quantize_annotate_layer(x, quantize_config=NoOpQuantizeConfig()) if 'lambda' in x.name else x
    )
    annotate_model = tfmot.quantization.keras.quantize_annotate_model(annotate_model)
    depth_to_space = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 3))
    with tfmot.quantization.keras.quantize_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig, 'depth_to_space': depth_to_space, 'tf': tf}):
        q_model = tfmot.quantization.keras.quantize_apply(annotate_model)
    return q_model

k_model = tf.keras.models.load_model(args[-1],compile=False)
psnr = test_step(k_model,0,3)
model = model_quantizer(k_model)
scale = 3
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001*(batch_size/8)),
              loss='mean_squared_error')

train_ds=MyDataLoader(scale)
count=0
dame_flag=0
best_psnr=0
best_model=None
while True:
    model.fit(train_ds,epochs=1,steps_per_epoch=40000//batch_size)
    psnr_qat = test_step(model,0,3)
    print(f"PSNR(pretrained):{psnr:5f},PSNR(quantized):{psnr_qat:5f}")
    if psnr_qat > best_psnr:
        best_model = model
        best_psnr = psnr_qat
        print("updated")
    if psnr_qat > psnr*0.999:
        print("PSNR recovered")
        best_model = model
        break

    if count >=10:
        print("much time past so abort")
        break 
    count+=1

input_name = best_model.input_names[0]
index = best_model.input_names.index(input_name)
best_model.inputs[index].set_shape([1,360,640,3])

converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
OPTIMIZATIONS = [tf.lite.Optimize.DEFAULT]
CHECKPOINT_BASE="model_tflite"
OUT_DIR = "out_tfmodel"
converter.optimizations =  [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_types = [tf.int8]
converter.experimental_new_converter=True
converter.experimental_new_quantizer=True
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
if not(os.path.isdir(f'{OUT_DIR}')):
    os.mkdir(f'{OUT_DIR}')
tflite_filepath = os.path.join(OUT_DIR, f"{CHECKPOINT_BASE}1.tflite")
tflite_model = converter.convert()

with open(tflite_filepath, "wb") as f:
    f.write(tflite_model)    