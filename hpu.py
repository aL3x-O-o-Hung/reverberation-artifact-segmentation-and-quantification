from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import tensorflow_probability as tfp
from data_augmentation import *
from load_test_data import *

BASE_NUM_KERNELS=64

print(tf.__version__)

"""
Our proposed network, which is the second network in the paper
"""


class BatchNormRelu(tf.keras.layers.Layer):
    """Batch normalization + ReLu"""

    def __init__(self,name=None,dtype=None):
        super(BatchNormRelu,self).__init__(name=name)
        self.bnorm=tf.keras.layers.BatchNormalization(momentum=0.999,
                                                      scale=False,
                                                      dtype=dtype)
        self.relu=tf.keras.layers.ReLU(dtype=dtype)

    def call(self,inputs,is_training):
        x=self.bnorm(inputs,training=is_training)
        x=self.relu(x)
        return x


class Conv2DTranspose(tf.keras.layers.Layer):
    """Conv2DTranspose layer"""

    def __init__(self,output_channels,kernel_size,name=None,dtype=None):
        super(Conv2DTranspose,self).__init__(name=name)
        self.tconv1=tf.keras.layers.Conv2DTranspose(
            filters=output_channels,
            kernel_size=kernel_size,
            strides=2,
            padding='same',
            activation=tf.keras.activations.relu,
            dtype=dtype
        )

    def call(self,inputs):
        x=self.tconv1(inputs)
        return x


class Conv2DFixedPadding(tf.keras.layers.Layer):
    """Conv2D Fixed Padding layer"""

    def __init__(self,filters,kernel_size,stride,name=None,dtype=None):
        super(Conv2DFixedPadding,self).__init__(name=name)
        self.conv1=tf.keras.layers.Conv2D(filters,
                                          kernel_size,
                                          strides=1,
                                          dilation_rate=1,
                                          padding=('same' if stride==1 else 'valid'),
                                          activation=None,
                                          dtype=dtype
                                          #kernel_initializer=tf.keras.initializers.ones
                                          )

    def call(self,inputs):
        x=self.conv1(inputs)
        return x


class ConvBlock(tf.keras.layers.Layer):
    """Downsampling ConvBlock on Encoder side"""

    def __init__(self,filters,kernel_size=3,do_max_pool=True,name=None):
        super(ConvBlock,self).__init__(name=name)
        self.do_max_pool=do_max_pool
        self.conv1=Conv2DFixedPadding(filters=filters,
                                      kernel_size=kernel_size,
                                      stride=1)
        self.brelu1=BatchNormRelu()
        self.conv2=Conv2DFixedPadding(filters=filters,
                                      kernel_size=kernel_size,
                                      stride=1)
        self.brelu2=BatchNormRelu()
        self.max_pool=tf.keras.layers.MaxPool2D(pool_size=2,
                                                strides=2,
                                                padding='valid')

    def call(self,inputs,is_training):
        x=self.conv1(inputs)
        x=self.brelu1(x,is_training)
        x=self.conv2(x)
        x=self.brelu2(x,is_training)
        output_b=x
        if self.do_max_pool:
            x=self.max_pool(x)
        return x,output_b


class DeConvBlock(tf.keras.layers.Layer):
    """Upsampling DeConvBlock on Decoder side"""

    def __init__(self,filters,kernel_size=2,name=None):
        super(DeConvBlock,self).__init__(name=name)
        self.tconv1=Conv2DTranspose(output_channels=filters,kernel_size=kernel_size)
        self.conv1=Conv2DFixedPadding(filters=filters,
                                      kernel_size=3,
                                      stride=1)
        self.brelu1=BatchNormRelu()
        self.conv2=Conv2DFixedPadding(filters=filters,
                                      kernel_size=3,
                                      stride=1)
        self.brelu2=BatchNormRelu()

    def call(self,inputs,output_b,is_training):
        x=self.tconv1(inputs)

        """Cropping is only used when convolution padding is 'valid'"""
        src_shape=output_b.shape[1]
        tgt_shape=x.shape[1]
        start_pixel=int((src_shape-tgt_shape)/2)
        end_pixel=start_pixel+tgt_shape

        cropped_b=output_b[:,start_pixel:end_pixel,start_pixel:end_pixel,:]
        """Assumes that data format is NHWC"""
        x=tf.concat([cropped_b,x],axis=-1)

        x=self.conv1(x)
        x=self.brelu1(x,is_training)
        x=self.conv2(x)
        x=self.brelu2(x,is_training)
        return x


class PriorBlock(tf.keras.layers.Layer):
    """calculating Prior Block"""

    def __init__(self,filters,name=None):  #filters: number of the layers incorporated into the decoder
        super(PriorBlock,self).__init__(name=name)
        self.conv=Conv2DFixedPadding(filters=filters*2,kernel_size=1,stride=1)

    def call(self,inputs):
        x=self.conv(inputs)
        s=x.get_shape().as_list()[3]
        mean=x[:,:,:,0:s//2]
        mean=tf.keras.activations.tanh(mean)
        logvar=x[:,:,:,s//2:]
        logvar=tf.keras.activations.sigmoid(logvar)
        var=K.exp(logvar)
        #var=K.abs(logvar)
        return tf.concat([mean,var],axis=-1)



class PriorBlockmean(tf.keras.layers.Layer):
    """calculating Prior Block"""

    def __init__(self,filters,name=None):  #filters: number of the layers incorporated into the decoder
        super(PriorBlockmean,self).__init__(name=name)
        self.conv=Conv2DFixedPadding(filters=filters,kernel_size=1,stride=1)

    def call(self,inputs):
        mean=self.conv(inputs)
        mean=tf.keras.activations.tanh(mean)
        return mean


class PriorBlockvar(tf.keras.layers.Layer):
    """calculating Prior Block"""

    def __init__(self,filters,name=None):  #filters: number of the layers incorporated into the decoder
        super(PriorBlockvar,self).__init__(name=name)
        self.conv=Conv2DFixedPadding(filters=filters,kernel_size=1,stride=1)

    def call(self,inputs):
        var=self.conv(inputs)
        var=tf.keras.activations.sigmoid(var)
        var=K.exp(var)
        return var


@tf.function
def prob_function(inputs):
    s=inputs.get_shape().as_list()
    s[3]=int(s[3]/2)
    dist=tfp.distributions.Normal(loc=0.0,scale=1.0)
    samp=dist.sample([1,s[1],s[2],s[3]])
    #g=tf.random.Generator.from_seed(1234)
    #dis=g.normal(shape=s)
    dis=tf.math.multiply(samp,inputs[:,:,:,s[3]:])
    dis=tf.math.add(dis,inputs[:,:,:,0:s[3]])
    return dis


class Prob(tf.keras.layers.Layer):
    def __init__(self,name=None):
        super(Prob,self).__init__(name=name)

    def call(self,inputs):
        s=inputs.get_shape().as_list()
        s[3]=int(s[3]/2)
        dist=tfp.distributions.Normal(loc=0.0,scale=1.0)
        samp=dist.sample([1,s[1],s[2],s[3]])
        #g=tf.random.Generator.from_seed(1234)
        #dis=g.normal(shape=s)
        dis=tf.math.multiply(samp,K.sqrt(inputs[:,:,:,s[3]:]))
        dis=tf.math.add(dis,inputs[:,:,:,0:s[3]])
        return dis


class Encoder(tf.keras.layers.Layer):
    """encoder of the network"""

    def __init__(self,num_layers,num_filters,name=None):
        super(Encoder,self).__init__(name=name)
        self.convs=[]
        for i in range(num_layers):
            if i<num_layers-1:
                conv_temp=ConvBlock(filters=num_filters[i],name=name+'_conv'+str(i+1))
            else:
                conv_temp=ConvBlock(filters=num_filters[i],do_max_pool=False,name=name+'_conv'+str(i+1))
            self.convs.append(conv_temp)

    def call(self,inputs,is_training=True):
        list_b=[]
        x=inputs
        for i in range(len(self.convs)):
            x,b=self.convs[i](x,is_training=is_training)
            list_b.append(b)
        return x,list_b


class DecoderWithPriormeanBlockPosterior(tf.keras.layers.Layer):
    """decoder of the network with prior block in Posterior"""

    def __init__(self,num_layers,num_filters,num_filters_prior,name=None):
        super(DecoderWithPriormeanBlockPosterior,self).__init__(name=name)
        self.num_layers=num_layers
        self.num_filters=num_filters
        self.num_filters_prior=num_filters_prior
        self.deconvs=[]
        self.priors=[]
        #
        #self.prob_function=Prob()
        for i in range(num_layers):
            self.deconvs.append(DeConvBlock(num_filters[i],name=name+'_dconv'+str(i)))
            self.priors.append(PriorBlockmean(num_filters_prior[i],name=name+'prior'+str(i)))

    def call(self,inputs,blocks,is_training=True):
        x=inputs
        prior=[]
        for i in range(self.num_layers):
            p=self.priors[i](x)
            prior.append(p)
            #p=prob_function(p)
            #p=self.prob_function(p)
            if i!=self.num_layers-1:
                x=tf.concat([x,p],axis=-1)
                x=self.deconvs[i](x,blocks[i],is_training=is_training)
        return prior





class DecoderWithPriorvarBlockPosterior(tf.keras.layers.Layer):
    """decoder of the network with prior block in the variance branch in Posterior"""

    def __init__(self,num_layers,num_filters,num_filters_prior,name=None):
        super(DecoderWithPriorvarBlockPosterior,self).__init__(name=name)
        self.num_layers=num_layers
        self.num_filters=num_filters
        self.num_filters_prior=num_filters_prior
        self.deconvs=[]
        self.priors=[]
        #
        #self.prob_function=Prob()
        for i in range(num_layers):
            self.deconvs.append(DeConvBlock(num_filters[i],name=name+'_dconv'+str(i)))
            self.priors.append(PriorBlockvar(num_filters_prior[i],name=name+'prior'+str(i)))

    def call(self,inputs,blocks,is_training=True):
        x=inputs
        prior=[]
        for i in range(self.num_layers):
            p=self.priors[i](x)
            prior.append(p)
            #p=prob_function(p)
            #p=self.prob_function(p)
            if i!=self.num_layers-1:
                x=tf.concat([x,p],axis=-1)
                x=self.deconvs[i](x,blocks[i],is_training=is_training)
        return prior


class DecoderWithPriorBlockPosterior(tf.keras.layers.Layer):
    """decoder of the network with prior block in the mean branch in Posterior"""

    def __init__(self,num_layers,num_filters,num_filters_prior,name=None):
        super(DecoderWithPriorBlockPosterior,self).__init__(name=name)
        self.num_layers=num_layers
        self.num_filters=num_filters
        self.num_filters_prior=num_filters_prior
        self.prob=Prob()
        #
        #self.prob_function=Prob()
        self.mean_decoder=DecoderWithPriormeanBlockPosterior(num_layers,num_filters,num_filters_prior,name=name+'_meandecoder')
        self.var_decoder=DecoderWithPriorvarBlockPosterior(num_layers,num_filters,num_filters_prior,name=name+'_vardecoder')

    def call(self,inputs,blocks,is_training=True):
        s=inputs.get_shape().as_list()[3]
        s=(s-1)//2
        x=inputs[:,:,:,0:1]
        m=inputs[:,:,:,1:1+s]
        v=inputs[:,:,:,1+s:]
        x1=tf.concat([x,m],axis=-1)
        x2=tf.concat([x,v],axis=-1)
        prior1=self.mean_decoder(x1,blocks,is_training=is_training)
        prior2=self.var_decoder(x2,blocks,is_training=is_training)
        prior=[]
        prob=[]
        for i in range(len(prior1)):
            prior.append(tf.concat([prior1[i],prior2[i]],axis=-1))
            prob.append(self.prob(prior[i]))
        return prior,prob





class DecoderWithPriorBlock(tf.keras.layers.Layer):
    """decoder of the network with prior block"""

    def __init__(self,num_layers,num_filters,num_filters_prior,name=None):
        super(DecoderWithPriorBlock,self).__init__(name=name)
        self.num_layers=num_layers
        self.num_filters=num_filters
        self.num_filters_prior=num_filters_prior
        self.deconvs=[]
        self.priors=[]
        #
        self.prob_function=Prob()
        for i in range(num_layers):
            self.deconvs.append(DeConvBlock(num_filters[i],name=name+'_dconv'+str(i)))
            self.priors.append(PriorBlock(num_filters_prior[i],name=name+'_prior'+str(i)))

    def call(self,inputs,blocks,prob,is_training=True):
        x=inputs
        prior=[]
        for i in range(self.num_layers):
            p=self.priors[i](x)
            prior.append(p)
            x=tf.concat([x,prob[i]],axis=-1)
            x=self.deconvs[i](x,blocks[i],is_training=is_training)
        return x,prior

    def sample(self,x,blocks,is_training=False):
        for i in range(self.num_layers):
            p=self.priors[i](x)
            #prob=prob_function(p)
            prob=prob_function(p)
            x=tf.concat([x,prob],axis=-1)
            x=self.deconvs[i](x,blocks[i],is_training=is_training)
        return x


class Decoder(tf.keras.layers.Layer):
    """decoder of the network"""

    def __init__(self,num_layers,num_prior_layers,num_filters,num_filters_in_prior,num_filters_prior,name=None):
        """
        :param num_layers: number of layers in the non-prior part
        :param num_prior_layers: number of layers in the prior part
        :param num_filters: list of numbers of filters in different layers in non-prior part
        :param num_filters_in_prior: list of numbers of filters in different layers in non-prior part
        :param num_filters_prior: list of numbers of priors in different prior blocks
        :param name: name
        """
        super(Decoder,self).__init__(name=name)
        self.num_layers=num_layers
        self.num_filters_prior=num_filters_prior
        self.num_filters=num_filters
        self.num_filters_in_prior=num_filters_in_prior
        self.num_prior_layers=num_prior_layers
        self.prior_decode=DecoderWithPriorBlock(num_prior_layers,num_filters_in_prior,num_filters_prior,name=name+'_with_prior')
        self.tconvs=[]
        for i in range(num_layers):
            self.tconvs.append(DeConvBlock(num_filters[i],name=name+'_without_prior'))

    def call(self,inputs,b,prob,is_training=True):
        x=inputs
        x,prior=self.prior_decode(x,b[0:self.num_prior_layers],prob,is_training=is_training)
        for i in range(self.num_layers):
            x=self.tconvs[i](x,b[self.num_prior_layers+i],is_training=is_training)
        return x,prior

    def sample(self,x,b,is_training=False):
        x=self.prior_decode.sample(x,b[0:self.num_prior_layers],is_training=is_training)
        for i in range(self.num_layers):
            x=self.tconvs[i](x,b[self.num_prior_layers+i],is_training=is_training)
        return x


def kl_gauss(y_true,y_pred):
    s=y_true.get_shape().as_list()[3]
    mean_true=y_true[:,:,:,0:s//2]
    var_true=y_true[:,:,:,s//2:]
    mean_pred=y_pred[:,:,:,0:s//2]
    var_pred=y_pred[:,:,:,s//2:]
    first=math_ops.log(math_ops.divide(var_pred,var_true))
    second=math_ops.divide(var_true+K.square(mean_true-mean_pred),var_pred)
    loss=first+second-1
    loss=K.flatten(loss*0.5)
    loss=tf.reduce_mean(loss)
    return loss


'''
def kl_gauss(y_true,y_pred):
    s=y_true.get_shape().as_list()[3]
    mean_true=y_true[:,:,:,0:s//2]
    var_true=y_true[:,:,:,s//2:]
    mean_pred=y_pred[:,:,:,0:s//2]
    var_pred=y_pred[:,:,:,s//2:]
    first=var_pred-var_true
    second=math_ops.divide(K.exp(var_true)+K.square(mean_true-mean_pred),K.exp(var_pred))
    loss=first+second-1
    loss=K.flatten(loss*0.5)
    loss=tf.reduce_mean(loss)
    return loss




def kl_gauss(y_true,y_pred):
    loss=K.square(y_true-y_pred)
    loss=K.flatten(loss)
    loss=tf.reduce_mean(loss)
    return loss

'''

def weighted_mse(seg,y_pred):
    s=seg.get_shape().as_list()[3]
    y_true=seg[:,:,:,0:s//2]
    var=seg[:,:,:,s//2:]
    mask=K.abs(y_true-y_pred)<var
    mask=tf.cast(mask,dtype=tf.float32)
    maskk=array_ops.ones_like(y_true,dtype=y_true.dtype)
    mask=math_ops.multiply(mask,0.8)
    mask=maskk-mask
    zeros=math_ops.multiply(array_ops.ones_like(y_true,dtype=y_true.dtype),0.005)
    z=(y_true+y_pred)>zeros
    z=tf.cast(z,dtype=tf.float32)
    #z=math_ops.multiply(z,200)
    #print(z)
    loss=K.flatten(K.square(z*mask*(y_pred-y_true)))
    #kl=tf.keras.losses.KLDivergence()
    #loss=K.flatten(z*mask*y_true*K.log((y_true+0.005)/(y_pred+0.005)))
    return tf.reduce_mean(loss)


def temp_m(y_true,y_pred):
    return tf.reduce_max(y_pred)


class HierarchicalProbUNet(tf.keras.Model):
    def __init__(self,num_classes,num_layers,num_filters,num_prior_layers,num_filters_prior,name=None):
        super(HierarchicalProbUNet,self).__init__(name=name)
        self.num_classes=num_classes
        self.num_layers=num_layers
        self.num_filters=num_filters
        self.num_prior_layers=num_prior_layers
        self.num_filters_decoder=num_filters[::-1]
        self.num_filters_prior=num_filters_prior
        self.encoder=Encoder(num_layers,num_filters,name=name+'_encoder')
        self.encoder_post=Encoder(num_layers,num_filters,name=name+'_encoder_post')
        self.decoder=Decoder(num_layers-num_prior_layers-1,num_prior_layers,self.num_filters_decoder[num_prior_layers+1:],self.num_filters_decoder[1:num_prior_layers+1],num_filters_prior,name=name+'_decoder')
        self.decoder_post=DecoderWithPriorBlockPosterior(num_prior_layers,self.num_filters_decoder[1:num_prior_layers+1],num_filters_prior,name=name+'_decoder_post')
        self.conv=Conv2DFixedPadding(filters=num_classes,kernel_size=1,stride=1,name='conv_final')

    def call(self,inputs,is_training=True):
        x1=inputs[:,:,:,0:1]
        seg=inputs[:,:,:,1:]
        x2=tf.concat([x1,seg],axis=3)
        x1,b_list1=self.encoder(x1,is_training=is_training)
        b_list1=b_list1[0:-1]
        x2,b_list2=self.encoder_post(x2,is_training=is_training)
        b_list2=b_list2[0:-1]
        b_list1.reverse()
        b_list2.reverse()
        prior2,prob=self.decoder_post(x2,b_list2[0:self.num_prior_layers],is_training=is_training)
        x1,prior1=self.decoder(x1,b_list1,prob,is_training=is_training)
        x1=self.conv(x1)
        x1=tf.keras.activations.sigmoid(x1)
        #los=tf.keras.backend.zeros(1)
        #kl=tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
        for i in range(len(prior1)):
            self.add_metric(temp_m(prior2[i],prior1[i]),name='temp'+str(i),aggregation='mean')
            if i==0:
                #los=kl(prior2[i],prior1[i])
                #los=K.flatten(los)
                #los=tf.reduce_mean(los)
                los=kl_gauss(prior2[i],prior1[i])
                self.add_metric(los,name='kl'+str(i),aggregation='mean')
            #los=tf.keras.backend.sum(los,kl(prior2[i],prior1[i]))
            else:
                #temp=kl(prior2[i],prior1[i])
                temp=kl_gauss(prior2[i],prior1[i])
                self.add_metric(temp,name='kl'+str(i),aggregation='mean')
                #temp=K.flatten(temp)
                #temp=tf.reduce_mean(temp)
                los=math_ops.add(los,temp)
        #los=tf.keras.backend.sum(los,mse(x1,seg))
        #mse=tf.keras.losses.mean_squared_error(seg,x1)
        #mse=tf.reduce_mean(mse)
        #mse=tf.keras.losses.categorical_crossentropy(seg[:,:,:,0:self.num_classes],x1)
        mse=weighted_mse(seg,x1)
        #mse=K.flatten(mse)
        #mse=tf.reduce_mean(mse)
        #mse=tf.reduce_mean(mse)
        loss=math_ops.add(los,mse)

        #self.add_loss(loss)
        self.add_metric(los,name='kl',aggregation='mean')
        return x1

    def sample(self,x,is_training=False):
        x,b_list=self.encoder(x,is_training=is_training)
        b_list=b_list[0:-1]
        b_list.reverse()
        x=self.decoder.sample(x,b_list,is_training=is_training)
        x=self.conv(x)
        x=tf.keras.activations.sigmoid(x)
        return x

'''
def load_data(input_dir1,input_dir2):
    crop=[89,624,178,678]
    i=0
    xx=[]
    needles=[]
    artifacts=[]
    needles_var=[]
    artifacts_var=[]
    backgrounds=[]
    backgrounds_var=[]
    while os.path.exists(input_dir1+str(i)+'.png'):
        x=cv2.imread(input_dir1+str(i)+'.png')
        x=cv2.cvtColor(x[crop[0]:crop[1],crop[2]:crop[3],:],cv2.COLOR_BGR2GRAY)
        x=cv2.resize(x,(256,256))/255.0
        x=np.reshape(x,(256,256,1))
        xx.append(x)
        needle=cv2.imread(input_dir2+str(i)+'needle.png')
        artifact=cv2.imread(input_dir2+str(i)+'artifact.png')
        needle_var=cv2.imread(input_dir2+str(i)+'needle_std.png')
        artifact_var=cv2.imread(input_dir2+str(i)+'artifact_std.png')
        needle=cv2.cvtColor(needle,cv2.COLOR_BGR2GRAY)
        artifact=cv2.cvtColor(artifact,cv2.COLOR_BGR2GRAY)
        needle_var=cv2.cvtColor(needle_var,cv2.COLOR_BGR2GRAY)
        artifact_var=cv2.cvtColor(artifact_var,cv2.COLOR_BGR2GRAY)
        needle=np.reshape(needle,(256,256,1))/255.0
        artifact=np.reshape(artifact,(256,256,1))/255.0
        needle_var=np.reshape(needle_var,(256,256,1))/255.0
        artifact_var=np.reshape(artifact_var,(256,256,1))/255.0
        background=1-needle-artifact
        background_var=needle_var+artifact_var
        needles.append(needle)
        artifacts.append(artifact)
        needles_var.append(needle_var)
        artifacts_var.append(artifact_var)
        backgrounds.append(background)
        backgrounds_var.append(background_var)
        i+=1
    xx=np.array(xx)
    needles=np.array(needles)
    artifacts=np.array(artifacts)
    backgrounds=np.array(backgrounds)
    needles_var=np.array(needles_var)
    artifacts_var=np.array(artifacts_var)
    backgrounds_var=np.array(backgrounds_var)
    res=np.concatenate((needles,artifacts,needles_var,artifacts_var),axis=-1)
    return xx,res


def train():
    gamma_para=[0.9,1,1.1]
    gauss_para=[-1,0,0.5]
    x,y=load_data('../data/needle/training/','../data/needle/new_hpu_training_refined/')
    print(x.shape,y.shape)
    x_,y_=flip(x,y)
    x=np.concatenate((x,x_),axis=0)
    y=np.concatenate((y,y_),axis=0)
    out='test_model_second_refined/'
    model=HierarchicalProbUNet(2,5,[64,128,256,512,1024],3,[4,8,16],name='ProbUNet')

    #model.build(input_shape=(16,256,256,2))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01))
    #xx=np.zeros((16,256,256,2))
    #y=np.zeros((16,))

    for i in range(30):
        for p1 in gauss_para:
            for p2 in gamma_para:
                x_,y_=gamma(x,y,p2)
                x_,y_=gaussian_filter(x_,y_,p1)
                data=np.concatenate((x_,y_),axis=-1)
                model.fit(data,y_,epochs=1,batch_size=16)

        model.save_weights(out+str(i)+'.h5',save_format='h5')

def continue_train():
    gamma_para=[0.9,1,1.1]
    gauss_para=[-1,0,0.5]
    x,y=load_data('../data/needle/training/','../data/needle/new_hpu_training/')
    x_,y_=flip(x,y)
    x=np.concatenate((x,x_),axis=0)
    y=np.concatenate((y,y_),axis=0)
    out='test_model_second/'
    model=HierarchicalProbUNet(3,5,[64,128,256,512,1024],3,[4,8,16],name='ProbUNet')
    model.build(input_shape=(None,256,256,7))
    model.load_weights(out+'4.h5',by_name=True,skip_mismatch=True)
    #model.build(input_shape=(16,256,256,2))
    #model.compile()
    #xx=np.zeros((16,256,256,2))
    #y=np.zeros((16,))

    for i in range(5,30):
        for p1 in gauss_para:
            for p2 in gamma_para:
                x_,y_=gamma(x,y,p2)
                x_,y_=gaussian_filter(x_,y_,p1)
                data=np.concatenate((x_,y_),axis=-1)
                model.fit(data,y_,epochs=1,batch_size=16)

        model.save_weights(out+str(i)+'.h5',save_format='h5')



def load():
    xx,y=load_data('../data/needle/training/','../data/needle/new_hpu_training/')
    out='test_model_second/'
    out_f='../data/needle/final_result/'
    model=HierarchicalProbUNet(2,5,[64,128,256,512,1024],3,[4,8,16],name='ProbUNet')

    model.build(input_shape=(None,256,256,5))
    model.load_weights(out+'9.h5',by_name=True,skip_mismatch=True)
    ii=0
    flag=True
    while flag:
        if ii+20<=xx.shape[0]:
            x=xx[ii:ii+20,:,:,0:1]
        else:
            x=xx[ii:,:,:,0:1]
            flag=False
        #print(np.shape(data[0:16,:,:,:]))
        #z=model.predict(data[320:336,:,:,:],batch_size=16)
        z_=np.zeros((20,x.shape[0],256,256,2))
        for i in range(20):
            print(ii,'epoch:',i)
            z_[i,:,:,:,:]=model.sample(x,is_training=True)
        mean=np.mean(z_,axis=0)
        var=(np.std(z_,axis=0))
        for i in range(x.shape[0]):
            cv2.imwrite(out_f+str(ii+i)+'_needle.png',mean[i,:,:,0]*255)
            cv2.imwrite(out_f+str(ii+i)+'_artifact.png',mean[i,:,:,1]*255)
            ii+=20

def test():
    xx,y=load_test_data()
    out='test_model_second_refined/'
    model=HierarchicalProbUNet(2,5,[64,128,256,512,1024],3,[4,8,16],name='ProbUNet')
    model.build(input_shape=(None,256,256,5))
    model.load_weights(out+'11.h5',by_name=True,skip_mismatch=True)
    z_=np.zeros((5,xx.shape[0],256,256,2))
    for i in range(5):
        print('epoch:',i)
        z_[i,:,:,:,:]=model.sample(xx,is_training=True)
    mean=np.mean(z_,axis=0)
    var=np.std(z_,axis=0)
    dic={}
    dic=evaluate_batch(mean[:,:,:,0],mean[:,:,:,1],y,dic)
    final_evaluation(dic)
    visualize(xx,mean)

#test()
'''