import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.engine import network

from layers import *


class Encoder(network.Network):
    '''
        First encode input modality into prior judgement 
        of y_hat, then use contrastive and additive layer 
        to get statistics of modality specific quality 
        embedding variable s and true latent labels z.
    '''    
    def __init__(self,
                base_model,
                dense_params,
                activation,
                contr_params,
                addit_params,
                **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.base_model  = base_model
        self.dense_list  = []
        for params in dense_params[:-1]:
            self.dense_list.append(
                layers.Dense(params, activation=activation)
            )
        self.dense_list.append(
            layers.Dense(dense_params[-1], activation="softmax")
        )
        self.contr = ContrastiveLayer(**contr_params)
        self.addit = AdditiveLayer(**addit_params)

    def build(self, input_shapes):
        x_shp, y_shp = input_shapes
        x_in  = layers.Input(x_shp[1:])
        y_in  = layers.Input(y_shp[1:])
        y_hat = self.base_model(x_in) if self.base_model else x_in
        for dense in self.dense_list:
            y_hat = dense(y_hat)
        #quality variable
        q = self.contr([y_in, y_hat])
        #true latent label
        z = self.addit([y_in, y_hat])
        self._init_graph_network([x_in, y_in], [*q, z])
        super(Encoder, self).build(input_shapes)
            

class Decoder(network.Network):
    '''
        Give samples from variational distribution q 
        and z, reconstruct noisy labels through multi-
        layer perceptron.
    '''
    def __init__(self, dense_params, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.__init_weights()
        self.__init_layers(dense_params)

    def __init_weights(self):
        self.rho = self.add_weight(shape=(), name="rho", trainable=False)

    def __init_layers(self, dense_params):    
        self.dense_layers = []
        for params in dense_params[:-1]:
            self.dense_layers.append(
                layers.Dense(params, activation="relu")
            )
        self.dense_layers.append(
            layers.Dense(dense_params[-1], activation="softmax")
        )
        self.cent_loss = layers.Lambda(
            lambda x: K.mean(K.categorical_crossentropy(x[0], x[1]))
        )

    def build(self, input_shapes):
        q_shp, z_shp, y_shp = input_shapes
        q_in  = layers.Input(q_shp[1:])
        z_in  = layers.Input(z_shp[1:])
        y_in  = layers.Input(y_shp[1:])
        y_rec = layers.Concatenate()([q_in, z_in])
        for dense in self.dense_layers:
            y_rec = dense(y_rec)
        loss  = self.cent_loss([y_in, y_rec])
        self._init_graph_network([q_in, z_in, y_in], [y_rec, loss])
        super(Decoder, self).build(input_shapes)

    def call(self, inputs):
        inputs = generic_utils.to_list(inputs) 
        [y_rec, loss],_ = self._run_internal_graph(inputs)
        self.add_loss((1-self.rho)*loss)
        return y_rec


class BiModalClassifierNetwork(network.Network):
    def __init__(self,
                 input_shapes,
                 v_base_model,
                 a_base_model,
                 v_dense_params,
                 a_dense_params,
                 **kwargs):
        super(BiModalClassifierNetwork, self).__init__(**kwargs)
        assert(v_dense_params[-1] == a_dense_params[-1]), \
        "Dimensional of classes must be consistent!"
        self.__init_layers(v_base_model, a_base_model, v_dense_params, a_dense_params)
        self.build(input_shapes)

    def __init_layers(self, v_base, a_base, v_denses, a_denses):
        self.v_base_model = v_base
        self.a_base_model = a_base
        self.v_dense_list = []
        self.a_dense_list = []        
        for v_param in v_denses[:-1]:
            self.v_dense_list.append(
                layers.Dense(v_param , activation="relu")
            )
        self.v_dense_list.append(
            layers.Dense(v_denses[-1], activation="softmax")
        )
        for a_param in a_denses[:-1]:
            self.a_dense_list.append(
                layers.Dense(a_param , activation="relu")
            )
        self.a_dense_list.append(
            layers.Dense(a_denses[-1], activation="softmax")
        )
        self.poe_categ = ProductOfExpertCategorical(name="poe_categ", need_loss=False)        
        self.cent_loss = layers.Lambda(
            lambda x: K.mean(K.categorical_crossentropy(x[0], x[1]))
        )

    def build(self, input_shapes):
        v_shp, a_shp, y_shp = input_shapes
        v_in = layers.Input(shape=v_shp)
        a_in = layers.Input(shape=a_shp)    
        y_in = layers.Input(shape=y_shp)    
        v_x  = self.v_base_model(v_in) if self.v_base_model else v_in
        a_x  = self.a_base_model(a_in) if self.a_base_model else a_in
        for i, dense in enumerate(self.v_dense_list):
            v_x = dense(v_x)
            if i != len(self.v_dense_list)-1:
                v_x = layers.Dropout(0.5)(v_x)
        for i, dense in enumerate(self.a_dense_list):
            a_x = dense(a_x)
            if i != len(self.a_dense_list)-1:
                a_x = layers.Dropout(0.5)(a_x)            
        y_pred = self.poe_categ([v_x, a_x])
        cent_loss = self.cent_loss([y_in, y_pred])
        self._init_graph_network([v_in, a_in, y_in], [y_pred, cent_loss])
        super(BiModalClassifierNetwork, self).build(input_shapes)

    def call(self, inputs):
        inputs = generic_utils.to_list(inputs)
        [y_pred, cent_loss],_ = self._run_internal_graph(inputs)
        self.add_loss(cent_loss)
        return y_pred    


class BiModalDeepDenoiseModel(models.Model):
    def __init__(self,
                 input_shapes,
                 v_enc_args,
                 a_enc_args,
                 dec_args,
                 clf_args,
                 **kwargs):
        super(BiModalDeepDenoiseModel,self).__init__(**kwargs)
        self.__init_weights()
        self.__init_layers(v_enc_args, a_enc_args, dec_args, clf_args)
        self.build(input_shapes)

    def __init_weights(self):
        self.rho = self.add_weight(shape=(),
                                   name="rho",
                                   initializer="ones",
                                   trainable=False)

    def __init_layers(self, v_enc_args, a_enc_args, dec_args, clf_args):
        self.v_encoder = Encoder(**v_enc_args, name="v_enc")
        self.a_encoder = Encoder(**a_enc_args, name="a_enc")
        self.decoder   = Decoder(**dec_args ,  name="dec")        
        self.poe_gauss = ProductOfExpertGaussian   (name="poe_gauss")
        self.poe_categ = ProductOfExpertCategorical(name="poe_categ", need_loss=True)
        self.reparam_gauss = ReparameterizeGaussian   (name="rep_gauss")
        self.reparam_categ = ReparameterizeCategorical(name="rep_categ")
        self.bimodal_clf   = BiModalClassifierNetwork(**clf_args, name="clf")    

    def build(self, input_shapes):
        v_shp, a_shp, y_shp = input_shapes
        v_in  = layers.Input(shape=v_shp)
        a_in  = layers.Input(shape=a_shp)
        y_in  = layers.Input(shape=y_shp)
        #encode inputs into modality-specific quality vairables and true latent labels
        v_qmu, v_qstd, v_zprob = self.v_encoder([v_in, y_in])
        a_qmu, a_qstd, a_zprob = self.a_encoder([a_in, y_in])
        #product of expert 
        poe_q = self.poe_gauss([[v_qmu, v_qstd], [a_qmu, a_qstd]]) #with kl loss added
        poe_z = self.poe_categ([v_zprob, a_zprob, y_in])           #with kl loss added
        #reparameterization trick
        q_rep = self.reparam_gauss(poe_q)
        z_rep = self.reparam_categ(poe_z)
        #reconstruction of noisy labels
        y_rec = self.decoder([q_rep, z_rep, y_in]) #with reconstrcution loss added
        #with tf.device("/gpu:1"):
        #classification w.r.t clean label and add crossentropy loss
        z_pred = self.bimodal_clf([v_in, a_in, z_rep])
        self._init_graph_network ([v_in, a_in, y_in], [z_pred, y_rec])
        super(BiModalDeepDenoiseModel, self).build(input_shapes)

    def call(self, inputs):
        inputs = generic_utils.to_list(inputs)
        [z_pred, y_rec],_ = self._run_internal_graph(inputs)
        return z_pred


class ClassifierNetwork(network.Network):
    def __init__(self,
                 input_shapes,
                 base_model,
                 dense_params,
                 **kwargs):
        super(ClassifierNetwork, self).__init__(**kwargs)
        self.__init_layers(base_model, dense_params)
        self.build(input_shapes)

    def __init_layers(self, base_model, dense_params):
        self.base_model = base_model
        self.dense_list = []
        for param in dense_params[:-1]:
            self.dense_list.append(
                layers.Dense(param, activation="relu")
            )
        self.dense_list.append(
            layers.Dense(dense_params[-1], activation="softmax")
        )
        self.cent_loss = layers.Lambda(
            lambda x: K.mean(K.categorical_crossentropy(x[0], x[1]))
        )    

    def build(self, input_shapes):
        x_shp, y_shp = input_shapes
        x_in = layers.Input(shape=x_shp)
        y_in = layers.Input(shape=y_shp)    
        x_mid  = self.base_model(x_in) if self.base_model else x_in
        for i, dense in enumerate(self.dense_list):
            x_mid = dense(x_mid)
            if i != len(self.dense_list)-1:
                x_mid = layers.Dropout(0.5)(x_mid)
        y_pred = x_mid
        cent_loss = self.cent_loss([y_in, y_pred])
        self._init_graph_network([x_in, y_in], [y_pred, cent_loss])
        super(ClassifierNetwork, self).build(input_shapes)

    def call(self, inputs):
        inputs = generic_utils.to_list(inputs)
        [y_pred, cent_loss],_ = self._run_internal_graph(inputs)
        self.add_loss(cent_loss)
        return y_pred


class QualityEmbeddingNetwork(models.Model):
    def __init__(self,
                 input_shapes,
                 enc_args,
                 dec_args,
                 clf_args,
                 **kwargs):
        super(QualityEmbeddingNetwork,self).__init__(**kwargs)
        self.__init_weights()
        self.__init_layers(enc_args, dec_args, clf_args)
        self.build(input_shapes)

    def __init_weights(self):
        self.rho = self.add_weight(shape=(),
                                   name="rho",
                                   initializer="ones",
                                   trainable=False)

    def __init_layers(self, enc_args, dec_args, clf_args):
        self.encoder = Encoder(**enc_args, name="enc")
        self.decoder = Decoder(**dec_args, name="dec")
        self.single_gauss = SingleGaussian(name="poe_gauss")
        self.single_categ = SingleCategorical(name="poe_categ", need_loss=True)
        self.reparam_gauss = ReparameterizeGaussian   (name="rep_gauss")
        self.reparam_categ = ReparameterizeCategorical(name="rep_categ")
        self.bimodal_clf = ClassifierNetwork(**clf_args, name="clf")    

    def build(self, input_shapes):
        x_shp, y_shp = input_shapes
        x_in = layers.Input(shape=x_shp)
        y_in = layers.Input(shape=y_shp)
        #encode inputs into modality-specific quality vairables and true latent labels
        x_qmu, x_qstd, x_zprob = self.encoder([x_in, y_in])
        #reparameterization trick
        #product of expert 
        poe_q = self.single_gauss([x_qmu, x_qstd]) # with kl loss added
        poe_z = self.single_categ([x_zprob, y_in]) # with kl loss added
        #reparameterization trick
        q_rep = self.reparam_gauss(poe_q)
        z_rep = self.reparam_categ(poe_z)

        #reconstruction of noisy labels
        y_rec = self.decoder([q_rep, z_rep, y_in]) #with reconstrcution loss added
        #with tf.device("/gpu:1"):
        #classification w.r.t clean label and add crossentropy loss
        z_pred = self.bimodal_clf([x_in, z_rep])
        self._init_graph_network ([x_in, y_in], [z_pred, y_rec])
        super(QualityEmbeddingNetwork, self).build(input_shapes)

    def call(self, inputs):
        inputs = generic_utils.to_list(inputs)
        [z_pred, y_rec],_ = self._run_internal_graph(inputs)
        return z_pred