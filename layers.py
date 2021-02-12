import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow_probability import distributions as tfp

class ContrastiveLayer(layers.Layer):
	'''
		First, embed noisy label y and prior judgement y_hat 
		into the same subspace. Then, get their difference and 
		embed it into Gaussian space as quality variable s.
		The results are mean and log_std of quality variable s.
	'''	
	def __init__(self,
				 shared_dim,
				 quality_dim,
				 activation,
				 **kwargs):
		super(ContrastiveLayer, self).__init__(**kwargs)
		self.dense_share  = layers.Dense(shared_dim, 
										 activation=activation)
		self.dense_mean   = layers.Dense(quality_dim)
		self.dense_logstd = layers.Dense(quality_dim)
		self.sub  = layers.Subtract()		
		self.clip = layers.Lambda(lambda x:K.clip(x, -20, 2))
		self.exp  = layers.Lambda(lambda x:K.exp(x))

	def call(self, inputs):
		y, y_hat  = inputs
		y_emb     = self.dense_share(y)
		y_hat_emb = self.dense_share(y_hat)
		x         = self.sub([y_emb, y_hat_emb])
		mu        = self.dense_mean(x)
		log_std   = self.clip(self.dense_logstd(x))
		std       = self.exp(log_std)
		return [mu, std]


class AdditiveLayer(layers.Layer):
	'''
		Take weighted sum of noisy label y and prior judgement
		y_hat as true latent label z. z can be modeled as cate-
		gorical distribution thus the log prob is returned. 
	'''	
	def __init__(self,
				 shared_dim,
				 label_dim,
				 **kwargs):
		super(AdditiveLayer, self).__init__(**kwargs)
		self.dense_y     = layers.Dense(shared_dim)
		self.dense_y_hat = layers.Dense(shared_dim)
		self.dense_embed = layers.Dense(label_dim, activation="softmax")
		self.add     = layers.Add()

	def call(self, inputs):
		y, y_hat  = inputs
		y_emb     = self.dense_y(y)
		y_hat_emb = self.dense_y_hat(y_hat)
		x         = self.add([y_emb, y_hat_emb])
		prob      = self.dense_embed(x)
		return prob


class AddGaussianLoss(layers.Layer):
	'''
		Add weighted KL divergence loss between variational gaussian 
		distribution and standard normal distribution to total loss.
	'''
	def __init__(self, 
				 **kwargs):
		super(AddGaussianLoss, self).__init__(**kwargs)
		self.rho      = self.add_weight(shape=(),
										name="rho",
										trainable=False)				
		self.lamb_kl  = self.add_weight(shape=(), 
										name="lamb_kl", 
										trainable=False)
		self.lamb_ent = self.add_weight(shape=(),
										name="lamb_ent",
										trainable=False)

	def call(self, inputs):
		mu, std  = inputs
		#variational distribution
		var_dist = tfp.MultivariateNormalDiag(loc=mu, scale_diag=std)
		#prior standard normal distribution
		pri_dist = tfp.MultivariateNormalDiag(loc=K.zeros_like(mu), scale_diag=K.ones_like(std))
		ent_loss = K.mean(var_dist.entropy())		
		kl_loss  = K.mean(tfp.kl_divergence(var_dist, pri_dist))
		self.add_loss((1-self.rho)*(self.lamb_kl*kl_loss+self.lamb_ent*ent_loss))
		return kl_loss


class AddCategoricalLoss(layers.Layer):
	'''
		Add weighted KL divergence between variational categorical 
		distribution and uniform distribution to total loss.
	'''
	def __init__(self, 
				 **kwargs):
		super(AddCategoricalLoss, self).__init__(**kwargs)
		self.rho      = self.add_weight(shape=(),
										name="rho",
										trainable=False)		
		self.lamb_kl  = self.add_weight(shape=(),
										name="lamb_kl",
										trainable=False)
		self.lamb_ent = self.add_weight(shape=(),
										name="lamb_ent",
										trainable=False)	

	def call(self, inputs):
		prob, y   = inputs
		uni_prob  = K.ones_like(prob)/tf.cast(K.shape(prob)[-1], tf.float32)
		#variational distribution
		var_dist  = tfp.Categorical(prob)
		#prior uniform distribution
		pri_dist  = tfp.Categorical(uni_prob)
		ent_loss  = K.mean(var_dist.entropy())
		kl_loss   = K.mean(tfp.kl_divergence(var_dist, pri_dist))
		cent_loss = K.mean(K.categorical_crossentropy(y, prob))			
		self.add_loss(
			self.rho*cent_loss + \
			(1-self.rho)*(self.lamb_kl*kl_loss+self.lamb_ent*ent_loss)
		)
		return kl_loss


class ProductOfExpertGaussian(layers.Layer):
	'''
		get mu and std from product of gaussian distributions
	'''
	def __init__(self, **kwargs):
		super(ProductOfExpertGaussian, self).__init__(**kwargs)
		self.add_loss = AddGaussianLoss()	

	def call(self, inputs):
		v_stats, a_stats = inputs
		mu_list, std_list = zip(*[v_stats, a_stats])
		prec_list = [(1/(std**2)) for std in std_list]
		new_mu  = K.sum([mu*prec/K.sum(prec_list, axis=0) 
						for mu, prec in zip(mu_list, prec_list)], axis=0)
		new_std = K.sqrt(1/K.sum(prec_list, axis=0))
		self.add_loss([new_mu, new_std])
		return [new_mu, new_std]


class ProductOfExpertCategorical(layers.Layer):
	'''
		Get products of categorical distributions and renormalize
	'''
	def __init__(self, need_loss, **kwargs):
		super(ProductOfExpertCategorical, self).__init__(**kwargs)
		self.need_loss = need_loss
		if self.need_loss:
			self.add_loss = AddCategoricalLoss()

	def call(self, inputs):
		if self.need_loss:
			v_prob, a_prob, y = inputs
		else:
			v_prob, a_prob = inputs
		prob_list = [v_prob, a_prob]
		unn_prob = K.prod(prob_list, axis=0)
		new_prob = unn_prob/K.sum(unn_prob, axis=-1, keepdims=True)
		if self.need_loss:
			self.add_loss([new_prob, y])
		return new_prob


class ReparameterizeGaussian(layers.Layer):
	'''
		Continous reparameterization trick
	'''
	def __init__(self, **kwargs):
		super(ReparameterizeGaussian, self).__init__(**kwargs)

	def call(self, stats):
		mu, std = stats
		dist = tfp.MultivariateNormalDiag(loc=mu, scale_diag=std)
		return dist.sample()


class ReparameterizeCategorical(layers.Layer):
	'''
		Discrete reparameterization trick (Gumbel Max Trick)
		Temperature should be annealed to near zero gradually
	'''	
	def __init__(self, **kwargs):
		super(ReparameterizeCategorical, self).__init__(**kwargs)
		self.temp = self.add_weight(shape=(), 
									name="temp", 
									initializer="ones", 
									trainable=False)

	def call(self, prob):
		gumbel_dist   = tfp.Gumbel(0, 1)
		gumbel_sample = gumbel_dist.sample(K.shape(prob))
		categ_sample  = K.exp((K.log(prob)+gumbel_sample)/self.temp)
		categ_sample  = categ_sample/K.sum(categ_sample, axis=-1, keepdims=True)
		return categ_sample 


class SingleGaussian(layers.Layer):
	'''
		get mu and std from product of gaussian distributions
	'''
	def __init__(self, **kwargs):
		super(SingleGaussian, self).__init__(**kwargs)
		self.add_loss = AddGaussianLoss()	

	def call(self, inputs):
		self.add_loss(inputs)
		return inputs


class SingleCategorical(layers.Layer):
	'''
		Get products of categorical distributions and renormalize
	'''
	def __init__(self, need_loss, **kwargs):
		super(SingleCategorical, self).__init__(**kwargs)
		self.need_loss = need_loss
		if self.need_loss:
			self.add_loss = AddCategoricalLoss()

	def call(self, inputs):
		if self.need_loss:
			prob, y = inputs
			self.add_loss([prob, y])
			return prob
		else:
			return inputs