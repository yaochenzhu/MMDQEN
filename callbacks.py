import os

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import models
from tensorflow.keras import callbacks


def _name_var_dict():
	name_var_dict = {
		"kl_gauss"    : "self.model.get_layer(\"poe_gauss\").add_loss.lamb_kl",
		"kl_categ"    : "self.model.get_layer(\"poe_categ\").add_loss.lamb_kl",
		"ent_gauss"   : "self.model.get_layer(\"poe_gauss\").add_loss.lamb_ent",
		"ent_categ"   : "self.model.get_layer(\"poe_categ\").add_loss.lamb_ent",
		"rho_decoder" : "self.model.get_layer(\"decoder\").rho",	
		"rho_gauss"   : "self.model.get_layer(\"poe_gauss\").add_loss.rho",
		"rho_categ"   : "self.model.get_layer(\"poe_categ\").add_loss.rho",
		"temp"        : "self.model.get_layer(\"rep_categ\").temp",
		"lr"          : "self.model.optimizer.lr"
	}
	return name_var_dict


class AnnealParametersEpoch(callbacks.Callback):
	'''
		Anneal parameters according to some fixted
		schedule every time an epoch ends
	'''
	def __init__(self, name_schedule_dict, **kwargs):
		super(AnnealParametersEpoch, self).__init__(**kwargs)
		self.name_schedule_dict = name_schedule_dict

	def on_train_begin(self, epoch, logs=None):
		name_var_dict = _name_var_dict()
		self.var_schedule_dict = {
			name_var_dict[name]:schedule
				for name, schedule in self.name_schedule_dict.items()
		}

	def on_epoch_begin(self, epoch, logs=None):
		for var, schedule in self.var_schedule_dict.items():
			K.set_value(eval(var), schedule.value(epoch))

	def on_epoch_end(self, epoch, logs=None):
		print()
		print("|"+"-"*13+"|"+"-"*10+"|")	
		for var, _ in self.var_schedule_dict.items():
			print("|{:^13}|{:^10.5f}|".format(
				eval(var).name, K.get_value(eval(var))
			))	
		print("|"+"-"*13+"|"+"-"*10+"|")
		print()


class AnnealParametersBatch(callbacks.Callback):
	'''
		Anneal parametrs according to some fixed 
		schedule every time an iteration ends
	'''
	def __init__(self, name_schedule_dict, **kwargs):
		super(AnnealParametersBatch, self).__init__(**kwargs)
		self.name_schedule_dict = name_schedule_dict

	def on_train_begin(self, logs=None):
		self.iters = 0		
		name_var_dict = _name_var_dict()		
		self.var_schedule_dict = {
			name_var_dict[name]:schedule 
				for name, schedule in self.name_schedule_dict.items()
		}

	def on_batch_begin(self, batch, logs=None):
		self.iters += 1
		for var, schedule in self.var_schedule_dict.items():
			K.set_value(eval(var), schedule.value(self.iters))

	def on_epoch_end(self, epoch, logs=None):
		print("|"+"-"*13+"|"+"-"*10+"|")		
		for var, _ in self.var_schedule_dict.items():
			print("|{:^13}|{:^10.5f}|".format(
				eval(var).name, K.get_value(eval(var))
			))	
		print("|"+"-"*13+"|"+"-"*10+"|")
		print()


class EvalClassifierOnCleanLabels(callbacks.Callback):
	'''
		Get clean label from trained multimodal deep 
		denoise network to train an outside bimodal 
		classifier. Clf is trained c_steps every d_steps 
		training of denoise network. Model with highest
		validation accuracy is saved. 
	'''
	def __init__(self,
				 val_gen,
				 save_path,
				 **kwargs):
		super(EvalClassifierOnCleanLabels, self).__init__(**kwargs)
		self.val_gen   = val_gen
		self.save_path = save_path
		self.best_acc  = 0

	def on_epoch_end(self, epoch, logs=None):
		clf_val_gen = self.val_gen
		inputs      = self.model.bimodal_clf.inputs[:-1]
		outputs     = self.model.bimodal_clf.outputs[0]
		clf_model   = models.Model(inputs=inputs,outputs=outputs)
		clf_model.compile(optimizer="sgd",
						  loss="categorical_crossentropy",
						  metrics=["accuracy"])

		print("Begin eval clf model...")
		result  = clf_model.evaluate_generator(clf_val_gen)
		cur_acc = result[-1]
		if cur_acc > self.best_acc:
			self.best_acc = cur_acc
			clf_model.save_weights(self.save_path)
		print()
		print("Cur  accuracy: {}".format(cur_acc))
		print("Best accuracy: {}".format(self.best_acc))
		print("Clf model eval ends!")