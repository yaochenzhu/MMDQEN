import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from skimage import io
from tensorflow import keras


class BimodalDenoiseDataGen(keras.utils.Sequence):
	'''
		Generate train/validation/test samples for our multimodal
		denoise network. Inputs are static images, spectrograms and 
		corresponding noisy labels. Outputs are noisy labels. In 
		order to decorrelate training samples, we randonly shuffle
		movie sequences, sequentially fetch sel_movie movie clips 
		from that sequence, then randomly select sel_frames frames
		from each moive clip. 
	'''
	def __init__(self,
				 label_file,
				 length_file,
				 sample_rate,
				 video_root,
				 audio_root,
				 video_shape,
				 audio_shape,
				 video_preproc,
				 audio_preproc,
				 sel_movies,
				 sel_frames,
				 n_classes,
				 affective_type,
				 ret_label_X=True,
				 ret_label_y=True):
		self.__parse_label_file (label_file , affective_type)
		self.__parse_length_file(length_file, sample_rate)
		self.file_list      =  list(self.label_dict.keys())
		self.video_root     =  video_root
		self.audio_root     =  audio_root
		self.video_preproc  =  video_preproc
		self.audio_preproc  =  audio_preproc
		self.sel_movies     =  sel_movies
		self.sel_frames     =  sel_frames
		self._video_shape   =  video_shape
		self._audio_shape   =  audio_shape
		self._n_classes     =  n_classes
		self._batch_size    =  self.sel_movies*self.sel_frames
		self.ret_label_X    =  ret_label_X
		self.ret_label_y    =  ret_label_y
		self.on_epoch_end()	

	def on_epoch_end(self):
		np.random.shuffle(self.file_list)

	def __parse_label_file(self, label_file, affective_type):
		label_table = pd.read_table(label_file)
		self.label_dict = dict(
			zip(
				label_table["name"],
				label_table["valenceClass"] if affective_type == "val"
				else label_table["arousalClass"]
		))

	def __parse_length_file(self, length_file, sample_rate):
		length_table = pd.read_table(length_file)
		self.length_dict = dict(
			zip(
				length_table["name"],
				[l//sample_rate for l in length_table["length"]]
		))	

	def __len__(self):
		num = len(self.label_dict)
		return num // self.sel_movies

	def __getitem__(self, i):
		batch_file_list = self.file_list[i*self.sel_movies:(i+1)*self.sel_movies]
		X, y = self._data_generator(batch_file_list)
		return X, y

	def _data_generator(self, batch_file_list):
		videos = np.zeros((self._batch_size, *self.video_shape), dtype=np.float32)
		audios = np.zeros((self._batch_size, *self.audio_shape), dtype=np.float32)
		labels = []
		for i, filename in enumerate(batch_file_list):
			length    = self.length_dict[filename]
			frame_idx = np.random.choice(length, self.sel_frames)
			for j, idx in enumerate(frame_idx):
				videos[i*self.sel_frames+j] = io.imread(
					Path(self.video_root)/"{}_{}.jpg".format(filename, idx)
				)
				audios[i*self.sel_frames+j] = np.load(
					Path(self.audio_root)/"{}_{}.npy".format(filename, idx)
				)[..., None]
			labels += [self.label_dict[filename]]*self.sel_frames

		if self.video_preproc:
			videos = self.video_preproc(videos)
		if self.audio_preproc:
			audios = self.audio_preproc(audios)

		labels = keras.utils.to_categorical(labels, self._n_classes)
		X = [videos, audios]
		y = []
		if self.ret_label_X:
			X += [labels]
		if self.ret_label_y:
			y += [labels]
		return X, y

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def video_shape(self):
		return self._video_shape

	@property
	def audio_shape(self):
		return self._audio_shape

	@property
	def n_classes(self):
		return self._n_classes


class BimodalClassifierDataGen(BimodalDenoiseDataGen):
	def __init__(self,
				 training,
				 denoise_model=None,				 
				 **kwargs):
		super(BimodalClassifierDataGen, self).__init__(**kwargs)
		self.training = training
		if self.training:
			assert denoise_model is not None, \
				"must specify denoise model in training mode!"
		self.denoise_model = denoise_model

	def __getitem__(self, i):
		batch_file_list = self.file_list[i*self.sel_movies:(i+1)*self.sel_movies]
		X, _ = self._data_generator(batch_file_list)
		#if self.training == True:
		#	y = self.denoise_model.predict(X)
		#else:
		y = X[-1]
		X = [X[0], X[1]]
		return X, y


class DenoiseDataGen(keras.utils.Sequence):
	def __init__(self,
				 label_file,
				 length_file,
				 sample_rate,
				 video_root,
				 audio_root,
				 video_shape,
				 audio_shape,
				 video_preproc,
				 audio_preproc,
				 sel_movies,
				 sel_frames,
				 n_classes,
				 affective_type,
				 modality,
				 ret_label_X=True,
				 ret_label_y=True):
		self.__parse_label_file (label_file , affective_type)
		self.__parse_length_file(length_file, sample_rate)
		self.file_list      =  list(self.label_dict.keys())
		self.video_root     =  video_root
		self.audio_root     =  audio_root
		self.video_preproc  =  video_preproc
		self.audio_preproc  =  audio_preproc
		self.sel_movies     =  sel_movies
		self.sel_frames     =  sel_frames
		self._video_shape   =  video_shape
		self._audio_shape   =  audio_shape
		self._n_classes     =  n_classes
		self._batch_size    =  self.sel_movies*self.sel_frames
		self.ret_label_X    =  ret_label_X
		self.ret_label_y    =  ret_label_y

		self.modality = modality
		assert modality in ["visual", "aural"]
		self.on_epoch_end()	

	def on_epoch_end(self):
		np.random.shuffle(self.file_list)

	def __parse_label_file(self, label_file, affective_type):
		label_table = pd.read_table(label_file)
		self.label_dict = dict(
			zip(
				label_table["name"],
				label_table["valenceClass"] if affective_type == "val"
				else label_table["arousalClass"]
		))

	def __parse_length_file(self, length_file, sample_rate):
		length_table = pd.read_table(length_file)
		self.length_dict = dict(
			zip(
				length_table["name"],
				[l//sample_rate for l in length_table["length"]]
		))	

	def __len__(self):
		num = len(self.label_dict)
		return num // self.sel_movies

	def __getitem__(self, i):
		batch_file_list = self.file_list[i*self.sel_movies:(i+1)*self.sel_movies]
		X, y = self._data_generator(batch_file_list)
		return X, y

	def _data_generator(self, batch_file_list):
		videos = np.zeros((self._batch_size, *self.video_shape), dtype=np.float32)
		audios = np.zeros((self._batch_size, *self.audio_shape), dtype=np.float32)
		labels = []
		for i, filename in enumerate(batch_file_list):
			length = self.length_dict[filename]
			frame_idx = np.random.choice(length, self.sel_frames)
			if self.modality == "visual":
				for j, idx in enumerate(frame_idx):
					videos[i*self.sel_frames+j] = io.imread(
						Path(self.video_root)/"{}_{}.jpg".format(filename, idx)
					)
				labels += [self.label_dict[filename]]*self.sel_frames
			elif self.modality == "aural":
				for j, idx in enumerate(frame_idx):			
					audios[i*self.sel_frames+j] = np.load(
						Path(self.audio_root)/"{}_{}.npy".format(filename, idx)
					)[..., None]
				labels += [self.label_dict[filename]]*self.sel_frames


		if self.video_preproc and self.modality == "visual":
			videos = self.video_preproc(videos)
		if self.audio_preproc and self.modality == "aural":
			audios = self.audio_preproc(audios)

		labels = keras.utils.to_categorical(labels, self._n_classes)
		X = [videos] if self.modality == "visual" else [audios]
		y = []
		if self.ret_label_X:
			X += [labels]
		if self.ret_label_y:
			y += [labels]
		return X, y

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def video_shape(self):
		return self._video_shape

	@property
	def audio_shape(self):
		return self._audio_shape

	@property
	def n_classes(self):
		return self._n_classes


class ClassifierDataGen(DenoiseDataGen):
	def __init__(self,
				 training,
				 denoise_model=None,				 
				 **kwargs):
		super(ClassifierDataGen, self).__init__(**kwargs)
		self.training = training
		if self.training:
			assert denoise_model is not None, \
				"must specify denoise model in training mode!"
		self.denoise_model = denoise_model

	def __getitem__(self, i):
		batch_file_list = self.file_list[i*self.sel_movies:(i+1)*self.sel_movies]
		X, _ = self._data_generator(batch_file_list)
		#if self.training == True:
		#	y = self.denoise_model.predict(X)
		#else:
		y = X[-1]
		X = X[0]
		return X, y