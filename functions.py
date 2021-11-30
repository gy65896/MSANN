# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 15:03:36 2020

@author: Administrator
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
import torch.utils.model_zoo as model_zoo
import numpy as np

def concatenate_input_noise_map(img):

	N, C, H, W = img.size()
	dtype = img.type()
	sca = 2
	sca2 = sca*sca
	Cout = sca2*C
	Hout = H//sca
	Wout = W//sca
	idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

	# Fill the downsampled image with zeros
	downsampledfeatures = torch.cuda.FloatTensor(N, Cout, Hout, Wout).fill_(0)
	#downsamplednoisemap = torch.cuda.FloatTensor(N, Cout, Hout, Wout).fill_(0)

	# Populate output
	for idx in range(sca2):
		downsampledfeatures[:, idx:Cout:sca2, :, :] = img[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca]
		#downsamplednoisemap[:, idx:Cout:sca2, :, :] = noisemap[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca]      
	# concatenate de-interleaved mosaic with noise map
	#return torch.cat((downsampledfeatures,downsamplednoisemap), 1)
	return downsampledfeatures

class UpSampleFeaturesFunction(Function):

	@staticmethod
	def forward(ctx, input):
		N, Cin, Hin, Win = input.size()
		dtype = input.type()
		sca = 2
		sca2 = sca*sca
		Cout = Cin//sca2
		Hout = Hin*sca
		Wout = Win*sca
		idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

		assert (Cin%sca2 == 0), \
			'Invalid input dimensions: number of channels should be divisible by 4'

		result = torch.zeros((N, Cout, Hout, Wout)).type(dtype)
		for idx in range(sca2):
			result[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca] = \
				input[:, idx:Cin:sca2, :, :]

		return result

	@staticmethod
	def backward(ctx, grad_output):
		N, Cg_out, Hg_out, Wg_out = grad_output.size()
		dtype = grad_output.data.type()
		sca = 2
		sca2 = sca*sca
		Cg_in = sca2*Cg_out
		Hg_in = Hg_out//sca
		Wg_in = Wg_out//sca
		idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

		# Build output
		grad_input = torch.zeros((N, Cg_in, Hg_in, Wg_in)).type(dtype)
		# Populate output
		for idx in range(sca2):
			grad_input[:, idx:Cg_in:sca2, :, :] = \
				grad_output.data[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca]

		return Variable(grad_input)

# Alias functions
upsamplefeatures = UpSampleFeaturesFunction.apply


