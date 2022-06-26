import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing



CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
				"U": 19, "T": 20, "W": 21,
				"V": 22, "Y": 23, "X": 24,
				"Z": 25 }

CHARPROTLEN = 25

CHARCANSMISET = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
			 ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
			 "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
			 "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
			 "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
			 "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
			 "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
			 "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
			 "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
			 "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
			 "t": 61, "y": 62}

CHARCANSMILEN = 62

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

def one_hot_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros((MAX_SMI_LEN, len(smi_ch_ind))) #+1

	for i, ch in enumerate(line[:MAX_SMI_LEN]):
		X[i, (smi_ch_ind[ch]-1)] = 1

	return X #.tolist()

def one_hot_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
	X = np.zeros((MAX_SEQ_LEN, len(smi_ch_ind)))
	for i, ch in enumerate(line[:MAX_SEQ_LEN]):
		X[i, (smi_ch_ind[ch])-1] = 1

	return X #.tolist()


def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros(MAX_SMI_LEN)
	for i, ch in enumerate(line[:MAX_SMI_LEN]): #	x, smi_ch_ind, y
		X[i] = smi_ch_ind[ch]

	return X #.tolist()

def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
	X = np.zeros(MAX_SEQ_LEN)

	for i, ch in enumerate(line[:MAX_SEQ_LEN]):
		X[i] = smi_ch_ind[ch]

	return X #.tolist()



class DataSet(object):
	def __init__(self, with_label=True):
		self.SEQLEN = 1000
		self.SMILEN = 100
		self.charseqset = CHARPROTSET
		self.charseqset_size = CHARPROTLEN

		self.charsmiset = CHARISOSMISET  ###HERE CAN BE EDITED
		self.charsmiset_size = CHARISOSMILEN
		self.with_label = with_label

	def load_smi_seq(self, drug_fea_path, target_fea_path, y_path):
		with_label = self.with_label
		df = pd.read_csv(drug_fea_path, header=None)
		tf = pd.read_csv(target_fea_path, header=None)
		Y = torch.from_numpy(np.array(pd.read_csv(y_path, header=None).values))

		if with_label:
			drugFeatureVectors = torch.zeros([df.values.shape[0], self.SMILEN])
		else:
			drugFeatureVectors = torch.zeros([df.values.shape[0], self.SMILEN, len(CHARISOSMISET)])
		for ii in range(df.values.shape[0]):
			aa = df.values[ii]
			aa = aa[0]
			if with_label:
				bb = label_smiles(aa, self.SMILEN, CHARISOSMISET)
				drugFeatureVectors[ii, :] = torch.tensor(torch.from_numpy(bb), dtype=torch.long)
			else:
				bb = one_hot_smiles(aa, self.SMILEN, CHARISOSMISET)
				drugFeatureVectors[ii, :, :] = torch.from_numpy(bb)


		if with_label:
			targetFeatureVectors = torch.zeros([tf.values.shape[0], self.SEQLEN])
		else:
			targetFeatureVectors = torch.zeros([tf.values.shape[0], self.SEQLEN, len(CHARPROTSET)])
		for ii in range(tf.values.shape[0]):
			aa = tf.values[ii]
			aa = aa[0]
			if with_label:
				bb = label_sequence(aa, self.SEQLEN, CHARPROTSET)
				targetFeatureVectors[ii, :] = torch.tensor(torch.from_numpy(bb), dtype=torch.long)
			else:
				bb = one_hot_sequence(aa, self.SEQLEN, CHARPROTSET)
				targetFeatureVectors[ii, :, :] = torch.from_numpy(bb)

		return drugFeatureVectors, targetFeatureVectors, Y

	def load_smiles(self, drug_fea_path):
		smiles = pd.read_csv(drug_fea_path, header=None).values
		return smiles

def load_data(drug_fea_path, target_fea_path, y_path):
    aa = np.array(pd.read_csv(drug_fea_path, header=None))
    aa = preprocessing.scale(aa)
    drugFeatureVectors = torch.from_numpy(aa)

    aa = np.array(pd.read_csv(target_fea_path, header=None))
    aa = preprocessing.scale(aa)
    targetFeatureVectors = torch.from_numpy(aa)

    Y = torch.from_numpy(np.array(pd.read_csv(y_path, header=None).values))
    drugFeatureVectors = torch.as_tensor(drugFeatureVectors, dtype=torch.float32)
    targetFeatureVectors = torch.as_tensor(targetFeatureVectors, dtype=torch.float32)
    Y = torch.as_tensor(Y, dtype=torch.long)
    return drugFeatureVectors, targetFeatureVectors, Y

def load_data_ID(drug_ID_path, target_ID_path):
    drug_IDs = np.array(pd.read_csv(drug_ID_path, header=None))
    target_IDs = np.array(pd.read_csv(target_ID_path, header=None))

    return drug_IDs, target_IDs