import LoadData
import DimReductionController
import torch
import csv
import numpy as np

Enzyme = [r'data\Enzyme_df.csv', r'data\Enzyme_tf.csv', r'data\Enzyme_Y.csv']
GPCR = [r'data\GPCR_df.csv', r'data\GPCR_tf.csv', r'data\GPCR_Y.csv']
IC = [r'data\IC_df.csv', r'data\IC_tf.csv', r'data\IC_Y.csv']
NR = [r'data\NR_df.csv', r'data\NR_tf.csv', r'data\NR_Y.csv']
drug_bank = [r'data\drug_bank_df.csv', r'data\drug_bank_tf.csv', r'data\drug_bank_Y.csv']
tabei = [r'data\tabei_df.csv', r'data\tabei_tf.csv', r'data\tabei_Y.csv']#1757
drug_bank_approv = [r'data\drug_bank_approv_df.csv', r'data\drug_bank_approve_tf.csv', r'data\drug_bank_approve_Y.csv']

f = open(r'c:\drugFeatureVectors.csv', 'w', newline='', encoding='utf-8')
f1 = open(r'c:\targetFeatureVectors.csv', 'w', newline='', encoding='utf-8')
f2 = open(r'c:\Y.csv', 'w', newline='', encoding='utf-8')

#drug_bank = [r'data\drug_bank_df.csv', r'data\drug_bank_tf.csv', r'data\drug_bank_Y.csv']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

drugFeatureVectors, targetFeatureVectors, Y = LoadData.load_data(drug_bank[0], drug_bank[1], drug_bank[2])
print(drugFeatureVectors.shape)
print(targetFeatureVectors.shape)

drugFeatureVectors1, targetFeatureVectors1 = DimReductionController.get_AutoEncoder_fea(drugFeatureVectors, targetFeatureVectors, Y, 100,  device, 100000)
drugFeatureVectors1 = drugFeatureVectors1.cpu()
targetFeatureVectors1 = targetFeatureVectors1.cpu()
Y = Y.cpu()
drugFeatureVectors1 = np.array(drugFeatureVectors1)
targetFeatureVectors1 = np.array(targetFeatureVectors1)
Y = np.array(Y)

print(drugFeatureVectors1.shape)
print(targetFeatureVectors1.shape)
print(Y.shape)
csv_writer = csv.writer(f)
for ii in range(drugFeatureVectors1.shape[0]):
 #   print(ii)
    csv_writer.writerow(drugFeatureVectors1[ii])

csv_writer = csv.writer(f1)
for ii in range(targetFeatureVectors1.shape[0]):
 #   print(ii)
    csv_writer.writerow(targetFeatureVectors1[ii])

csv_writer = csv.writer(f2)
for ii in range(Y.shape[0]):
#    print(ii)
    csv_writer.writerow(Y[ii])

f.close()
f1.close()
f2.close()

