import propy
import rdkit
from rdkit import Chem
import csv
from Bio import SeqIO
import pandas as pd
from propy import PyPro
import numpy as np
from propy.GetProteinFromUniprot import GetProteinSequence

drug_bank_save_drug = r'E:\DTI\data_src\drug_bank_drug.csv'
drug_bank_save_target = r'E:\DTI\data_src\drug_bank_target.csv'
drug_bank_save_target_feas = r'E:\DTI\data_src\drug_bank_target_feas_all.csv'
drug_bank_save_Y = r'E:\DTI\data_src\drug_bank_Y.csv'
aaa = r'D:\aaa.csv'


def find_string_in_array(strs, str):
    for ii in range(len(strs)):
        if str == strs[ii]:
            return ii

    return -1

def drugbank_sdf2csv(drug_sdf_path = '', save_path = ''):
    drugbank_drug = Chem.SDMolSupplier(r'E:\DTI\drug_bank_all\sdf\structures.sdf')
    drug_num = len(drugbank_drug)
    f = open(drug_bank_save_drug, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(f)
    for ii in range(drug_num):
        if not isinstance(drugbank_drug[ii], rdkit.Chem.rdchem.Mol):
            continue
        csv_writer.writerow([drugbank_drug[ii].GetProp('DRUGBANK_ID'), drugbank_drug[ii].GetProp('SMILES')])
    f.close()

def drug_bank_fasta2csv(target_path = '', save_path = ''):
    f = open(drug_bank_save_target, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(f)
    fa = r"E:\DTI\drug_bank_approved\fasta\protein.fasta"
    for seq_record in SeqIO.parse(fa, "fasta"):
       csv_writer.writerow([seq_record.id[16:23], seq_record.seq])
    f.close()

def drug_bank_ajmatrix2csv(drug_IDs, target_IDs, Y_path, save_path = ''):
    f = open(drug_bank_save_Y, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(f)
    Y = np.zeros((len(drug_IDs), len(target_IDs)), dtype=int)
    ajmatrix = pd.read_csv(Y_path).values
    #print(ajmatrix.shape)
    target_num, cols = ajmatrix.shape
    num = 0
    num1 = 0
    aa = []
    for ii in range(target_num):
        target_ID = ajmatrix[ii][5].strip()
        target_ID = target_ID[0:6]
        print(target_ID)
        #print(target_ID)
        #y_col_i = np.where(target_IDs == target_ID)
        y_col_i = find_string_in_array(target_IDs, target_ID)
        if y_col_i == -1:
            print(ii)
            print(target_ID)


       # print(y_col_i)
        #y_col_i = target_IDs.index(target_ID)
        inter_drug_IDs = ajmatrix[ii][12].split(';')
        #print( len(inter_drug_IDs))
        num = num + len(inter_drug_IDs)

        for jj in range(len(inter_drug_IDs)):
            drug_ID = inter_drug_IDs[jj].strip()
            #y_row_i = np.where(drug_IDs == drug_ID)
            y_row_i = find_string_in_array(drug_IDs, drug_ID)
            #y_row_i = drug_IDs.index(drug_ID)
            if y_row_i != -1:
                aa.append([y_row_i, y_col_i])
                Y[y_row_i, ii] = 1
                num1 = num1 + 1


    print(len(aa))
    #for ii in range(len(aa)):
        #Y[aa[ii][0]][aa[ii][1]] = 1

    print(sum(sum(Y)))
    csv_writer.writerows(Y)

def convet_drugbank2csv(drug_path, target_path, Y_path):
    drugs = pd.read_csv(drug_bank_save_drug,  header=None).values
    print(drugs.shape)
    targets = pd.read_csv(drug_bank_save_target,  header=None).values
    #print(targets[:,0])
    drugs_IDs = drugs[:, 0]
    #print(drugs_IDs.shape)
    targets_IDs = targets[:, 0]
   # print(targets_IDs.shape)
    #print(find_string_in_array(drugs_IDs, 'DB00303'))
    drug_bank_ajmatrix2csv(drugs_IDs, targets_IDs, Y_path)

def extract_feas_by_fasta():
    f = open(drug_bank_save_target_feas, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(f)
    fa = r"E:\DTI\drug_bank_all\fasta\protein.fasta"
    targets = pd.read_csv(drug_bank_save_target, header=None).values

    for ii in range(targets.shape[0]):
        print(ii)
        pseq = targets[ii][1]
        DesObject = PyPro.GetProDes(pseq)
        csv_writer.writerow(DesObject.GetALL().values())

       # print(type(aa))
        #DesObject = PyPro.GetProDes()
        #csv_writer.writerow(DesObject.GetALL().values())

def fasta2csv(target_path = '', save_path = ''):
    f = open(r"D:\uniprot_sprot.tet", 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(f)
    fa = r"F:\迅雷下载\uniprot_sprot.fasta\uniprot_sprot.fasta"
    for seq_record in SeqIO.parse(fa, "fasta"):
       csv_writer.writerow([seq_record.id[16:23], seq_record.seq])
    f.close()

'''
drug_path = r"E:\DTI\drug_bank_approved\fasta\protein.fasta"
target_path = r'E:\DTI\drug_bank_approved\sdf\structures.sdf'
Y_path = r'E:\DTI\drug_bank_approved\Y\all.csv'
#drugbank_sdf2csv()
#drug_bank_fasta2csv()
convet_drugbank2csv(drug_path, target_path, Y_path)
'''

#extract_feas_by_fasta()
#fasta2csv()
drugbank_sdf2csv()