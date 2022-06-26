import DTI_all_randoms_FS

#Enzyme = [r'/dataset/Enzyme_df.csv', r'/dataset/Enzyme_tf.csv', r'/dataset/Enzyme_Y.csv']
#GPCR = [r'/dataset/GPCR_df.csv', r'/dataset/GPCR_tf.csv', r'/dataset/GPCR_Y.csv']
#IC = [r'/dataset/IC_df.csv', r'/dataset/IC_tf.csv', r'/dataset/IC_Y.csv']
#NR = [r'/dataset/NR_df.csv', r'/dataset/NR_tf.csv', r'/dataset/NR_Y.csv']
Enzyme = [r'data2\Enzyme_df.csv', r'data2\Enzyme_tf.csv', r'data2\Enzyme_Y.csv']
GPCR = [r'data2\GPCR_df.csv', r'data2\GPCR_tf.csv', r'data2\GPCR_Y.csv']
IC = [r'data2\IC_df.csv', r'data2\IC_tf.csv', r'data2\IC_Y.csv']
NR = [r'data2\NR_df.csv', r'data2\NR_tf.csv', r'data2\NR_Y.csv']
#drug_bank = [r'/dataset/drug_bank_df.csv', r'/dataset/drug_bank_tf.csv', r'/dataset/drug_bank_Y.csv']
#tabei = [r'/dataset/tabei_df.csv', r'/dataset/tabei_tf.csv', r'/dataset/tabei_Y.csv']#1757
#drug_bank_approv = [r'/dataset/drug_bank_approv_df.csv', r'/dataset/drug_bank_approve_tf.csv', r'/dataset/drug_bank_approve_Y.csv']

datasets = []
#datasets.append(NR)
#datasets.append(GPCR)
#datasets.append(IC)
datasets.append(Enzyme)
#datasets.append(tabei)
#datasets.append(drug_bank)
#datasets.append(drug_bank_approv)
import warnings
warnings.filterwarnings('ignore')

#DR_types = ['GCNAutoEncode', 'DNNAutoEncode', 'SRC']
DR_types = ['SRC']
#DR_types = ['UDFS']
exp_types = ['CVP']
#exp_types = ['CVP']
model_types = ['RBMNet_2000', 'RBMNet_3', 'RBMNet_2000_d', 'RBMNet_3_d']#RBM_n_MS, GCN_3_MS
#model_types = ['RBMNet_3_d']#RBM_n_MS, GCN_3_MS
model_types = ['SKlearn_KBESt_FS3', 'SKlearn_KBESt_FS']#RBM_n_MS, GCN_3_MS
#model_types = ['RBMTansDTIES']
graph_types = ['knn_graph']

#hider_dims = [300, 500]
#drop_ratios = [0.5, 0.7]
#layer_UUs = [1, 3, 5, 7, 9, 11]
hider_dims = [300]
drop_ratios = [0.5]
layer_UUs = [2]
neg_ratios = [5]
is_SMOTEs = [False]
neg_ratio = neg_ratios[0]
if neg_ratio == 1:
    is_SMOTEs = [False]
print('test use all')
f = open("out.txt", "w")
for exp_type in exp_types:
    for DR_type in DR_types:
        for dataset in datasets:
            print(dataset)
            for model_type in model_types:
                for neg_ratio in neg_ratios:
                    for hider_dim in hider_dims:
                        for drop_ratio in drop_ratios:
                            for is_SMOTE in is_SMOTEs:     
                                for layer_UU in layer_UUs:
                                    print('负面样本比例:', neg_ratio, end='')
                                    print(',  实验类型:', exp_type, end='')
                                    print(',  降维类型:', DR_type, end='')
                                    print(',  深度学习类型:', model_type, end='')
                                    print(',  层数:', layer_UU, end='')
                                    print(',  隐藏层维度:', hider_dim, end='')
                                    print(',  drop比例:', drop_ratio, end='')
                                    print(',  使用SMOTE:', is_SMOTE)

                                    print('负面样本比例:', neg_ratio, end='', file=f)
                                    print(',  实验类型:', exp_type, end='', file=f)
                                    print(',  降维类型:', DR_type, end='', file=f)
                                    print(',  深度学习类型:', model_type, end='', file=f)
                                    print(',  层数:', layer_UU, end='', file=f)
                                    print(',  隐藏层维度:', hider_dim, end='', file=f)
                                    print(',  drop比例:', drop_ratio, end='', file=f)
                                    print(',  使用SMOTE:', is_SMOTE, file=f)
                                    DTI_all_randoms_FS.DTI_all_randoms(dataset[0], dataset[1], dataset[2], exp_type, layer_UU, hider_dim, drop_ratio, DR_type, model_type, 'knn_graph', neg_ratio, is_SMOTE, False, f)

f.close()