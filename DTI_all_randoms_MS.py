import DTI_one_random_MS

def DTI_all_randoms(df_path, tf_path, Y_path, exp_type, layer_UU, hider_dim, drop_ratio, DR_type, model_type, graph_type, neg_ratio, k, sub_ratio, is_SMOTE = False, is_gcn=False):
    random_states = [1000, 2000]
    mean_aucs = 0
    for ii in random_states:
        #mean_auc = DTI_one_random_MS.DTI_one_random(df_path, tf_path, Y_path, ii, exp_type, layer_UU, hider_dim, drop_ratio, DR_type, model_type, graph_type, neg_ratio, is_SMOTE, is_gcn)
        mean_auc = DTI_one_random_MS.DTI_one_random(df_path, tf_path, Y_path, ii, exp_type, layer_UU, hider_dim, drop_ratio, DR_type, model_type, graph_type, neg_ratio, k, sub_ratio, is_SMOTE,  is_gcn)
        mean_aucs = mean_aucs + mean_auc

    mean_aucs = mean_aucs / len(random_states)
    [aa, bb, cc] = mean_aucs.shape
    for ii in range(cc):
        for jj in range(bb):
            for kk in range(aa):
                print("%6.3f" % mean_aucs[kk][jj][ii], end=' ')
            print(' ')
        print(' ')