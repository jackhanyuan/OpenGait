import os
from time import strftime, localtime
import numpy as np
from my_main import opt
from utils import get_msg_mgr, mkdir, config_loader
from evaluation.calculate_probability import calc_similarity

from .metric import mean_iou, cuda_dist, compute_ACC_mAP, evaluate_rank
from .re_rank import re_ranking


def de_diag(acc, each_angle=False):
    # Exclude identical-view cases
    dividend = acc.shape[1] - 1.
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / dividend
    if not each_angle:
        result = np.mean(result)
    return result


def cross_view_gallery_evaluation(feature, label, seq_type, view, dataset, metric):
    '''More details can be found: More details can be found in 
        [A Comprehensive Study on the Evaluation of Silhouette-based Gait Recognition](https://ieeexplore.ieee.org/document/9928336).
    '''
    probe_seq_dict = {'CASIA-B': {'NM': ['nm-01'], 'BG': ['bg-01'], 'CL': ['cl-01']},
                      'OUMVLP': {'NM': ['00']}}

    gallery_seq_dict = {'CASIA-B': ['nm-02', 'bg-02', 'cl-02'],
                        'OUMVLP': ['01']}

    msg_mgr = get_msg_mgr()
    acc = {}
    mean_ap = {}
    view_list = sorted(np.unique(view))
    for (type_, probe_seq) in probe_seq_dict[dataset].items():
        acc[type_] = np.zeros(len(view_list)) - 1.
        mean_ap[type_] = np.zeros(len(view_list)) - 1.
        for (v1, probe_view) in enumerate(view_list):
            pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                view, probe_view)
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]
            gseq_mask = np.isin(seq_type, gallery_seq_dict[dataset])
            gallery_y = label[gseq_mask]
            gallery_x = feature[gseq_mask, :]
            dist = cuda_dist(probe_x, gallery_x, metric)
            eval_results = compute_ACC_mAP(
                dist.cpu().numpy(), probe_y, gallery_y, view[pseq_mask], view[gseq_mask])
            acc[type_][v1] = np.round(eval_results[0] * 100, 2)
            mean_ap[type_][v1] = np.round(eval_results[1] * 100, 2)

    result_dict = {}
    msg_mgr.log_info(
        '===Cross View Gallery Evaluation (Excluded identical-view cases)===')
    out_acc_str = "========= Rank@1 Acc =========\n"
    out_map_str = "============= mAP ============\n"
    for type_ in probe_seq_dict[dataset].keys():
        avg_acc = np.mean(acc[type_])
        avg_map = np.mean(mean_ap[type_])
        result_dict[f'scalar/test_accuracy/{type_}-Rank@1'] = avg_acc
        result_dict[f'scalar/test_accuracy/{type_}-mAP'] = avg_map
        out_acc_str += f"{type_}:\t{acc[type_]}, mean: {avg_acc:.2f}%\n"
        out_map_str += f"{type_}:\t{mean_ap[type_]}, mean: {avg_map:.2f}%\n"
    # msg_mgr.log_info(f'========= Rank@1 Acc =========')
    msg_mgr.log_info(f'{out_acc_str}')
    # msg_mgr.log_info(f'========= mAP =========')
    msg_mgr.log_info(f'{out_map_str}')
    return result_dict

# Modified From https://github.com/AbnerHqC/GaitSet/blob/master/model/utils/evaluator.py


def single_view_gallery_evaluation(feature, label, seq_type, view, dataset, metric):
    probe_seq_dict = {'CASIA-B': {'NM': ['nm-05', 'nm-06'], 'BG': ['bg-01', 'bg-02'], 'CL': ['cl-01', 'cl-02']},
                      'OUMVLP': {'NM': ['00']},
                      'CASIA-E': {'NM': ['H-scene2-nm-1', 'H-scene2-nm-2', 'L-scene2-nm-1', 'L-scene2-nm-2', 'H-scene3-nm-1', 'H-scene3-nm-2', 'L-scene3-nm-1', 'L-scene3-nm-2', 'H-scene3_s-nm-1', 'H-scene3_s-nm-2', 'L-scene3_s-nm-1', 'L-scene3_s-nm-2',],
                                  'BG': ['H-scene2-bg-1', 'H-scene2-bg-2', 'L-scene2-bg-1', 'L-scene2-bg-2', 'H-scene3-bg-1', 'H-scene3-bg-2', 'L-scene3-bg-1', 'L-scene3-bg-2', 'H-scene3_s-bg-1', 'H-scene3_s-bg-2', 'L-scene3_s-bg-1', 'L-scene3_s-bg-2'],
                                  'CL': ['H-scene2-cl-1', 'H-scene2-cl-2', 'L-scene2-cl-1', 'L-scene2-cl-2', 'H-scene3-cl-1', 'H-scene3-cl-2', 'L-scene3-cl-1', 'L-scene3-cl-2', 'H-scene3_s-cl-1', 'H-scene3_s-cl-2', 'L-scene3_s-cl-1', 'L-scene3_s-cl-2']
                                  }

                      }
    gallery_seq_dict = {'CASIA-B': ['nm-01', 'nm-02', 'nm-03', 'nm-04'],
                        'OUMVLP': ['01'],
                        'CASIA-E': ['H-scene1-nm-1', 'H-scene1-nm-2', 'L-scene1-nm-1', 'L-scene1-nm-2']}
    msg_mgr = get_msg_mgr()
    acc = {}
    view_list = sorted(np.unique(view))
    if dataset == 'CASIA-E':
        view_list.remove("270")
    view_num = len(view_list)
    num_rank = 1
    for (type_, probe_seq) in probe_seq_dict[dataset].items():
        acc[type_] = np.zeros((view_num, view_num)) - 1.
        for (v1, probe_view) in enumerate(view_list):
            pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                view, probe_view)
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]

            for (v2, gallery_view) in enumerate(view_list):
                gseq_mask = np.isin(seq_type, gallery_seq_dict[dataset]) & np.isin(
                    view, [gallery_view])
                gallery_y = label[gseq_mask]
                gallery_x = feature[gseq_mask, :]
                dist = cuda_dist(probe_x, gallery_x, metric)
                idx = dist.topk(num_rank, largest=False)[1].cpu().numpy()
                acc[type_][v1, v2] = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx], 1) > 0,
                                                     0) * 100 / dist.shape[0], 2)

    result_dict = {}
    msg_mgr.log_info('===Rank-1 (Exclude identical-view cases)===')
    out_str = ""
    for type_ in probe_seq_dict[dataset].keys():
        sub_acc = de_diag(acc[type_], each_angle=True)
        msg_mgr.log_info(f'{type_}: {sub_acc}')
        result_dict[f'scalar/test_accuracy/{type_}'] = np.mean(sub_acc)
        out_str += f"{type_}: {np.mean(sub_acc):.2f}%\t"
    msg_mgr.log_info(out_str)
    return result_dict


def evaluate_indoor_dataset(data, dataset, metric='euc', cross_view_gallery=False):
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    view = np.array(view)

    if dataset not in ('CASIA-B', 'OUMVLP', 'CASIA-E'):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    if cross_view_gallery:
        return cross_view_gallery_evaluation(
            feature, label, seq_type, view, dataset, metric)
    else:
        return single_view_gallery_evaluation(
            feature, label, seq_type, view, dataset, metric)


def evaluate_real_scene(data, dataset, metric='euc', rerank=False):
    msg_mgr = get_msg_mgr()
    feature, label, seq_type = data['embeddings'], data['labels'], data['types']
    label = np.array(label)
    seq_type = np.array(seq_type)

    gallery_seq_type = {'0001-1000': ['1', '2'],
                        "HID2021": ['0'], '0001-1000-test': ['0'],
                        'GREW': ['01'], 'TTG-200': ['1'],
                        'OUMVLP': ['01'],
                        'OutdoorGait': ['nm-01'],
                        'HID': ['01']
                        }
    probe_seq_type = {'0001-1000': ['3', '4', '5', '6'],
                      "HID2021": ['1'], '0001-1000-test': ['1'],
                      'GREW': ['02'], 'TTG-200': ['2', '3', '4', '5', '6'],
                      'OUMVLP': ['00'],
                      'OutdoorGait': ['bg-01', 'cl-01', 'nm-02', 'nm-03', 'bg-02', 'bg-03', 'cl-02', 'cl-03'],
                      'HID': ['00']
                      }

    num_rank = 20
    acc = np.zeros([num_rank]) - 1.
    
    if dataset in (probe_seq_type or gallery_seq_type):
        gseq_mask = np.isin(seq_type, gallery_seq_type[dataset])
        pseq_mask = np.isin(seq_type, probe_seq_type[dataset])
    else:
        gseq_mask = (label != "probe")
        pseq_mask = (label == "probe")

    gallery_x = feature[gseq_mask, :]
    gallery_y = label[gseq_mask]
    probe_x = feature[pseq_mask, :]
    probe_y = label[pseq_mask]
    probe_type = seq_type[pseq_mask]

    if dataset not in (probe_seq_type or gallery_seq_type):
        probe_y, probe_type = probe_type, probe_y

    if rerank:
        feat = np.concatenate([probe_x, gallery_x])
        dist = cuda_dist(feat, feat, metric).cpu().numpy()
        msg_mgr.log_info("Starting Re-ranking")
        dist = re_ranking(dist, probe_x.shape[0], k1=6, k2=6, lambda_value=0.3)
        
    else:
        dist = cuda_dist(probe_x, gallery_x, metric).cpu().numpy()

    idx = np.argsort(dist, axis=1)
    acc = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                          0) * 100 / dist.shape[0], 2)
    msg_mgr.log_info('==Rank-1==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[0])))
    msg_mgr.log_info('==Rank-5==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[4])))
    msg_mgr.log_info('==Rank-10==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[9])))
    msg_mgr.log_info('==Rank-20==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[19])))
    return {"scalar/test_accuracy/Rank-1": np.mean(acc[0]), "scalar/test_accuracy/Rank-5": np.mean(acc[4])}


def GREW_submission(data, dataset, metric='euc'):
    get_msg_mgr().log_info("Evaluating GREW")
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    view = np.array(view)
    gallery_seq_type = {'GREW': ['01', '02']}
    probe_seq_type = {'GREW': ['03']}
    gseq_mask = np.isin(seq_type, gallery_seq_type[dataset])
    gallery_x = feature[gseq_mask, :]
    gallery_y = label[gseq_mask]
    pseq_mask = np.isin(seq_type, probe_seq_type[dataset])
    probe_x = feature[pseq_mask, :]
    probe_y = view[pseq_mask]

    num_rank = 20
    dist = cuda_dist(probe_x, gallery_x, metric)
    idx = dist.topk(num_rank, largest=False)[1].cpu().numpy()

    save_path = os.path.join(
        "GREW_result/"+strftime('%Y-%m%d-%H%M%S', localtime())+".csv")
    mkdir("GREW_result")
    with open(save_path, "w") as f:
        f.write("videoId,rank1,rank2,rank3,rank4,rank5,rank6,rank7,rank8,rank9,rank10,rank11,rank12,rank13,rank14,rank15,rank16,rank17,rank18,rank19,rank20\n")
        for i in range(len(idx)):
            r_format = [int(idx) for idx in gallery_y[idx[i, 0:num_rank]]]
            output_row = '{}'+',{}'*num_rank+'\n'
            f.write(output_row.format(probe_y[i], *r_format))
        print("GREW result saved to {}/{}".format(os.getcwd(), save_path))
    return


def HID_submission(data, dataset, rerank=True, metric='euc'):
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info("Evaluating HID")
    feature, label, seq_type = data['embeddings'], data['labels'], data['views']
    label = np.array(label)
    seq_type = np.array(seq_type)
    probe_mask = (label == "probe")
    gallery_mask = (label != "probe")
    gallery_x = feature[gallery_mask, :]
    gallery_y = label[gallery_mask]
    probe_x = feature[probe_mask, :]
    probe_y = seq_type[probe_mask]
    if rerank:
        feat = np.concatenate([probe_x, gallery_x])
        dist = cuda_dist(feat, feat, metric).cpu().numpy()
        msg_mgr.log_info("Starting Re-ranking")
        re_rank = re_ranking(dist, probe_x.shape[0], k1=6, k2=6, lambda_value=0.3)
        idx = np.argsort(re_rank, axis=1)
    else:
        dist = cuda_dist(probe_x, gallery_x, metric)
        idx = dist.cpu().sort(1)[1].numpy()

    save_path = os.path.join(
        "HID_result/"+strftime('%Y-%m%d-%H%M%S', localtime())+".csv")
    mkdir("HID_result")
    with open(save_path, "w") as f:
        f.write("videoID,label\n")
        for i in range(len(idx)):
            f.write("{},{}\n".format(probe_y[i], gallery_y[idx[i, 0]]))
        print("HID result saved to {}/{}".format(os.getcwd(), save_path))
    return


def evaluate_segmentation(data, dataset):
    labels = data['mask']
    pred = data['pred']
    miou = mean_iou(pred, labels)
    get_msg_mgr().log_info('mIOU: %.3f' % (miou.mean()))
    return {"scalar/test_accuracy/mIOU": miou}


def evaluate_Gait3D(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()

    features, labels, cams, time_seqs = data['embeddings'], data['labels'], data['types'], data['views']
    import json
    probe_sets = json.load(
        open('./datasets/Gait3D/Gait3D.json', 'rb'))['PROBE_SET']
    probe_mask = []
    for id, ty, sq in zip(labels, cams, time_seqs):
        if '-'.join([id, ty, sq]) in probe_sets:
            probe_mask.append(True)
        else:
            probe_mask.append(False)
    probe_mask = np.array(probe_mask)

    # probe_features = features[:probe_num]
    probe_features = features[probe_mask]
    # gallery_features = features[probe_num:]
    gallery_features = features[~probe_mask]
    # probe_lbls = np.asarray(labels[:probe_num])
    # gallery_lbls = np.asarray(labels[probe_num:])
    probe_lbls = np.asarray(labels)[probe_mask]
    gallery_lbls = np.asarray(labels)[~probe_mask]

    results = {}
    msg_mgr.log_info(f"The test metric you choose is {metric}.")
    dist = cuda_dist(probe_features, gallery_features, metric).cpu().numpy()
    cmc, all_AP, all_INP = evaluate_rank(dist, probe_lbls, gallery_lbls)

    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    for r in [1, 5, 10]:
        results['scalar/test_accuracy/Rank-{}'.format(r)] = cmc[r - 1] * 100
    results['scalar/test_accuracy/mAP'] = mAP * 100
    results['scalar/test_accuracy/mINP'] = mINP * 100

    # print_csv_format(dataset_name, results)
    msg_mgr.log_info(results)
    return results


def evaluate_dist(data, dataset, metric='euc', rerank=False):
    cfgs = config_loader(opt.cfgs)
    save_result = True

    msg_mgr = get_msg_mgr()
    msg_mgr.log_info("Evaluating Dist")
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    label_list = list(set(label))
    seq_type = np.array(seq_type)
    view = np.array(view)
    view_list = list(set(view))
    view_list.sort()

    probe_dict = {'OutdoorGait': ['nm-02', 'nm-03', 'bg-01', 'bg-02', 'bg-03', 'cl-01', 'cl-02', 'cl-03'],
                  'CASIA-B': ['nm-05', 'nm-06', 'bg-01', 'bg-02', 'cl-01', 'cl-02'],
                  'OUMVLP': ['00'],
                  'GREW': ['02'],
                  'HID': ['00']
                  }

    gallery_dict = {'OutdoorGait': ['nm-01'],
                    'CASIA-B': ['nm-01', 'nm-02', 'nm-03', 'nm-04'],
                    'OUMVLP': ['01'],
                    'GREW': ['01'],
                    'HID': ['01']
                    }
    gallery_view = view_list[::1]
    probe_view = gallery_view

    gallery_label = label_list[::1] if dataset == 'CASIA-B' else label_list[::1]
    probe_label = gallery_label
    if dataset in (probe_dict or gallery_dict):
        gallery_mask = np.isin(seq_type, gallery_dict[dataset]) & np.isin(view, gallery_view) & np.isin(label, gallery_label)
        probe_mask = np.isin(seq_type, probe_dict[dataset]) & np.isin(view, probe_view) & np.isin(label, probe_label)
    else:
        gallery_mask = (label != "probe")
        probe_mask = (label == "probe")

    gallery_feature = feature[gallery_mask, :]
    gallery_label = label[gallery_mask]
    gallery_seq_type = seq_type[gallery_mask]
    gallery_view = view[gallery_mask]

    probe_feature = feature[probe_mask, :]
    probe_label = label[probe_mask]
    probe_seq_type = seq_type[probe_mask]
    probe_view = view[probe_mask]

    if dataset not in (probe_dict or gallery_dict):
        probe_label, probe_seq_type = probe_seq_type, probe_label

    if rerank:
        msg_mgr.log_info("Starting Re-ranking")
        feat = np.concatenate([probe_feature, gallery_feature])
        dist = cuda_dist(feat, feat, metric).cpu().numpy()
        dist = re_ranking(dist, probe_feature.shape[0], k1=6, k2=6, lambda_value=0.3)
    else:
        dist = cuda_dist(probe_feature, gallery_feature, metric).cpu().numpy()

    import os
    import pandas as pd
    from time import strftime, localtime

    header = '-train_' + cfgs['data_cfg']['dataset_name'] + '-test_' + cfgs['data_cfg'][
        'test_dataset_name'] + '-restore_' + str(cfgs['evaluator_cfg']['restore_hint']) + '-rerank_' + str(
        rerank) + '-' + metric
    save_path = os.path.join("dist_result/" + "dist-" + strftime('%Y-%m%d-%H%M%S', localtime()) + header + ".csv")
    os.makedirs("dist_result", exist_ok=True)

    index = [probe_label[i] + '-' + probe_seq_type[i] + '-' + probe_view[i] for i in range(len(probe_label))]
    columns = [gallery_label[i] + '-' + gallery_seq_type[i] + '-' + gallery_view[i] for i in range(len(gallery_label))]

    dist_data = pd.DataFrame(dist, index=index, columns=columns)
    dist_data.to_csv(save_path, sep=',', float_format='%12.4f')

    msg_mgr.log_info("Dist result saved to {}/{}".format(os.getcwd(), save_path))

    if save_result:
        idx = np.argsort(dist, axis=1)
        save_path = os.path.join(
            "result/" + "res-" + strftime('%Y-%m%d-%H%M%S', localtime()) + header + ".csv")
        os.makedirs("result", exist_ok=True)
        with open(save_path, "w") as f:
            f.write("videoID,label\n")
            for i in range(len(idx)):
                f.write("{},{}\n".format(probe_label[i], gallery_label[idx[i, 0]]))
            msg_mgr.log_info("Result saved to {}/{}".format(os.getcwd(), save_path))

    if dataset in ['GREW', 'HID', 'OutdoorGait']:
        msg_mgr.log_info('Starting identification, rerank={}'.format(rerank))
        evaluate_real_scene(data, dataset, metric='euc', rerank=rerank)
        msg_mgr.log_info('Starting identification, rerank ={}'.format(not rerank))
        evaluate_real_scene(data, dataset, metric='euc', rerank=not rerank)

    else:
        msg_mgr.log_info('Starting identification, rerank=False')
        evaluate_indoor_dataset(data, dataset, metric='euc')

    return


def evaluate_similarity(data, dataset, metric='euc', rerank=False):
    rank_k = 6

    msg_mgr = get_msg_mgr()
    msg_mgr.log_info("Evaluating Dist")
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    seq_type = np.array(seq_type)

    gallery_mask = (label != "probe")
    probe_mask = (label == "probe")

    gallery_feature = feature[gallery_mask, :]
    gallery_label = label[gallery_mask]
    gallery_seq_type = seq_type[gallery_mask]

    probe_feature = feature[probe_mask, :]
    probe_label = seq_type[probe_mask]

    if rerank:
        print('Starting Re-ranking')
        feat = np.concatenate([probe_feature, gallery_feature])
        dist = cuda_dist(feat, feat, metric).cpu().numpy()
        dist = re_ranking(dist, probe_feature.shape[0], k1=6, k2=6, lambda_value=0.3)
        
    else:
        dist = cuda_dist(probe_feature, gallery_feature, metric).cpu().numpy()

    idx = np.argsort(dist, axis=1)
    simi = list(map(calc_similarity, dist))
    res = {}
    n = min(len(idx[0]), rank_k)  # 只要排名前rank_k的
    for i in range(len(idx)):
        label = {}
        for j in range(n):
            index = idx[i, j]
            g_label = str(gallery_label[index] + "-" + gallery_seq_type[index])
            label[g_label] = {"dist": np.float(dist[i][index]), "similarity": np.float(simi[i][index])}
        res[probe_label[i]] = label

    msg_mgr.log_info(res)

    # for i in res:
    #     print(i)
    #     for j in res[i]:
    #         print("\t{0:13}\t{1:5.3f}\t{2:6.3f}%".format(j, res[i][j]["dist"], res[i][j]["similarity"] * 100))

    return res
