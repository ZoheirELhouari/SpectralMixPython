import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score



def clustering(X, y, k, test_ids):   
    X_test = np.zeros((len(test_ids), X.shape[1]))
    y_test = [0 for i in range(0, len(test_ids))]


    for t in range(len(test_ids)):
        X_test[t] = X[int(test_ids[t])]
        y_test[t] = y[int(test_ids[t])]

        
    estimator = KMeans(n_clusters=k, n_init=10)

    NMI_list = []
    ARI_list = []

    for i in range(10):
        estimator.fit(X_test)
        y_pred = estimator.predict(X_test)
        nmi = normalized_mutual_info_score(y_test, y_pred, average_method='arithmetic')
        ari = adjusted_rand_score(y_test,y_pred)

        NMI_list.append(nmi)
        ARI_list.append(ari)


    mean = np.mean(NMI_list)
    std = np.std(NMI_list)
    mean_a = np.mean(ARI_list)
    std_a = np.std(ARI_list)

    print('[Clustering] NMI: {:.4f} | {:.4f}'.format(mean, std))
    print('[Clustering] ARI: {:.4f} | {:.4f}'.format(mean_a, std_a))
    return mean, mean_a;
