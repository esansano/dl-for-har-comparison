import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
from src.utils.dataimport import paths_dict

if __name__ == '__main__':

    x_train_file_name = os.path.join(paths_dict['uci-har'], 'train', 'X_train.txt')
    y_train_file_name = os.path.join(paths_dict['uci-har'], 'train', 'y_train.txt')
    u_train_file_name = os.path.join(paths_dict['uci-har'], 'train', 'subject_train.txt')
    x_test_file_name = os.path.join(paths_dict['uci-har'], 'test', 'X_test.txt')
    y_test_file_name = os.path.join(paths_dict['uci-har'], 'test', 'y_test.txt')
    u_test_file_name = os.path.join(paths_dict['uci-har'], 'test', 'subject_test.txt')

    x_tr = pd.read_csv(x_train_file_name, delim_whitespace=True).to_numpy(dtype=float)
    y_tr = pd.read_csv(y_train_file_name, delim_whitespace=True).to_numpy(dtype=int).ravel()
    u_tr = pd.read_csv(u_train_file_name, delim_whitespace=True).to_numpy(dtype=int).ravel()
    x_ts = pd.read_csv(x_test_file_name, delim_whitespace=True).to_numpy(dtype=float)
    y_ts = pd.read_csv(y_test_file_name, delim_whitespace=True).to_numpy(dtype=int).ravel()
    u_ts = pd.read_csv(u_test_file_name, delim_whitespace=True).to_numpy(dtype=int).ravel()

    print(x_tr.shape)
    print(y_tr.shape)
    print(u_tr.shape)
    print(x_ts.shape)
    print(y_ts.shape)
    print(u_ts.shape)

    classifiers = {
        'svc': SVC(gamma='scale', random_state=42),
        # '1nn': KNeighborsClassifier(n_neighbors=1),
        # '2nn': KNeighborsClassifier(n_neighbors=2),
        '3nn': KNeighborsClassifier(n_neighbors=3),
        'rfc': RandomForestClassifier(random_state=42)
    }

    def run_classifiers(x_train, y_train, x_test, y_test):
        results = []
        for name, cls in classifiers.items():
            cls.fit(x_train, y_train)
            y_hat = cls.predict(x_test)
            acc = accuracy_score(y_test, y_hat)
            f1 = f1_score(y_test, y_hat, average='weighted')
            print(f'acc: {name}: {acc:6.4f}')
            print(f'f1:  {name}: {f1:6.4f}')
            results.append(f1)
        return results

    run_classifiers(x_tr, y_tr, x_ts, y_ts)

    users = np.unique(u_tr)
    users_svc = []
    users_knn = []
    users_rfc = []
    for user in users:
        ind_tr = np.where(u_tr == user)
        ind_ts = np.where(u_tr != user)

        print(f'user: {user}')
        res = run_classifiers(x_tr[ind_tr], y_tr[ind_tr], x_tr[ind_ts], y_tr[ind_ts])
        users_svc.append(res[0])
        users_knn.append(res[1])
        users_rfc.append(res[2])

    print(np.mean(users_svc))
    print(np.mean(users_knn))
    print(np.mean(users_rfc))
