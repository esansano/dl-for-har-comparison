import os
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split

position_list = ['unknown', 'wrist', 'waist', 'chest', 'ankle', 'arm', 'pocket']
device_list = ['unknown', 'smartphone', 'smartwatch', 'imu']
PATH_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')

SEED = 2018

paths_dict = {'activemiles': os.path.join(PATH_DATA, 'activemiles'),
              'hhar': os.path.join(PATH_DATA, 'hhar'),
              'fusion': os.path.join(PATH_DATA, 'fusion'),
              'mhealth': os.path.join(PATH_DATA, 'mhealth'),
              'swell': os.path.join(PATH_DATA, 'swell'),
              'usc-had': os.path.join(PATH_DATA, 'usc-had'),
              'uci-har': os.path.join(PATH_DATA, 'uci-har'),
              'pamap2': os.path.join(PATH_DATA, 'pamap2'),
              'opportunity': os.path.join(PATH_DATA, 'opportunity'),
              'realworld': os.path.join(PATH_DATA, 'realworld')
              }

datasets_activities = {
    'activemiles': ['Walking', 'Cycling', 'Running', 'Standing', 'Public Transport', 'Casual Movement', 'No Activity'],
    'hhar': ['bike', 'sit', 'stand', 'walk', 'stairsup', 'stairsdown'],
    'mhealth': ['standing', 'sitting', 'laying', 'walking', 'stairs_up', 'bend forward', 'arms elevation',
                'crouching', 'cycling', 'jogging', 'running', 'jump front and back'],
    'fusion': ['walking', 'sitting', 'standing', 'jogging', 'biking', 'upstairs', 'downstairs'],
    'swell': ['Standing', 'Sitting', 'Walking', 'Downstairs', 'Upstairs', 'Running'],
    'uci-har': ['walking', 'upstairs', 'downstairs', 'sitting', 'standing', 'laying'],
    'usc-had': ['Walking Forward', 'Walking Left', 'Walking Right', 'Walking Upstairs',
                'Walking Downstairs', 'Running Forward', 'Jumping Up', 'Sitting', 'Standing',
                'Sleeping', 'Elevator Up', 'Elevator Down'],
    'pamap2': ['lying', 'sitting', 'standing', 'walking', 'running', 'cycling', 'Nordic walking',
               'ascending stairs', 'descending stairs', 'vacuum cleaning', 'ironing', 'rope jumping'],
    'opportunity': ['Stand', 'Walk', 'Sit', 'Lie'],
    'realworld': ['climbingdown', 'climbingup', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking']}


class Dataset:

    def __init__(self, x_acc_train, x_gyr_train, y_train, x_acc_test, x_gyr_test, y_test):
        self.x_acc_train = x_acc_train
        self.x_gyr_train = x_gyr_train
        self.y_train = y_train
        self.x_acc_test = x_acc_test
        self.x_gyr_test = x_gyr_test
        self.y_test = y_test


def build_activemiles(seq_length):
    data_file = os.path.join(paths_dict['activemiles'], 'ActiveMiles.txt')
    curr_activity = 'empty'
    with open(data_file) as f:
        x_acc = []
        x_gyr = []
        acc = []
        gyr = []
        y = []
        for line in f:
            if line[:2] == 'L>':
                if len(acc) >= 240 and len(gyr) >= 240:
                    act = datasets_activities['activemiles'].index(curr_activity)
                    acc = __downsample(np.array(acc), 250)  # 5 seconds * 50Hz
                    gyr = __downsample(np.array(gyr), 250)
                    for i in range(0, 250 - seq_length + 1, seq_length // 2):
                        x_acc.append(acc[i:(i + seq_length), :])
                        x_gyr.append(gyr[i:(i + seq_length), :])
                        # device, position, activity
                        y.append([device_list.index('smartphone'), position_list.index('unknown'), act])
                curr_activity = line[2:-1]
                acc = []
                gyr = []
            elif line[:2] == 'N>' or line[:2] == 'R>':
                curr_activity = 'empty'
                gyr = []
                acc = []
            elif line[:2] == 'A>' or line[:2] == 'G>':
                if curr_activity in datasets_activities['activemiles']:
                    m = line[2:]
                    if '>' not in m:
                        if line[:2] == 'A>':
                            acc.append([float(v) for v in m.split(',')])
                        else:
                            gyr.append([float(v) for v in m.split(',')])
                    else:
                        curr_activity = 'empty'
                        gyr = []
                        acc = []

    acc_data = np.array(x_acc, dtype=float)
    gyr_data = np.array(x_gyr, dtype=float)
    y = np.array(y)[:, 2]

    save_data(acc_data, gyr_data, y, 'activemiles', seq_length)

    print("ACTIVEMILES:")
    print("acc", acc_data.shape)
    print("gyr", gyr_data.shape)
    print("y", y.shape)


def build_hhar(seq_length):
    devices_dict = {'nexus4': device_list.index('smartphone'), 's3': device_list.index('smartphone'),
                    's3mini': device_list.index('smartphone'), 'samsungold': device_list.index('smartphone'),
                    'gear': device_list.index('smartwatch'), 'lgwatch': device_list.index('smartwatch')}

    acc_files = [os.path.join(paths_dict['hhar'], 'Phones_accelerometer.csv'),
                 os.path.join(paths_dict['hhar'], 'Watch_accelerometer.csv')]
    gyr_files = [os.path.join(paths_dict['hhar'], 'Phones_gyroscope.csv'),
                 os.path.join(paths_dict['hhar'], 'Watch_gyroscope.csv')]
    positions = [position_list.index('waist'), position_list.index('wrist')]

    x_acc = []
    x_gyr = []
    y = []
    for file_acc, file_gyr, position in zip(acc_files, gyr_files, positions):
        print(file_acc, file_gyr, position)
        acc_df = pd.read_csv(file_acc)
        gyr_df = pd.read_csv(file_gyr)

        acc_np = acc_df.drop(['Index', 'Arrival_Time', 'Device'], axis=1).values
        gyr_np = gyr_df.drop(['Index', 'Arrival_Time', 'Device'], axis=1).values
        acc_np = acc_np[acc_np[:, 6] != 'null']
        gyr_np = gyr_np[gyr_np[:, 6] != 'null']

        activities = np.unique(acc_np[:, 6])
        devices = np.unique(acc_np[:, 5])
        users = np.unique(acc_np[:, 4])
        print('activities:', activities)
        print('devices', devices)
        print('users', users)

        for user in users:
            for device in devices:
                for activity in activities:
                    uda_acc = acc_np[(acc_np[:, 4] == user) & (acc_np[:, 5] == device) & (acc_np[:, 6] == activity),
                                     0:4]
                    uda_gyr = gyr_np[(gyr_np[:, 4] == user) & (gyr_np[:, 5] == device) & (gyr_np[:, 6] == activity),
                                     0:4]
                    uda_acc = uda_acc[uda_acc[:, 0].argsort()]
                    uda_gyr = uda_gyr[uda_gyr[:, 0].argsort()]
                    a = datasets_activities['hhar'].index(activity)
                    d, p = devices_dict[device], position
                    index = 0
                    last_size = len(y)
                    while index < uda_acc.shape[0]:
                        acc_window = uda_acc[index:(index + seq_length), 0:4]
                        gyr_window = uda_gyr[index:(index + seq_length), 0:4]
                        if acc_window.shape[0] == seq_length and gyr_window.shape[0] == seq_length:
                            x_acc.append(acc_window[:, 1:4])
                            x_gyr.append(gyr_window[:, 1:4])
                            y.append([d, p, a])
                            index += acc_window.shape[0] // 2 + 1
                        else:
                            index += seq_length // 2 + 1

                    print('user: {}, device: {} ({}), activity: {} ({}), windows: {} ({})'
                          .format(user, device, d, activity, a, len(y) - last_size, len(y)))

    acc_data = np.array(x_acc, dtype=float)
    gyr_data = np.array(x_gyr, dtype=float)
    y = np.array(y)[:, 2]

    print("HHAR:")
    print("acc", acc_data.shape)
    print("gyr", gyr_data.shape)
    print("y", y.shape)

    save_data(acc_data, gyr_data, y, 'hhar', seq_length)


def build_mhealth(seq_length):
    ankle_columns_acc = [5, 6, 7]
    ankle_columns_gyr = [8, 9, 10]
    arm_columns_acc = [14, 15, 16]
    arm_columns_gyr = [17, 18, 19]
    data_files = ['mHealth_subject' + str(i + 1) + '.log' for i in range(10)]
    x_acc = []
    x_gyr = []
    y = []
    user_id = 0
    for data_file in data_files:
        file = os.path.join(paths_dict['mhealth'], data_file)
        user_id += 1
        data_np = pd.read_csv(file, sep='\t').values
        activities = range(1, len(datasets_activities['mhealth']) + 1)
        for activity in activities:
            act_np = data_np[data_np[:, 23] == activity, :]

            print('user: %d, device: %d, activity: %d, data length: %d' %
                  (user_id, device_list.index('imu'), activity, act_np.shape[0]))

            for index in range(0, act_np.shape[0] - seq_length + 1, seq_length // 2):
                x_acc.append(act_np[index:(index + seq_length), ankle_columns_acc])
                x_gyr.append(act_np[index:(index + seq_length), ankle_columns_gyr])
                y.append([device_list.index('imu'), position_list.index('ankle'), activity - 1])
                x_acc.append(act_np[index:(index + seq_length), arm_columns_acc])
                x_gyr.append(act_np[index:(index + seq_length), arm_columns_gyr])
                y.append([device_list.index('imu'), position_list.index('wrist'), activity - 1])

    acc_data = np.array(x_acc, dtype=float)
    gyr_data = np.array(x_gyr, dtype=float)
    y = np.array(y)[:, 2]

    print("MHEALTH:")
    print("acc", acc_data.shape)
    print("gyr", gyr_data.shape)
    print("y", y.shape)

    save_data(acc_data, gyr_data, y, 'mhealth', seq_length)


def build_fusion(seq_length):
    lpocket_columns_acc = [1, 2, 3]
    lpocket_columns_gyr = [7, 8, 9]
    rpocket_columns_acc = [15, 16, 17]
    rpocket_columns_gyr = [21, 22, 23]
    wrist_columns_acc = [29, 30, 31]
    wrist_columns_gyr = [35, 36, 37]
    arm_columns_acc = [43, 44, 45]
    arm_columns_gyr = [49, 50, 51]
    belt_columns_acc = [57, 58, 59]
    belt_columns_gyr = [63, 64, 65]
    activity_column = 69
    x_acc = []
    x_gyr = []
    y = []
    data_files = ['Participant_' + str(i + 1) + '.csv' for i in range(10)]
    user_id = 1
    for data_file in data_files:
        file = os.path.join(paths_dict['fusion'], data_file)
        data_np = pd.read_csv(file, skiprows=[1], header=None).values[1:, :]
        activities = datasets_activities['fusion']
        for activity in activities:
            act_np = data_np[data_np[:, activity_column] == activity, :]
            act = activities.index(activity)

            for index in range(0, act_np.shape[0] - seq_length + 1, seq_length // 2):
                # Left pocket
                x_acc.append(act_np[index:(index + seq_length), lpocket_columns_acc])
                x_gyr.append(act_np[index:(index + seq_length), lpocket_columns_gyr])
                y.append([device_list.index('smartphone'), position_list.index('pocket'), act])
                # Right pocket
                x_acc.append(act_np[index:(index + seq_length), rpocket_columns_acc])
                x_gyr.append(act_np[index:(index + seq_length), rpocket_columns_gyr])
                y.append([device_list.index('smartphone'), position_list.index('pocket'), act])
                # Wrist
                x_acc.append(act_np[index:(index + seq_length), wrist_columns_acc])
                x_gyr.append(act_np[index:(index + seq_length), wrist_columns_gyr])
                y.append([device_list.index('smartphone'), position_list.index('wrist'), act])
                # Wrist
                x_acc.append(act_np[index:(index + seq_length), arm_columns_acc])
                x_gyr.append(act_np[index:(index + seq_length), arm_columns_gyr])
                y.append([device_list.index('smartphone'), position_list.index('arm'), act])
                # Waist
                x_acc.append(act_np[index:(index + seq_length), belt_columns_acc])
                x_gyr.append(act_np[index:(index + seq_length), belt_columns_gyr])
                y.append([device_list.index('smartphone'), position_list.index('waist'), act])

            print('user: %d, device: %d, activity: %d, data length: %d' %
                  (user_id, device_list.index('smartphone'), act, act_np.shape[0]))

        user_id += 1

    acc_data = np.array(x_acc, dtype=float)
    gyr_data = np.array(x_gyr, dtype=float)
    y = np.array(y)[:, 2]

    print("FUSION:")
    print("acc", acc_data.shape)
    print("gyr", gyr_data.shape)
    print("y", y.shape)

    save_data(acc_data, gyr_data, y, 'fusion', seq_length)


def build_swell(seq_length):
    x_acc = []
    x_gyr = []
    y = []
    file_pos_dict = {'Arm.xlsx': 'arm', 'Belt.xlsx': 'waist', 'Pocket.xlsx': 'pocket', 'Wrist.xlsx': 'wrist'}
    for file_name in file_pos_dict.keys():
        data_np = pd.read_excel(file = os.path.join(paths_dict['swell'], file_name)).values
        activities = datasets_activities['swell']
        device = device_list.index('smartphone')
        position = position_list.index(file_pos_dict[file_name])
        for activity in activities:
            act_np = data_np[data_np[:, 10] == activity, :]
            act = activities.index(activity)
            print(act, act_np.shape)
            print('device: %d, position: %s, activity: %s' % (device, position, activity))
            index = 0
            while index < act_np.shape[0]:
                data_window = act_np[index:(index + seq_length), :]
                if data_window.shape[0] == seq_length:
                    acc_down = data_window[:, 1:4]
                    gyr_down = data_window[:, 4:7]
                    x_acc.append(acc_down)
                    x_gyr.append(gyr_down)
                    y.append([device, position, act])
                index += seq_length // 2 + 1

    acc_data = np.array(x_acc, dtype=float)
    gyr_data = np.array(x_gyr, dtype=float)
    y = np.array(y)[:, 2]

    print("SWELL:")
    print("acc", acc_data.shape)
    print("gyr", gyr_data.shape)
    print("y", y.shape)

    save_data(acc_data, gyr_data, y, 'swell', seq_length)


def build_uci_har(seq_length):
    x_acc = []
    x_gyr = []
    y = []
    key_file_name = os.path.join(paths_dict['uci-har'], 'RawData', 'labels.txt')
    activities = range(1, len(datasets_activities['uci-har']) + 1)
    for key_line in open(key_file_name):
        keys = key_line[:-1].split(' ')
        activity = int(keys[2])
        if activity not in activities:
            continue
        keys[0] = '0%s' % keys[0] if int(keys[0]) < 10 else keys[0]
        keys[1] = '0%s' % keys[1] if int(keys[1]) < 10 else keys[1]
        acc_file_name = '%s%sRawData%sacc_exp%s_user%s.txt' % (paths_dict['uci-har'], os.sep, os.sep, keys[0], keys[1])
        gyr_file_name = '%s%sRawData%sgyro_exp%s_user%s.txt' % (paths_dict['uci-har'], os.sep, os.sep, keys[0], keys[1])
        acc_np = pd.read_csv(acc_file_name, sep=' ', header=None).values[int(keys[3]):(int(keys[4]) + 1)]
        gyr_np = pd.read_csv(gyr_file_name, sep=' ', header=None).values[int(keys[3]):(int(keys[4]) + 1)]
        print(acc_np.shape)
        print(gyr_np.shape)

        for index in range(0, acc_np.shape[0] - seq_length + 1, seq_length // 2):
            x_acc.append(acc_np[index:(index + seq_length), :])
            x_gyr.append(gyr_np[index:(index + seq_length), :])
            y.append([device_list.index('smartphone'), position_list.index('waist'), activity - 1])

    acc_data = np.array(x_acc, dtype=float)
    gyr_data = np.array(x_gyr, dtype=float)
    y = np.array(y)[:, 2]

    print("UCI-HAR:")
    print("acc", acc_data.shape)
    print("gyr", gyr_data.shape)
    print("y", y.shape)

    save_data(acc_data, gyr_data, y, 'uci-har', seq_length)


def build_usc_had(seq_length):
    x_acc = []
    x_gyr = []
    y = []
    activities = range(1, 13)
    position = position_list.index('waist')
    device = device_list.index('imu')

    for subject in range(1, 15):
        for activity in activities:
            for trial in range(1, 6):
                data = sio.loadmat("%sSubject%d%sa%dt%d.mat" % (paths_dict['usc-had'], subject, os.sep, activity, trial))
                data = np.array(data['sensor_readings'])[::2]  # Only even rows -> sampling rate 50Hz
                for i in range(0, data.shape[0] - seq_length + 1, seq_length // 2):
                    i1 = i
                    i2 = i + seq_length
                    acc_xyz = data[i1:i2, 0:3]
                    gyr_xyz = data[i1:i2, 3:6]
                    x_acc.append(acc_xyz)
                    x_gyr.append(gyr_xyz)
                    y.append(np.array([device, position, activity - 1]))

    acc_data = np.array(x_acc, dtype=float)
    gyr_data = np.array(x_gyr, dtype=float)
    y = np.array(y)[:, 2]

    print("USC-HAD:")
    print("acc", acc_data.shape)
    print("gyr", gyr_data.shape)
    print("y", y.shape)

    save_data(acc_data, gyr_data, y, 'usc-had', seq_length)


def build_pamap2(seq_length):
    time_step = (seq_length / 100)  # 100 Hz
    margin = 0.05
    lower_ts = time_step * (1 - margin)
    upper_ts = time_step * (1 + margin)
    lower_len = int(seq_length * (1 - margin))
    activities = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
    device = device_list.index('imu')
    x_acc = []
    x_gyr = []
    y = []

    for user in range(1, 10):
        fn = '%s%s%d.dat' % (paths_dict['pamap2'], 'Protocol' + os.sep + 'subject10', user)
        data = pd.read_csv(fn, header=None, delim_whitespace=True)
        data.sort_values(by=0)
        data = data.values
        for activity in activities:
            data_act = data[data[:, 1] == activity, :]
            if data_act.size == 0:
                continue
            print('user {}, activity {}, measurements: {}'.format(user, activity, data_act.shape[0]))
            index = 0
            while index < data_act.shape[0]:
                it = data_act[index, 0]
                et = it + time_step
                t_data = data_act[(data_act[:, 0] >= it) & (data_act[:, 0] < et), :]
                ts = t_data[t_data.shape[0] - 1, 0] - t_data[0, 0]

                if t_data.shape[0] >= lower_len and lower_ts < ts < upper_ts:
                    act = activities.index(activity)

                    position = position_list.index('wrist')
                    acc1 = t_data[~np.isnan(t_data[:, 4:7]).any(axis=1), 4:7]
                    acc2 = t_data[~np.isnan(t_data[:, 7:10]).any(axis=1), 7:10]
                    gyr = t_data[~np.isnan(t_data[:, 10:13]).any(axis=1), 10:13]
                    if acc1.shape[0] >= lower_len and gyr.shape[0] >= lower_len:
                        x_acc.append(__downsample(acc1, seq_length))
                        x_gyr.append(__downsample(gyr, seq_length))
                        y.append([device, position, act])
                    if acc2.shape[0] >= lower_len and gyr.shape[0] >= lower_len:
                        x_acc.append(__downsample(acc2, seq_length))
                        x_gyr.append(__downsample(gyr, seq_length))
                        y.append([device, position, act])

                    position = position_list.index('chest')
                    acc1 = t_data[~np.isnan(t_data[:, 21:24]).any(axis=1), 21:24]
                    acc2 = t_data[~np.isnan(t_data[:, 24:27]).any(axis=1), 24:27]
                    gyr = t_data[~np.isnan(t_data[:, 27:30]).any(axis=1), 27:30]
                    if acc1.shape[0] >= lower_len and gyr.shape[0] >= lower_len:
                        x_acc.append(__downsample(acc1, seq_length))
                        x_gyr.append(__downsample(gyr, seq_length))
                        y.append([device, position, act])
                    if acc2.shape[0] >= lower_len and gyr.shape[0] >= lower_len:
                        x_acc.append(__downsample(acc2, seq_length))
                        x_gyr.append(__downsample(gyr, seq_length))
                        y.append([device, position, act])

                    position = position_list.index('ankle')
                    acc1 = t_data[~np.isnan(t_data[:, 38:41]).any(axis=1), 38:41]
                    acc2 = t_data[~np.isnan(t_data[:, 41:44]).any(axis=1), 41:44]
                    gyr = t_data[~np.isnan(t_data[:, 44:47]).any(axis=1), 44:47]
                    if acc1.shape[0] >= lower_len and gyr.shape[0] >= lower_len:
                        x_acc.append(__downsample(acc1, seq_length))
                        x_gyr.append(__downsample(gyr, seq_length))
                        y.append([device, position, act])
                    if acc2.shape[0] >= lower_len and gyr.shape[0] >= lower_len:
                        x_acc.append(__downsample(acc2, seq_length))
                        x_gyr.append(__downsample(gyr, seq_length))
                        y.append([device, position, act])

                    index += t_data.shape[0] // 2 + 1
                else:
                    index += t_data.shape[0]

    acc_data = np.array(x_acc, dtype=float)
    gyr_data = np.array(x_gyr, dtype=float)
    y = np.array(y)[:, 2]

    print("PAMAP2:")
    print("acc", acc_data.shape)
    print("gyr", gyr_data.shape)
    print("y", y.shape)

    save_data(acc_data, gyr_data, y, 'pamap2', seq_length)


def build_opportunity(seq_length):
    dir_path = os.path.join(paths_dict['opportunity'], 'dataset')

    columns = [[[37, 40], [40, 43]], [[50, 53], [53, 56]], [[63, 66], [66, 69]], [[76, 79], [79, 82]],
               [[89, 92], [92, 95]], [[108, 111], [111, 114]], [[124, 127], [127, 130]]]

    files = []
    x_acc = []
    x_gyr = []
    y = []

    for file in os.listdir(dir_path):
        if file.endswith(".dat"):
            files.append(os.path.join(dir_path, file))

    for file in files:
        data = pd.read_csv(file, header=None, delim_whitespace=True).values
        data_act = data[:, 243]

        n = data_act.shape[0]

        for i in range(0, n, seq_length // 2):
            act = data_act[i:(i + seq_length)]
            if len(np.unique(act)) == 1 and act[0] > 0:
                activity = [1, 2, 4, 5].index(act[0])
                for col in columns:
                    acc = data[i:(i + seq_length), col[0][0]:col[0][1]]
                    gyr = data[i:(i + seq_length), col[1][0]:col[1][1]]
                    if not (np.isnan(acc).any() or np.isnan(gyr).any()):
                        x_acc.append(acc)
                        x_gyr.append(gyr)
                        y.append(activity)

    acc_data = np.array(x_acc, dtype=float)
    gyr_data = np.array(x_gyr, dtype=float)
    y = np.array(y)

    print("OPPORTUNITY:")
    print("acc", acc_data.shape)
    print("gyr", gyr_data.shape)
    print("y", y.shape)

    save_data(acc_data, gyr_data, y, 'opportunity', seq_length)


def build_realworld(seq_length):
    subjects = range(1, 16)
    positions = ['chest', 'forearm', 'head', 'shin', 'thigh', 'upperarm', 'waist']
    x_acc = []
    x_gyr = []
    y = []

    for subject in subjects:
        for activity in datasets_activities['realworld']:
            acc_path = '{}proband{}{}data/acc_{}_csv{}'.format(paths_dict['realworld'],
                                                              subject, os.sep, os.sep, activity, os.sep)
            gyr_path = '{}proband{}{}data{}gyr_{}_csv{}'.format(paths_dict['realworld'],
                                                                subject, os.sep, os.sep, activity, os.sep)
            for position in positions:
                acc_file = '{}acc_{}_{}.csv'.format(acc_path, activity, position)
                gyr_file = '{}Gyroscope_{}_{}.csv'.format(gyr_path, activity, position)
                if os.path.isfile(acc_file) and os.path.isfile(gyr_file):
                    acc_data = pd.read_csv(acc_file, header=None).values[1:, 1:].astype(float)
                    gyr_data = pd.read_csv(gyr_file, header=None).values[1:, 1:].astype(float)
                    size = len(y)
                    start_time = acc_data[0, 0]
                    end_time = acc_data[-1, 0]
                    while start_time < end_time:
                        step_time = start_time + (1000 * seq_length / 50) * 1.05
                        acc_window = acc_data[(acc_data[:, 0] >= start_time) & (acc_data[:, 0] < step_time), :]
                        gyr_window = gyr_data[(gyr_data[:, 0] >= start_time) & (gyr_data[:, 0] < step_time), :]
                        # print(len(acc_window), len(gyr_window))
                        if (100 <= len(acc_window) < 120) and (100 <= len(gyr_window) < 120):
                            acc = acc_window[:seq_length, 1:]
                            gyr = gyr_window[:seq_length, 1:]
                            x_acc.append(acc)
                            x_gyr.append(gyr)
                            y.append(datasets_activities['realworld'].index(activity))

                        start_time += 1000 * seq_length / 50
                    print('user: {}, activity: {}, position: {}, windows: {}'.
                          format(subject, activity, position, len(y) - size))

    acc_data = np.array(x_acc, dtype=float)
    gyr_data = np.array(x_gyr, dtype=float)
    y = np.array(y)

    print("REALWORLD:")
    print("acc", acc_data.shape)
    print("gyr", gyr_data.shape)
    print("y", y.shape)

    save_data(acc_data, gyr_data, y, 'realworld', seq_length)


def __downsample(data, seq_length):
    step = data.shape[0] / seq_length
    index = 0.0
    indices = []
    for i in range(seq_length):
        indices.append(round(index))
        index += step
    return data[indices]


def __add_magnitude(data):
    data = data.astype(float)
    new_data = []
    for observation in data:
        m = np.sqrt(np.square(observation[:, 0]) + np.square(observation[:, 1]) + np.square(observation[:, 2]))
        new_data.append(np.c_[observation, m])
    return np.array(new_data)


def load_saved_data(dataset_name, seq_length=100, gyro=False, preprocess=None):
    acc_file_tr = '{}{}x_acc_{}_train.npy'.format(paths_dict[dataset_name], os.sep, seq_length)
    gyr_file_tr = '{}{}x_gyr_{}_train.npy'.format(paths_dict[dataset_name], os.sep, seq_length)
    y_file_tr = '{}{}y_{}_train.npy'.format(paths_dict[dataset_name], os.sep, seq_length)

    acc_file_ts = '{}{}x_acc_{}_test.npy'.format(paths_dict[dataset_name], os.sep, seq_length)
    gyr_file_ts = '{}{}x_gyr_{}_test.npy'.format(paths_dict[dataset_name], os.sep, seq_length)
    y_file_ts = '{}{}y_{}_test.npy'.format(paths_dict[dataset_name], os.sep, seq_length)

    print(acc_file_tr)

    if os.path.isfile(acc_file_tr) and os.path.isfile(gyr_file_tr) and os.path.isfile(y_file_tr) \
            and os.path.isfile(acc_file_ts) and os.path.isfile(gyr_file_ts) and os.path.isfile(y_file_ts):
        print('Loading data from ' + paths_dict[dataset_name])
        acc_data_train = np.load(acc_file_tr)
        acc_data_test = np.load(acc_file_ts)
        gyr_data_train = None
        gyr_data_test = None
        if gyro:
            gyr_data_train = np.load(gyr_file_tr)
            gyr_data_test = np.load(gyr_file_ts)

        y_train = np.load(y_file_tr)
        y_test = np.load(y_file_ts)

        acc_data_train, acc_data_test = preprocess_data(acc_data_train, acc_data_test, preprocess)
        gyr_data_train, gyr_data_test = preprocess_data(gyr_data_train, gyr_data_test, preprocess)

        return Dataset(acc_data_train, gyr_data_train, y_train, acc_data_test, gyr_data_test, y_test)
    return None


def save_data(acc_data, gyr_data, y, dataset_name, seq_length):
    # Training and validation data (80%)
    acc_file_tv = '{}{}x_acc_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length)
    gyr_file_tv = '{}{}x_gyr_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length)
    y_file_tv = '{}{}y_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length)

    # Test data (20%)
    acc_file_ts = '{}{}x_acc_{}_test'.format(paths_dict[dataset_name], os.sep, seq_length)
    gyr_file_ts = '{}{}x_gyr_{}_test'.format(paths_dict[dataset_name], os.sep, seq_length)
    y_file_ts = '{}{}y_{}_test'.format(paths_dict[dataset_name], os.sep, seq_length)

    indices = range(y.shape[0])

    ind_tv, ind_ts = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=y)

    y_train = y[ind_tv]
    y_test = y[ind_ts]

    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_test, return_counts=True))

    print('Saving data to' + paths_dict[dataset_name])
    np.save(acc_file_tv, acc_data[ind_tv])
    np.save(acc_file_ts, acc_data[ind_ts])
    if gyr_data is not None:
        np.save(gyr_file_tv, gyr_data[ind_tv])
        np.save(gyr_file_ts, gyr_data[ind_ts])
    np.save(y_file_tv, y[ind_tv])
    np.save(y_file_ts, y[ind_ts])


def preprocess_data(train_data, test_data, preprocess=None):
    if preprocess['type'] == 'standardize':
        return __standardize(train_data, test_data)
    elif preprocess['type'] == 'scale':
        return __scale(train_data, test_data)
    else:
        return train_data, test_data


def __scale(train_data, test_data):
    print('scaling data...')
    data_max = max(max(np.max(train_data), abs(np.min(train_data))), max(np.max(test_data), abs(np.min(test_data))))
    return train_data / data_max, test_data / data_max


def __standardize(train_data, test_data):
    print('standardizing data...')
    merged_data = np.concatenate((train_data, test_data), axis=0)
    data_mean = np.mean(merged_data)
    data_std = np.std(merged_data)
    return (train_data - data_mean) / data_std, (test_data - data_mean) / data_std
