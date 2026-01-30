import numpy as np
import random, itertools
import itertools
import math
random.seed(42)  # 固定随机种子

def sawtooth_wave(t, n):
    """Generate a single term of the sawtooth wave harmonic series."""
    return (t / np.pi) - np.floor(t / np.pi + 0.5)

def gen_periodic_data(periodic_type, load_data=False):
    random.seed(42)  # 固定随机种子
    if periodic_type == 'sin':
        def format_number(x: float, sig_digits=6, max_int_digits=2) -> str:
            """
            固定符号 + 小数点 + 总有效数字 sig_digits
            max_int_digits: 最大整数位数，超过则缩放或截断
            """
            sign = '+' if x >= 0 else '-'
            x_abs = abs(x)
        
            # 整数位
            int_part = int(x_abs)
            int_len = len(str(int_part))
            if int_len > max_int_digits:
                # 超过最大整数位 → 缩放
                int_len = max_int_digits
                frac_digits = sig_digits - max_int_digits
                x_abs /= 10**(len(str(int_part)) - max_int_digits)
            else:
                frac_digits = sig_digits - int_len
        
            # 格式化
            fmt = f"{sign}{{0:.{frac_digits}f}}"
            return fmt.format(x_abs)


        def generate_periodic_data(num_samples, num_periods=100, is_train = True):
            if is_train:
                t = np.linspace(-num_periods * np.pi, num_periods * np.pi, num_samples)
            else:
                t = np.linspace(-num_periods * 3 * np.pi, num_periods * 3 * np.pi, num_samples)
            data = np.sin(t)
            # data = t
            return [format_number(v) for v in t],  [format_number(v) for v in data]
        print(f'generate data from the {periodic_type} function')

        PERIOD = 6
        BATCHSIZE = 32
        NUMEPOCH = 451
        PRINTEPOCH = 5
        lr = 1e-5
        wd = 0.01

        t, data = generate_periodic_data(int(40000*PERIOD), PERIOD)
        t_test, data_test = generate_periodic_data(4000, PERIOD, is_train = False)

        # ---------- 去重逻辑 ----------
        train_set = set(t)
        filtered_t_test = []
        filtered_data_test = []

        for tt, dd in zip(t_test, data_test):
            if tt not in train_set:  # 只保留未出现在训练集中的样本
                filtered_t_test.append(tt)
                filtered_data_test.append(dd)

        t_test, data_test = filtered_t_test, filtered_data_test
        print(f"After filtering, test set size: {len(t_test)} (removed {4000 - len(t_test)} duplicates)")
        print("Sample train data points:", t[:5], data[:5])
        print("Sample test data points:", t_test[:5], data_test[:5])
        y_uper = 1.5
        y_lower = -1.5
    
    # ----------------------------------------------------------------------------------------------------------
    elif periodic_type == 'seq':

        def generate_discrete_periodic_data(num_samples=10000, seq_len=30, is_train=True):
            """
            生成 next token prediction 数据：
            X = seq[:-1], Y = seq[1:]
            训练与测试集使用不同的全排列。
            """
            # 生成所有 0~9 的全排列并打乱
            all_perms = list(itertools.permutations(range(10)))
            random.shuffle(all_perms)

            # 划分训练/测试
            if is_train:
                perms = all_perms[:num_samples]
            else:
                perms = all_perms[-num_samples:]

            X, Y = [], []
            for perm in perms:
                seq = "".join(map(str, perm)) * 3  # 重复三次
                X.append(seq[:-1])
                Y.append(seq[1:])
            return X, Y

        print(f'generate data from the {periodic_type} function')

        PERIOD = 10
        BATCHSIZE = 32
        NUMEPOCH = 51
        PRINTEPOCH = 5
        lr = 1e-5
        wd = 0.01

        # 数据生成
        t, data = generate_discrete_periodic_data(10000, seq_len=30, is_train=True)
        t_test, data_test = generate_discrete_periodic_data(1000, seq_len=30, is_train=False)

        y_uper = 1.5
        y_lower = -1.5

        print("Sample train data points:", t[:3], data[:3])
        print("Sample test data points:", t_test[:3], data_test[:3])

    elif periodic_type == 'seq2':
        def generate_discrete_periodic_data(num_samples=10000, seq_len=30, is_train=True):
            """
            训练集：每个样本随机选择低集合(0~4)或高集合(5~9)
            测试集：使用 0~9 全排列
            """
            X, Y = [], []
        
            if is_train:
                for _ in range(num_samples):
                    # 随机选择低集合或高集合
                    if random.randint(0, 1) == 0:
                        vocab = list(range(5))   # 0~4
                    else:
                        vocab = list(range(5, 10))  # 5~9
        
                    # 生成长度10的周期
                    seq_base = [random.choice(vocab) for _ in range(10)]
                    seq = seq_base * 3  # 重复三遍，长度30
        
                    X.append("".join(map(str, seq[:-1])))
                    Y.append("".join(map(str, seq[1:])))
            else:
                # 测试集使用 0~9 全排列
                import itertools
                full_perm = list(itertools.permutations(range(10)))
                random.shuffle(full_perm)
                for i in range(min(num_samples, len(full_perm))):
                    perm_seq = list(full_perm[i])
                    seq = perm_seq * 3  # 重复三遍
                    X.append("".join(map(str, seq[:-1])))
                    Y.append("".join(map(str, seq[1:])))
            
            return X, Y

        print(f'generate data from the {periodic_type} function')
    
        PERIOD = 10
        BATCHSIZE = 32
        NUMEPOCH = 51
        PRINTEPOCH = 5
        lr = 1e-5
        wd = 0.01
    
        # 数据生成
        t, data = generate_discrete_periodic_data(10000, seq_len=30, is_train=True)
        t_test, data_test = generate_discrete_periodic_data(1000, seq_len=30, is_train=False)
    
        y_uper = 1.5
        y_lower = -1.5
    
        print("Sample train data points:", t[:3], data[:3])
        print("Sample test data points:", t_test[:3], data_test[:3])
    
    elif periodic_type == 'seq3': # 训5~11，测12
        def generate_discrete_periodic_data(num_samples=10000, is_train=True):
            """
            训练集：每个样本随机选择低集合(0~4)或高集合(5~9)
            测试集：使用 0~9 全排列
            """
            X, Y, period_list = [], [], []
            for _ in range(num_samples):
                # 随机选择低集合或高集合
                vocab = list(range(10)) # list(range(5)) if random.randint(0,1) == 0 else list(range(5,10))
                # 随机选择周期长度
                if is_train:
                    period = random.randint(5, 11)
                else:
                    period = 12
                # 生成周期
                seq_base = [random.choice(vocab) for _ in range(period)]
                # 输入序列
                seq = seq_base * 3
                X.append("".join(map(str, seq[:-1])))
                Y.append("".join(map(str, seq[1:])))
                # 保存周期长度
                period_list.append(period)
            
            return X, Y, period_list

        print(f'generate data from the {periodic_type} function')
    
        PERIOD = 10
        BATCHSIZE = 32
        NUMEPOCH = 101
        PRINTEPOCH = 10
        lr = 1e-5
        wd = 0.01
    
        # 数据生成
        t, data, y_uper = generate_discrete_periodic_data(10000, is_train=True)
        t_test, data_test, y_lower = generate_discrete_periodic_data(1000, is_train=False)
    
        print("Sample train data points:", t[:3], data[:3])
        print("Sample test data points:", t_test[:3], data_test[:3])

    elif periodic_type == 'seq4': # 训5~11（除了8），测8
        def generate_discrete_periodic_data(num_samples=10000, is_train=True):
            """
            训练集：每个样本随机选择低集合(0~4)或高集合(5~9)
            测试集：使用 0~9 全排列
            """
            X, Y, period_list = [], [], []
            for _ in range(num_samples):
                vocab = list(range(10))
                if is_train:
                    period = random.choice([4, 5, 7, 8])
                else:
                    period = 6
                # 生成周期
                seq_base = [random.choice(vocab) for _ in range(period)]
                cut_point = period * 2 + random.randint(0, period - 1)
                # 输入序列
                seq = seq_base * 4
                X.append("".join(map(str, seq[:-1])))
                Y.append("".join(map(str, seq[1:])))
                # 保存周期长度
                period_list.append((period, cut_point))
            
            return X, Y, period_list
        print(f'generate data from the {periodic_type} function')
    
        PERIOD = 10
        BATCHSIZE = 32
        NUMEPOCH = 101
        PRINTEPOCH = 10
        lr = 1e-5
        wd = 0.01
    
        # 数据生成
        t, data, y_uper = generate_discrete_periodic_data(50000, is_train=True)
        t_test, data_test, y_lower = generate_discrete_periodic_data(3000, is_train=False)
        print("Sample train data points:", t[:3], data[:3], y_uper[:3])
        print("Sample test data points:", t_test[:3], data_test[:3])
    
    elif periodic_type == '2seq_conv':  # 两个周期序列逐位相加
        VOCAB = list(range(10))
        PERIOD_RANGE = list(range(2, 17))
        # 边缘区（二元组表示）
        BORDER_PAIRS = {
            # 左边缘
            *((p, q) for p in {2, 3} for q in PERIOD_RANGE),
            # 右边缘
            *((p, q) for p in {15, 16} for q in PERIOD_RANGE),
            # 上下边缘（保持视觉一致性）
            *((p, q) for q in {2, 3} for p in PERIOD_RANGE),
            *((p, q) for q in {15, 16} for p in PERIOD_RANGE),
        }

        # 空洞区（二元组表示）
        HOLLOW_SET = {8, 9, 10, 11}
        HOLLOW_PAIRS = {(p1, p2) for p1 in HOLLOW_SET for p2 in HOLLOW_SET}

        def generate_two_seq_add_data(num_samples=10000, is_train=True):
            """
            两个周期序列 → LCM → 相加 mod10 → 下一个 token 预测任务
            训练集：排除边缘区与空洞区
            测试集：从边缘区或空洞区抽样
            """
            X, Y, period_list = [], [], []

            for _ in range(num_samples):

                # -------------------------
                # 采样周期（只允许在此写逻辑）
                # -------------------------

                if is_train:
                    # 训练集：必须不在边缘区，也不在空洞区
                    while True:
                        p1, p2 = random.choices(PERIOD_RANGE, k=2)
                        pair = (p1, p2)
                        if (pair not in BORDER_PAIRS) and (pair not in HOLLOW_PAIRS):
                            break
                else:
                    p = random.random()
                    if p < 0.33:
                        # 边缘区
                        while True:
                            p1, p2 = random.choices(PERIOD_RANGE, k=2)
                            if (p1, p2) in BORDER_PAIRS:
                                break
                    elif p < 0.67:
                        # 空洞区 - 直接从集合中抽
                        p1, p2 = random.choice(list(HOLLOW_PAIRS))
                    else:
                        while True:
                            p1, p2 = random.choices(PERIOD_RANGE, k=2)
                            pair = (p1, p2)
                            if (pair not in BORDER_PAIRS) and (pair not in HOLLOW_PAIRS):
                                break


                # -------------------------
                # 周期序列生成
                # -------------------------

                seq1 = [random.choice(VOCAB) for _ in range(p1)]
                seq2 = [random.choice(VOCAB) for _ in range(p2)]

                # LCM
                lcm_len = 2 * abs(p1 * p2) // math.gcd(p1, p2)

                full_seq1 = (seq1 * (lcm_len // p1 + 1))[:lcm_len]
                full_seq2 = (seq2 * (lcm_len // p2 + 1))[:lcm_len]

                seq_conv = []
                for t in range(lcm_len):
                    val = 0
                    for k in range(lcm_len):
                        if t - k >= 0:
                            val += full_seq1[t - k] * full_seq2[k]
                    seq_conv.append(val % 10)

                seq_full = seq1 + ["+"] + seq2 + ["="] + seq_conv

                # 预测下一个 token
                X.append("".join(map(str, seq_full[:-1])))
                Y.append("".join(map(str, seq_full[1:])))

                cut_point = len(seq1) + len(seq2) + 2
                period_list.append((len(seq_full), cut_point))

            return X, Y, period_list
    
        print(f'generate data from the {periodic_type} function')
    
        PERIOD = 10
        BATCHSIZE = 64
        NUMEPOCH = 451
        PRINTEPOCH = 15
        lr = 1e-5
        wd = 0.01
    
        # 数据生成
        if load_data:
            pass
        else:
            t, data, y_uper = generate_two_seq_add_data(50000, is_train=True)
            t_test, data_test, y_lower = generate_two_seq_add_data(3000, is_train=False)
    
        print("Sample train data points:", t[:3], data[:3], y_uper[:3])
        print("Sample test data points:", t_test[:3], data_test[:3])

    elif periodic_type == '2seq_add':  # 两个周期序列逐位相加
        VOCAB = list(range(10))
        PERIOD_RANGE = list(range(2, 17))
        # 边缘区（二元组表示）
        BORDER_PAIRS = {
            # 左边缘
            *((p, q) for p in {2, 3} for q in PERIOD_RANGE),
            # 右边缘
            *((p, q) for p in {15, 16} for q in PERIOD_RANGE),
            # 上下边缘（保持视觉一致性）
            *((p, q) for q in {2, 3} for p in PERIOD_RANGE),
            *((p, q) for q in {15, 16} for p in PERIOD_RANGE),
        }

        # 空洞区（二元组表示）
        HOLLOW_SET = {8, 9, 10, 11}
        HOLLOW_PAIRS = {(p1, p2) for p1 in HOLLOW_SET for p2 in HOLLOW_SET}

        def generate_two_seq_add_data(num_samples=10000, is_train=True):
            """
            两个周期序列 → LCM → 相加 mod10 → 下一个 token 预测任务
            训练集：排除边缘区与空洞区
            测试集：从边缘区或空洞区抽样
            """
            X, Y, period_list = [], [], []

            for _ in range(num_samples):

                # -------------------------
                # 采样周期（只允许在此写逻辑）
                # -------------------------

                if is_train:
                    # 训练集：必须不在边缘区，也不在空洞区
                    while True:
                        p1, p2 = random.choices(PERIOD_RANGE, k=2)
                        pair = (p1, p2)
                        if (pair not in BORDER_PAIRS) and (pair not in HOLLOW_PAIRS):
                            break
                else:
                    p = random.random()
                    if p < 0.33:
                        # 边缘区
                        while True:
                            p1, p2 = random.choices(PERIOD_RANGE, k=2)
                            if (p1, p2) in BORDER_PAIRS:
                                break
                    elif p < 0.67:
                        # 空洞区 - 直接从集合中抽
                        p1, p2 = random.choice(list(HOLLOW_PAIRS))
                    else:
                        while True:
                            p1, p2 = random.choices(PERIOD_RANGE, k=2)
                            pair = (p1, p2)
                            if (pair not in BORDER_PAIRS) and (pair not in HOLLOW_PAIRS):
                                break


                # -------------------------
                # 周期序列生成
                # -------------------------

                seq1 = [random.choice(VOCAB) for _ in range(p1)]
                seq2 = [random.choice(VOCAB) for _ in range(p2)]

                # LCM
                lcm_len = 2 * abs(p1 * p2) // math.gcd(p1, p2)

                full_seq1 = (seq1 * (lcm_len // p1 + 1))[:lcm_len]
                full_seq2 = (seq2 * (lcm_len // p2 + 1))[:lcm_len]

                seq_sum = [(a + b) % 10 for a, b in zip(full_seq1, full_seq2)]

                seq_full = seq1 + ["+"] + seq2 + ["="] + seq_sum

                # 预测下一个 token
                X.append("".join(map(str, seq_full[:-1])))
                Y.append("".join(map(str, seq_full[1:])))

                cut_point = len(seq1) + len(seq2) + 2
                period_list.append((len(seq_full), cut_point))

            return X, Y, period_list
    
        print(f'generate data from the {periodic_type} function')
    
        PERIOD = 10
        BATCHSIZE = 128
        NUMEPOCH = 451
        PRINTEPOCH = 15
        lr = 1e-5
        wd = 0.01
    
        # 数据生成
        if load_data:
            pass
        else:
            t, data, y_uper = generate_two_seq_add_data(50000, is_train=True)
            t_test, data_test, y_lower = generate_two_seq_add_data(3000, is_train=False)
    
        print("Sample train data points:", t[:3], data[:3], y_uper[:3])
        print("Sample test data points:", t_test[:3], data_test[:3])

    elif periodic_type == '2seq_add_dense':  # 两个周期序列逐位_dense
        VOCAB = list(range(10))
        PERIOD_RANGE = list(range(2, 14))
        # 边缘区（二元组表示）
        BORDER_PAIRS = {
            # 左边缘
            *((p, q) for p in {2} for q in PERIOD_RANGE),
            # 右边缘
            *((p, q) for p in {13} for q in PERIOD_RANGE),
            # 上下边缘（保持视觉一致性）
            *((p, q) for q in {2} for p in PERIOD_RANGE),
            *((p, q) for q in {13} for p in PERIOD_RANGE),
        }

        # 空洞区（二元组表示）
        #HOLLOW_SET = {8, 9, 10, 11}
        #HOLLOW_PAIRS = {(p1, p2) for p1 in HOLLOW_SET for p2 in HOLLOW_SET}
        HOLLOW_PAIRS = {(8, 7), (8, 8)}

        def generate_two_seq_add_data(num_samples=10000, is_train=True):
            """
            两个周期序列 → LCM → 相加 mod10 → 下一个 token 预测任务
            训练集：排除边缘区与空洞区
            测试集：从边缘区或空洞区抽样
            """
            X, Y, period_list = [], [], []

            for _ in range(num_samples):

                # -------------------------
                # 采样周期（只允许在此写逻辑）
                # -------------------------

                if is_train:
                    # 训练集：必须不在边缘区，也不在空洞区
                    while True:
                        p1, p2 = random.choices(PERIOD_RANGE, k=2)
                        pair = (p1, p2)
                        if (pair not in BORDER_PAIRS) and (pair not in HOLLOW_PAIRS):
                            break
                else:
                    p = random.random()
                    if p < 0.33:
                        # 边缘区
                        while True:
                            p1, p2 = random.choices(PERIOD_RANGE, k=2)
                            if (p1, p2) in BORDER_PAIRS:
                                break
                    elif p < 0.67:
                        # 空洞区 - 直接从集合中抽
                        p1, p2 = random.choice(list(HOLLOW_PAIRS))
                    else:
                        while True:
                            p1, p2 = random.choices(PERIOD_RANGE, k=2)
                            pair = (p1, p2)
                            if (pair not in BORDER_PAIRS) and (pair not in HOLLOW_PAIRS):
                                break


                # -------------------------
                # 周期序列生成
                # -------------------------

                seq1 = [random.choice(VOCAB) for _ in range(p1)]
                seq2 = [random.choice(VOCAB) for _ in range(p2)]

                # LCM
                lcm_len = 2 * abs(p1 * p2) // math.gcd(p1, p2)

                full_seq1 = (seq1 * (lcm_len // p1 + 1))[:lcm_len]
                full_seq2 = (seq2 * (lcm_len // p2 + 1))[:lcm_len]

                seq_sum = [(a + b) % 10 for a, b in zip(full_seq1, full_seq2)]

                seq_full = seq1 + ["+"] + seq2 + ["="] + seq_sum

                # 预测下一个 token
                X.append("".join(map(str, seq_full[:-1])))
                Y.append("".join(map(str, seq_full[1:])))

                cut_point = len(seq1) + len(seq2) + 2
                period_list.append((len(seq_full), cut_point))

            return X, Y, period_list
    
        print(f'generate data from the {periodic_type} function')
    
        PERIOD = 10
        BATCHSIZE = 32
        NUMEPOCH = 301
        PRINTEPOCH = 15
        lr = 1e-5
        wd = 0.01
    
        # 数据生成
        if load_data:
            pass
        else:
            t, data, y_uper = generate_two_seq_add_data(50000, is_train=True)
            t_test, data_test, y_lower = generate_two_seq_add_data(3000, is_train=False)
    
        print("Sample train data points:", t[:3], data[:3], y_uper[:3])
        print("Sample test data points:", t_test[:3], data_test[:3])

    elif periodic_type == '2seq_add_sub':  # 两个周期序列逐位相加+变周期
        VOCAB = list(range(10))
        PERIOD_RANGE = list(range(2, 17))
        # 边缘区（二元组表示）
        BORDER_PAIRS = {
            # 左边缘
            *((p, q) for p in {2, 3} for q in PERIOD_RANGE),
            # 右边缘
            *((p, q) for p in {15, 16} for q in PERIOD_RANGE),
            # 上下边缘（保持视觉一致性）
            *((p, q) for q in {2, 3} for p in PERIOD_RANGE),
            *((p, q) for q in {15, 16} for p in PERIOD_RANGE),
        }

        # 空洞区（二元组表示）
        HOLLOW_SET = {8, 9, 10, 11}
        HOLLOW_PAIRS = {(p1, p2) for p1 in HOLLOW_SET for p2 in HOLLOW_SET}

        def generate_two_seq_add_data(num_samples=10000, is_train=True):
            """
            两个周期序列 → LCM → 相加 mod10 → 下一个 token 预测任务
            训练集：排除边缘区与空洞区
            测试集：从边缘区或空洞区抽样
            """
            X, Y, period_list = [], [], []

            for _ in range(num_samples):

                # -------------------------
                # 采样周期（只允许在此写逻辑）
                # -------------------------

                if is_train:
                    # 训练集：必须不在边缘区，也不在空洞区
                    while True:
                        p1, p2 = random.choices(PERIOD_RANGE, k=2)
                        pair = (p1, p2)
                        if (pair not in BORDER_PAIRS) and (pair not in HOLLOW_PAIRS):
                            break
                else:
                    p = random.random()
                    if p < 0.33:
                        # 边缘区
                        while True:
                            p1, p2 = random.choices(PERIOD_RANGE, k=2)
                            if (p1, p2) in BORDER_PAIRS:
                                break
                    elif p < 0.67:
                        # 空洞区 - 直接从集合中抽
                        p1, p2 = random.choice(list(HOLLOW_PAIRS))
                    else:
                        while True:
                            p1, p2 = random.choices(PERIOD_RANGE, k=2)
                            pair = (p1, p2)
                            if (pair not in BORDER_PAIRS) and (pair not in HOLLOW_PAIRS):
                                break


                # -------------------------
                # 周期序列生成
                # -------------------------

                seq1 = [random.choice(VOCAB) for _ in range(p1)]
                seq2 = [random.choice(VOCAB) for _ in range(p2)]

                # LCM
                lcm_len = 2 * abs(p1 * p2) // math.gcd(p1, p2)

                full_seq1 = (seq1 * (lcm_len // p1 + 1))[:lcm_len]
                full_seq2 = (seq2 * (lcm_len // p2 + 1))[:lcm_len]

                seq_sum = []
                for i, (a, b) in enumerate(zip(full_seq1, full_seq2)):
                    if i % 2 == 0:
                        # 偶数位：加法
                        y = (a + b) % 10
                    else:
                        # 奇数位：减法
                        y = (a - b) % 10
                    seq_sum.append(y)

                # if not is_train and _==2:
                #     for i, (a, b) in enumerate(zip(full_seq1, full_seq2)):
                #         print(i, a, b, OFFSET_VALUES[i % OFFSET_PERIOD], (a + b + OFFSET_VALUES[i % OFFSET_PERIOD]) % 10)
                #         import pdb
                #         pdb.set_trace()

                seq_full = seq1 + ["+"] + seq2 + ["="] + seq_sum

                # 预测下一个 token
                X.append("".join(map(str, seq_full[:-1])))
                Y.append("".join(map(str, seq_full[1:])))

                cut_point = len(seq1) + len(seq2) + 2
                period_list.append((len(seq_full), cut_point))

            return X, Y, period_list
    
        print(f'generate data from the {periodic_type} function')
    
        PERIOD = 10
        BATCHSIZE = 96
        NUMEPOCH = 451
        PRINTEPOCH = 15
        lr = 1e-5
        wd = 0.01
    
        # 数据生成
        if load_data:
            pass
        else:
            t, data, y_uper = generate_two_seq_add_data(50000, is_train=True)
            t_test, data_test, y_lower = generate_two_seq_add_data(3000, is_train=False)
    
        print("Sample train data points:", t[:3], data[:3], y_uper[:3])
        print("Sample test data points:", t_test[:3], data_test[:3])

    elif periodic_type == '2seq_div':  # 两个周期序列逐位相加
        def generate_two_seq_add_data(num_samples=10000, is_train=True):
            """
            训练集：从 {2,3,6,7} 中随机选两个不同的周期长度
            测试集：从 {4,5} 中选两个不同的周期长度
            每个样本由两个周期序列组成，最终取位相加 mod 10
            输出周期为两个周期的最小公倍数
            """
            X, Y, period_list = [], [], []
            for _ in range(num_samples):
                vocab = list(range(10))
                if is_train:
                    while True:
                        p1, p2 = random.choices([1, 2, 3, 4], k=2)
                        # 禁止 (3,4) 和 (4,3)
                        if not ((p1 == 2 and p2 == 3) or (p1 == 2 and p2 == 3)):
                            break
                else:
                    p1, p2 = random.sample([2, 3], 2)
    
                # 生成两个基本周期序列
                seq1 = [random.choice(vocab) for _ in range(p1)]
                seq2 = [random.choice(vocab) for _ in range(p2)]
    
                # 求最小公倍数作为完整长度
                lcm_len = 2 * abs(p1 * p2) // math.gcd(p1, p2) 
    
                # 扩展两个序列至相同长度
                full_seq1 = (seq1 * (lcm_len // p1 + 1))[:lcm_len]
                full_seq2 = (seq2 * (lcm_len // p2 + 1))[:lcm_len]
    
                # 元素相加取模10
                seq_sum = [(a + b) % 10 for a, b in zip(full_seq1, full_seq2)]
                seq_full = seq_sum + full_seq1 + full_seq2 
                # 构造输入输出：预测下一个token
                X.append("".join(map(str, seq_full[:-1])))
                Y.append("".join(map(str, seq_full[1:])))
    
                # start = len(full_seq1) + len(full_seq2)
                # end = len(seq_full) - 1
                # cut_point = random.randint(start, end)
                cut_point = len(seq_sum) #len(full_seq1) + len(full_seq2)

                period_list.append((lcm_len, cut_point))
    
            return X, Y, period_list
    
        print(f'generate data from the {periodic_type} function')
    
        PERIOD = 10
        BATCHSIZE = 32
        NUMEPOCH = 101
        PRINTEPOCH = 10
        lr = 1e-5
        wd = 0.01
    
        # 数据生成
        t, data, y_uper = generate_two_seq_add_data(50000, is_train=True)
        t_test, data_test, y_lower = generate_two_seq_add_data(2000, is_train=False)
    
        print("Sample train data points:", t[:3], data[:3], y_uper[:3])
        print("Sample test data points:", t_test[:3], data_test[:3])
    elif periodic_type == 'mod':
        def generate_periodic_data(num_samples, PERIOD=100, is_train = True):
            if is_train:
                t = np.linspace(-PERIOD, PERIOD, num_samples)
            else:
                t = np.linspace(-2*PERIOD, 2*PERIOD, num_samples) 
            data = [i%5 for i in t]
            return t, data

        print(f'generate data from the {periodic_type} function')

        PERIOD = 20
        BATCHSIZE = 32
        NUMEPOCH = 10000
        PRINTEPOCH = 50
        lr = 1e-5
        wd = 0.01

        t, data = generate_periodic_data(int(10000*PERIOD))
        t_test, data_test = generate_periodic_data(4000, is_train = False)

        y_uper = 10
        y_lower = -5
    

    # ----------------------------------------------------------------------------------------------------------

    elif periodic_type == 'complex_1':

        # complex_period
        def generate_periodic_data(num_samples, PERIOD=100, is_train = True):
            if is_train:
                t = np.linspace(-PERIOD, PERIOD, num_samples)
            else:
                t = np.linspace(-2*PERIOD, 2*PERIOD, num_samples)
            data = np.exp(np.sin(np.pi * t)**2 + np.cos(t) + t%3 - 1)
            return t, data
        print(f'generate data from the {periodic_type} function')

        PERIOD = 4
        BATCHSIZE = 32
        NUMEPOCH = 10000
        PRINTEPOCH = 50
        lr = 1e-5
        wd = 0.01

        t, data = generate_periodic_data(int(10000*PERIOD))
        t_test, data_test = generate_periodic_data(4000, is_train = False)

        y_uper = 20
        y_lower = -20        
    
    # ----------------------------------------------------------------------------------------------------------

    elif periodic_type == 'complex_2':
        def generate_periodic_data(num_samples, PERIOD=100, is_train = True):
            if is_train:
                t = np.linspace(-PERIOD, PERIOD, num_samples)
            else:
                t = np.linspace(-2*PERIOD, 2*PERIOD, num_samples)   

            data = (1 + np.sin(t)) * np.sin(2 * t)
            return t, data
        print(f'generate data from the {periodic_type} function')

        PERIOD = 4
        BATCHSIZE = 32
        NUMEPOCH = 10000
        PRINTEPOCH = 50
        lr = 1e-5
        wd = 0.01

        t, data = generate_periodic_data(int(10000*PERIOD))
        t_test, data_test = generate_periodic_data(4000, is_train = False)

        y_uper = 4
        y_lower = -4

    # ----------------------------------------------------------------------------------------------------------

    elif periodic_type == 'complex_3':

        def generate_periodic_data(num_samples, PERIOD=100, is_train = True):
            if is_train:
                t = np.linspace(-PERIOD, PERIOD, num_samples)
            else:
                t = np.linspace(-2*PERIOD, 2*PERIOD, num_samples)   

            data = np.sin(t + np.sin(2 * t))
            return t, data
        print(f'generate data from the {periodic_type} function')

        PERIOD = 4
        BATCHSIZE = 32
        NUMEPOCH = 10000
        PRINTEPOCH = 50
        lr = 1e-5
        wd = 0.01

        t, data = generate_periodic_data(int(10000*PERIOD))
        t_test, data_test = generate_periodic_data(4000, is_train = False)

        y_uper = 2
        y_lower = -2

    # ----------------------------------------------------------------------------------------------------------

    elif periodic_type == 'complex_4':

        def generate_periodic_data(num_samples, PERIOD=100, is_train = True):
            if is_train:
                t = np.linspace(-PERIOD, PERIOD, num_samples)
            else:
                t = np.linspace(-2*PERIOD, 2*PERIOD, num_samples)   

            data = np.sin(t) * np.cos(2 * t)**2 + np.cos(t) * np.sin(3 * t)**2
            return t, data
        print(f'generate data from the {periodic_type} function')

        PERIOD = 4
        BATCHSIZE = 32
        NUMEPOCH = 10000
        PRINTEPOCH = 50
        lr = 1e-5
        wd = 0.01

        t, data = generate_periodic_data(int(10000*PERIOD))
        t_test, data_test = generate_periodic_data(4000, is_train = False)

        y_uper = 2
        y_lower = -2


    # ----------------------------------------------------------------------------------------------------------

    elif periodic_type == 'complex_5':

        def generate_periodic_data(num_samples, PERIOD=100, is_train = True):
            if is_train:
                t = np.linspace(-PERIOD, PERIOD, num_samples)
            else:
                t = np.linspace(-2*PERIOD, 2*PERIOD, num_samples) 

            N = 5
            data = np.zeros_like(t)
            for n in range(1, N+1):
                data += (1/n) * sawtooth_wave(n * t, n)

            return t, data
        print(f'generate data from the {periodic_type} function')

        PERIOD = 4
        BATCHSIZE = 32
        NUMEPOCH = 10000
        PRINTEPOCH = 50
        lr = 1e-5
        wd = 0.01

        t, data = generate_periodic_data(int(10000*PERIOD))
        t_test, data_test = generate_periodic_data(4000, is_train = False)

        y_uper = 1
        y_lower = -1

    # ----------------------------------------------------------------------------------------------------------

    elif periodic_type == 'complex_6':

        def generate_periodic_data(num_samples, PERIOD=100, is_train = True):
            if is_train:
                t = np.linspace(-PERIOD, PERIOD, num_samples)
            else:
                t = np.linspace(-2*PERIOD, 2*PERIOD, num_samples) 

            data = np.exp(np.sin(t)) / (1 + np.cos(2 * t)**2)

            return t, data
        print(f'generate data from the {periodic_type} function')

        PERIOD = 4
        BATCHSIZE = 32
        NUMEPOCH = 10000
        PRINTEPOCH = 50
        lr = 1e-5
        wd = 0.01

        t, data = generate_periodic_data(int(10000*PERIOD))
        t_test, data_test = generate_periodic_data(4000, is_train = False)

        y_uper = 3
        y_lower = 0


    return t, data, t_test, data_test, PERIOD, BATCHSIZE, NUMEPOCH, PRINTEPOCH, lr, wd, y_uper, y_lower


def plot_periodic_data(t, data, t_test, data_test, result, args, epoch, path, y_uper, y_lower):
    import matplotlib.pyplot as plt
    import numpy as np

    # 转回浮点数
    t = np.array([float(x) for x in t])
    t_test = np.array([float(x) for x in t_test])
    data = np.array([float(x) for x in data])
    data_test = np.array([float(x) for x in data_test])
    result_fixed = []
    for x in result:
        try:
            result_fixed.append(float(x))
        except Exception:
            result_fixed.append(0.0)
    result = np.array(result_fixed)

    print("x_train:", t)
    print("y_train:", data)
    print("x_test:", t_test)
    print("y_test:", data_test)
    print("prediction:", result)

    plt.figure(figsize=(35, 5))
    plt.plot(t_test, data_test, label='Domain of Test Data', color='blue')
    plt.plot(t, data, label='Domain of Training Data', color='green')
    plt.plot(t_test, result, label='Model Predictions', color='red', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(min(t_test),max(t_test))
    # y_lower = -60
    # y_uper = 60
    plt.ylim(y_lower, y_uper)
    # plt.legend()
    plt.savefig(f'{path}/epoch{epoch}.png')
    
def read_log_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        train_loss = []
        test_loss = []
        for line in lines:
            if 'Train Loss' in line:
                train_loss.append(float(line.split(' ')[-1].strip()))
            elif 'Test Loss' in line:
                test_loss.append(float(line.split(' ')[-1].strip()))
    return train_loss, test_loss

def plot_periodic_loss(log_file_path):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    train_log_loss, test_log_loss = read_log_file(log_file_path)
    
    log_file_name = log_file_path.split('.')[0]
    ax1.plot(np.arange(0,len(train_log_loss)*50,50),train_log_loss, label=log_file_name)
    ax2.plot(np.arange(0,len(test_log_loss)*50,50),test_log_loss, label=log_file_name)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.legend(loc='upper right')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Loss')
    ax2.legend(loc='upper right')
    plt.savefig(f'{log_file_name}.pdf')
