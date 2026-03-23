import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def load_data(file_path):
    """加载数据"""
    df = pd.read_csv(file_path)
    # 检查是否有年龄列，这是年龄匹配的前提
    if 'age' not in df.columns:
        raise ValueError("数据集中未找到 'age' 列，请先合并被试年龄信息后再运行此脚本。")
    print(f"数据加载完成: {len(df)} 行, {df['cope'].nunique()} 个COPE, {df['roi_id'].nunique()} 个脑区")
    return df


def get_twin_pairs(df):
    """获取双胞胎对"""
    twin_df = df[df['1-Twins;2-NoTwin'] == 1].copy()
    twin_pairs = {}
    for pair_id, group in twin_df.groupby('pair_id'):
        subjects = group['sub-clean'].unique()
        if len(subjects) == 2:
            twin_pairs[pair_id] = tuple(subjects)
    print(f"找到 {len(twin_pairs)} 对双胞胎")
    return twin_pairs


def get_non_twin_subjects(df):
    """获取非双胞胎被试"""
    non_twins = df[df['1-Twins;2-NoTwin'] == 2]['sub-clean'].unique()
    print(f"找到 {len(non_twins)} 个非双胞胎被试")
    return list(non_twins)


def calculate_icc_manual(icc_df):
    """手动计算ICC(3,1) - 绝对一致性，单个测量"""
    try:
        wide_data = icc_df.pivot(index='family', columns='member', values='activation')
        wide_data = wide_data.dropna()
        if len(wide_data) < 5:
            return np.nan
        data = wide_data.values
        n, k = data.shape
        grand_mean = np.mean(data)
        group_means = np.mean(data, axis=1)
        ss_between = k * np.sum((group_means - grand_mean) ** 2)
        ms_between = ss_between / (n - 1)
        ss_within = np.sum((data - group_means[:, np.newaxis]) ** 2)
        ms_within = ss_within / (n * (k - 1))
        if (ms_between + ms_within) > 0:
            return (ms_between - ms_within) / (ms_between + ms_within)
        return np.nan
    except Exception:
        return np.nan


def calculate_twin_icc_for_region(df, cope_num, region_id, twin_pairs):
    """计算单个脑区的双胞胎ICC"""
    region_data = df[(df['cope'] == cope_num) & (df['roi_id'] == region_id)].copy()
    if len(region_data) == 0: return np.nan, 0
    icc_data = []
    for pair_id, (sub1, sub2) in twin_pairs.items():
        val1 = region_data[region_data['sub-clean'] == sub1]['mean_beta'].values
        val2 = region_data[region_data['sub-clean'] == sub2]['mean_beta'].values
        if len(val1) > 0 and len(val2) > 0:
            icc_data.extend([
                {'family': f'TP_{pair_id}', 'member': 't1', 'activation': float(val1[0])},
                {'family': f'TP_{pair_id}', 'member': 't2', 'activation': float(val2[0])}
            ])
    if len(icc_data) < 10: return np.nan, len(icc_data) // 2
    return calculate_icc_manual(pd.DataFrame(icc_data)), len(icc_data) // 2


def calculate_random_icc_for_region(df, cope_num, region_id, non_twin_subjects, n_pairs, age_tolerance=1):
    """【更新】计算非双胞胎随机配对的ICC（加入年龄容差约束）"""
    region_data = df[(df['cope'] == cope_num) & (df['roi_id'] == region_id)].copy()
    pool_data = region_data[region_data['sub-clean'].isin(non_twin_subjects)].reset_index(drop=True)

    if len(pool_data) < 10: return np.nan

    icc_data = []
    pair_count = 0
    used_indices = set()
    shuffled_indices = np.random.permutation(pool_data.index)

    for i in shuffled_indices:
        if i in used_indices or pair_count >= n_pairs: continue
        target_sub = pool_data.iloc[i]

        # 寻找年龄容差范围内的配对
        matches = pool_data[
            (np.abs(pool_data['age'] - target_sub['age']) <= age_tolerance) &
            (~pool_data.index.isin(used_indices)) & (pool_data.index != i)
            ]

        if not matches.empty:
            m_idx = np.random.choice(matches.index)
            match_sub = pool_data.loc[m_idx]
            icc_data.extend([
                {'family': f'R_{pair_count}', 'member': 'r1', 'activation': float(target_sub['mean_beta'])},
                {'family': f'R_{pair_count}', 'member': 'r2', 'activation': float(match_sub['mean_beta'])}
            ])
            used_indices.update([i, m_idx])
            pair_count += 1

    if pair_count < (n_pairs / 2) or len(icc_data) < 10: return np.nan
    return calculate_icc_manual(pd.DataFrame(icc_data))


def analyze_cope_icc(df, cope_num, n_random_sets=1000, age_tol=1):
    """分析单个COPE"""
    print(f"\n开始分析COPE {cope_num} (年龄容差: {age_tol}岁)...")
    twin_pairs = get_twin_pairs(df)
    non_twin_subjects = get_non_twin_subjects(df)
    if len(twin_pairs) < 5: return pd.DataFrame()

    cope_data = df[df['cope'] == cope_num].copy()
    all_regions = sorted(cope_data['roi_id'].unique())
    results = []

    for region_id in all_regions:
        region_info = cope_data[cope_data['roi_id'] == region_id].iloc[0]
        print(f"  处理脑区 {region_id}: {region_info['roi_name']}...", end=" ")

        icc_twin, n_twin_pairs = calculate_twin_icc_for_region(df, cope_num, region_id, twin_pairs)
        if np.isnan(icc_twin):
            print("跳过(数据不足)");
            continue

        random_iccs = []
        for _ in range(n_random_sets):
            res = calculate_random_icc_for_region(df, cope_num, region_id, non_twin_subjects, n_twin_pairs,
                                                  age_tolerance=age_tol)
            if not np.isnan(res): random_iccs.append(res)

        if len(random_iccs) < 100:
            print("跳过(有效配对不足)");
            continue

        extreme_count = np.sum(np.array(random_iccs) >= icc_twin)
        p_val = (extreme_count + 1) / (len(random_iccs) + 1)
        mean_r, std_r = np.mean(random_iccs), np.std(random_iccs)
        d = (icc_twin - mean_r) / std_r if std_r > 0 else 0

        results.append({
            'cope': cope_num, 'region_id': region_id, 'region_name': region_info['roi_name'],
            'atlas': region_info['atlas'], 'n_twin_pairs': n_twin_pairs, 'icc_twin': icc_twin,
            'icc_random_mean': mean_r, 'p_value': p_val, 'cohens_d': d, 'significant': p_val < 0.05
        })
        print(f"ICC={icc_twin:.3f}, p={p_val:.4f} {'✓' if p_val < 0.05 else ''}")

    return pd.DataFrame(results)


def analyze_all_copes(df, n_random_sets=1000):
    all_results = []
    for cope in sorted(df['cope'].unique()):
        res = analyze_cope_icc(df, cope, n_random_sets)
        if not res.empty: all_results.append(res)
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def summarize_results(df):
    print("\n" + "=" * 50 + "\n汇总报告\n" + "=" * 50)
    summary = df.groupby('cope').agg({'region_id': 'count', 'significant': 'sum', 'icc_twin': 'mean'})
    summary.columns = ['总脑区', '显著数', '平均ICC']
    print(summary)


def main():
    # ================= 配置区 =================
    input_file = "C:/Users/01/Desktop/203/ICC 3.csv"
    output_file = "C:/Users/01/Desktop/203/icc_analysis_age_matched.csv"
    # ==========================================
    try:
        df = load_data(input_file)
        results_df = analyze_all_copes(df, n_random_sets=1000)
        if not results_df.empty:
            summarize_results(results_df)
            results_df.to_csv(output_file, index=False)
            print(f"\n分析完成！结果已保存至: {output_file}")
        else:
            print("\n未发现显著结果或数据不足。")
    except Exception as e:
        print(f"\n程序运行出错: {e}")


if __name__ == "__main__":
    main()a