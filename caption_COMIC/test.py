# -*- coding: utf-8 -*-
"""
Created on 23 Dec 2019 17:49:51

@author: jiahuei
"""


def merge_with_marketdata(input_df, df_returns):
    print('Original length: {}'.format(len(input_df)))
    input_df = input_df.copy()
    df_returns = df_returns.copy()
    # Merge data with Marketdata
    df_returns.index = df_returns['datetime']
    df_returns = df_returns[['barra_id', 'marketcap', 'close_price']]
    input_df = pd.merge_asof(
        left=input_df.sort_index(), right=df_returns,
        left_index=True, right_index=True,
        by='barra_id', tolerance=pd.Timedelta('10day'))
    print('Merged length: {}'.format(len(input_df)))
    return input_df


def construct_feature(input_df, replace_na='zero'):
    input_df = input_df.copy()
    input_df = input_df[['barra_id',
                         'S_PLEDGE_SHARES', 'S_PLEDGE_SHR_RATIO', 'IS_DISCHARGE',
                         'close_price', 'marketcap']].copy()
    # input_df = input_df.dropna(subset=['S_PLEDGE_SHR_RATIO'])
    cos = input_df['barra_id'].unique()
    feature_list = []
    for c in tqdm(cos):
        subset = input_df[input_df['barra_id'] == c].copy()
        subset['pledge_discharged'] = np.where(subset['IS_DISCHARGE'], -1, 1)
        subset['pledge_ratio'] = subset['S_PLEDGE_SHR_RATIO'].multiply(subset['pledge_discharged']) / 100
        if replace_na is None:
            # Drop NA ratio
            subset = subset.dropna(subset=['pledge_ratio'])
        elif replace_na == 'zero':
            # Replace NA ratio with zeroes
            pass
        else:
            # Replace NA ratio with calculated ratios
            subset_na = subset[subset['pledge_ratio'].isna()]
            subset.loc[subset['pledge_ratio'].isna(), 'pledge_ratio'] = subset_na['S_PLEDGE_SHARES'] \
                .multiply(subset_na['pledge_discharged']).multiply(subset_na['close_price'] * 1e4).divide(
                subset_na['marketcap'])
        subset['pledge_ratio'] = subset['pledge_ratio'].fillna(value=0.)
        subset = subset['pledge_ratio']
        feature = pd.DataFrame(subset.resample('D', label='right', closed='right').sum())
        feature['ema_60'] = feature['pledge_ratio'].ewm(span=60).mean()
        feature['ema_90'] = feature['pledge_ratio'].ewm(span=90).mean()
        feature['ema_180'] = feature['pledge_ratio'].ewm(span=180).mean()
        feature['pledge_ratio_feature_60'] = (feature['ema_60'] - feature['ema_180']) / feature['ema_180']
        feature['pledge_ratio_feature_90'] = (feature['ema_90'] - feature['ema_180']) / feature['ema_180']
        feature['pledge_ratio_feature'] = feature['pledge_ratio']
        feature['barra_id'] = c
        feature_list.append(feature)
    out = pd.concat(feature_list, axis=0)
    out.index = out.index + pd.DateOffset(hours=1)
    out.index.name = 'datetime'
    out = out.sort_values(by=['datetime', 'barra_id'])
    for span in tqdm(('', '_60', '_90')):
        col = 'pledge_ratio_feature' + span
        lo_clip = out[col].expanding().quantile(.003)
        hi_clip = out[col].expanding().quantile(1 - .003)
        lo_clip = pd.DataFrame(lo_clip.groupby(lo_clip.index).tail(1).rename('lo_clip' + span))
        hi_clip = pd.DataFrame(hi_clip.groupby(hi_clip.index).tail(1).rename('hi_clip' + span))
        out = out.merge(lo_clip, how='left', left_index=True, right_index=True).merge(hi_clip, how='left',
                                                                                      left_index=True, right_index=True)
        out[col] = out[col].clip(lower=out['lo_clip' + span], upper=out['hi_clip' + span])
        out = out.join(out[col].groupby(out.index).mean().rename('m' + span))
        out[col + '_demeaned'] = out[col] - out['m' + span]
    return out


def scale_pledge_by_post(input_df, threshold=-1):
    input_df = input_df.copy()
    input_df['post_scaling'] = np.where(input_df['post'] > threshold, 0.5, 1.0)
    input_df['S_PLEDGE_SHARES'] = input_df['S_PLEDGE_SHARES'].multiply(input_df['post_scaling'])
    input_df['S_PLEDGE_SHR_RATIO'] = input_df['S_PLEDGE_SHR_RATIO'].multiply(input_df['post_scaling'])
    return input_df


def construct_feature(input_df):
    input_df = input_df[['barra_id', 'S_PLEDGE_SHR_RATIO', 'IS_DISCHARGE', 'post']].copy()
    input_df = input_df.dropna(subset=['S_PLEDGE_SHR_RATIO'])
    cos = input_df['barra_id'].unique()
    feature_list = []
    for c in tqdm(cos):
        subset = input_df[input_df['barra_id'] == c].copy()
        subset['pledge_discharged'] = np.where(subset['IS_DISCHARGE'], -1, 1)
        subset['post_scaling'] = np.where(subset['post'] == 0, 1.0, 0.5)
        subset['pledge_ratio'] = subset['S_PLEDGE_SHR_RATIO'].multiply(subset['post_scaling']) \
                                     .multiply(subset['pledge_discharged']) / 100
        subset = subset[['pledge_discharged', 'post_scaling', 'S_PLEDGE_SHR_RATIO', 'pledge_ratio']]
        feature = pd.DataFrame(subset.resample('D', label='right', closed='right').sum())
        feature['pledge_ratio_feature'] = feature['pledge_ratio']
        feature['barra_id'] = c
        feature_list.append(feature)
    #         display(feature)
    #         break
    out = pd.concat(feature_list, axis=0)
    out.index = out.index + pd.DateOffset(hours=1)
    out.index.name = 'datetime'
    out = out.sort_values(by=['datetime', 'barra_id'])
    lo_clip = out['pledge_ratio_feature'].expanding().quantile(.003)
    hi_clip = out['pledge_ratio_feature'].expanding().quantile(1 - .003)
    lo_clip = pd.DataFrame(lo_clip.groupby(lo_clip.index).tail(1).rename('lo_clip'))
    hi_clip = pd.DataFrame(hi_clip.groupby(hi_clip.index).tail(1).rename('hi_clip'))
    out = out.merge(lo_clip, how='left', left_index=True, right_index=True).merge(hi_clip, how='left', left_index=True,
                                                                                  right_index=True)
    out['pledge_ratio_feature'] = out['pledge_ratio_feature'].clip(lower=out['lo_clip'], upper=out['hi_clip'])
    out = out.join(out['pledge_ratio_feature'].groupby(out.index).mean().rename('m'))
    out['pledge_ratio_feature_demeaned'] = out['pledge_ratio_feature'] - out['m']
    return out
