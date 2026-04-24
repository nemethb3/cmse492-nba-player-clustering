import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

RAW_COLS = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
            'MIN', 'PLUS_MINUS', 'FGA', 'FG3A', 'FTA', 'OREB', 'DREB']

FEATURES = ['PTS_PER_MIN', 'AST_PER_MIN', 'REB_PER_MIN', 'STL_PER_MIN', 'BLK_PER_MIN',
            'THREE_RATE', 'PAINT_SCORE', 'ASSIST_RATIO', 'FT_RATE', 'FG_PCT', 'PLUS_MINUS',
            'POSITION_SCORE', 'HEIGHT_IN']

# nba_api PlayerIndex uses abbreviated positions
_POS_MAP = {
    'G': 0.0, 'G-F': 0.33, 'F-G': 0.33,
    'F': 1.0, 'F-C': 1.67, 'C-F': 1.67, 'C': 2.0,
}

ARCHETYPES = {
    'Elite Scorer':       {'primary': 'PTS_PER_MIN',  'secondary': ['FT_RATE', 'ASSIST_RATIO']},
    'Playmaker':          {'primary': 'AST_PER_MIN',   'secondary': ['ASSIST_RATIO', 'PTS_PER_MIN']},
    '3PT Specialist':     {'primary': 'THREE_RATE',    'secondary': ['AST_PER_MIN', 'STL_PER_MIN']},
    'Rim Protector':      {'primary': 'BLK_PER_MIN',   'secondary': ['PAINT_SCORE', 'REB_PER_MIN']},
    'Rebounding Big':     {'primary': 'REB_PER_MIN',   'secondary': ['PAINT_SCORE', 'BLK_PER_MIN']},
    'Perimeter Defender': {'primary': 'STL_PER_MIN',   'secondary': ['THREE_RATE', 'AST_PER_MIN']},
}


def _height_to_inches(h):
    try:
        ft, inch = str(h).split('-')
        return int(ft) * 12 + int(inch)
    except Exception:
        return 78


def fetch_player_data(season='2023-24'):
    import time
    from nba_api.stats.endpoints import leaguedashplayerstats, playerindex

    time.sleep(1)
    endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season, per_mode_detailed='PerGame', season_type_all_star='Regular Season')
    df_raw = endpoint.get_data_frames()[0]

    time.sleep(1)
    try:
        pi = playerindex.PlayerIndex(season=season)
        df_pos = (pi.get_data_frames()[0][['PERSON_ID', 'POSITION', 'HEIGHT']]
                  .rename(columns={'PERSON_ID': 'PLAYER_ID'}))
        df_raw = df_raw.merge(df_pos, on='PLAYER_ID', how='left')
    except Exception:
        df_raw['POSITION'] = 'F'
        df_raw['HEIGHT'] = '6-6'

    return df_raw


def preprocess_data(df_raw, min_gp=30, min_min=15):
    df = df_raw[(df_raw['GP'] >= min_gp) & (df_raw['MIN'] >= min_min)].copy()
    df = df[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'GP', 'POSITION', 'HEIGHT'] + RAW_COLS].copy()
    df.dropna(subset=RAW_COLS, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['PTS_PER_MIN']  = df['PTS'] / df['MIN']
    df['AST_PER_MIN']  = df['AST'] / df['MIN']
    df['REB_PER_MIN']  = df['REB'] / df['MIN']
    df['STL_PER_MIN']  = df['STL'] / df['MIN']
    df['BLK_PER_MIN']  = df['BLK'] / df['MIN']
    df['THREE_RATE']   = (df['FG3A'].replace(0, np.nan) / df['FGA'].replace(0, np.nan)).fillna(0)
    df['PAINT_SCORE']  = (df['OREB'] + df['BLK']) / df['MIN']
    df['ASSIST_RATIO'] = (df['AST'].replace(0, np.nan) / (df['AST'] + df['TOV']).replace(0, np.nan)).fillna(0)
    df['FT_RATE']      = (df['FTA'].replace(0, np.nan) / df['FGA'].replace(0, np.nan)).fillna(0)
    df['POSITION_SCORE'] = df['POSITION'].map(_POS_MAP).fillna(1.0)
    df['HEIGHT_IN']    = df['HEIGHT'].apply(_height_to_inches)

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    return df, X, scaler


def run_kmeans(X, k=6, random_state=42, n_init=20):
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = kmeans.fit_predict(X)
    return kmeans, labels


def auto_label(centers_df, K):
    archetypes = list(ARCHETYPES.keys())
    n_arch = len(archetypes)

    scores = np.zeros((K, n_arch))
    for j, arch in enumerate(archetypes):
        defn = ARCHETYPES[arch]
        if defn['primary'] in centers_df.columns:
            scores[:, j] += centers_df[defn['primary']].values * 2.0
        for sec in defn['secondary']:
            if sec in centers_df.columns:
                scores[:, j] += centers_df[sec].values * 0.5

    labels = {}
    score_mat = scores.copy()
    for _ in range(min(K, n_arch)):
        flat_idx = int(np.argmax(score_mat))
        c_idx, a_idx = divmod(flat_idx, n_arch)
        labels[c_idx] = archetypes[a_idx]
        score_mat[c_idx, :] = -np.inf
        score_mat[:, a_idx] = -np.inf

    for i in range(K):
        if i not in labels:
            labels[i] = 'Role Players'
    return labels


def plot_clusters(df, x_col='PC1', y_col='PC2', archetype_col='Archetype'):
    import matplotlib.pyplot as plt
    archetypes = df[archetype_col].unique()
    fig, ax = plt.subplots(figsize=(12, 8))
    for arch in archetypes:
        subset = df[df[archetype_col] == arch]
        ax.scatter(subset[x_col], subset[y_col], label=arch, alpha=0.7, s=40)
    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    ax.set_title('NBA Player Clusters')
    ax.legend()
    plt.tight_layout()
    return fig
