import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.rcParams['font.family'] = 'Times New Roman'

# エクセルファイルを読み込み、データフレームに格納
def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

# 1つのグラフに5つの箱ひげ図を作成
def create_boxplot(data, whisker_color, mean_color, fill_color, titles=None):
    if data.shape[1] < 4:
        print("エラー: データフレームに4列以上のデータが必要です。")
        return
    
    # 各列データをタプルに変換
    box_data = tuple(data.iloc[:, i].dropna() for i in range(4))

    # プロット設定
    fig, ax = plt.subplots(figsize=(6, 6))
    box = ax.boxplot(box_data, patch_artist=True, showmeans=True,
                     meanprops=dict(marker='o', markerfacecolor=mean_color, markeredgecolor=mean_color),
                     boxprops=dict(color=whisker_color,facecolor=fill_color),
                     whiskerprops=dict(color=whisker_color),
                     capprops=dict(color=whisker_color),
                     medianprops=dict(color=whisker_color),
                     showfliers=False)  # 中央値の色は緑
    
    # x軸のラベルを設定
    ax.set_xticklabels(titles if titles else ['Column 1', 'Column 2', 'Column 3', 'Column 4'])
    plt.show()

# ファイルパスと各設定を指定
file_path = 'data/serpenoid/5arcs_error_for_box-and-hide.csv'  # エクセルファイルのパス
titles = ['N = 2', 'N = 3', 'N = 4', 'N = 5']
whisker_color = (30/255,117/255,248/255)
mean_color = (255/255,173/255,0/255)
fill_color = (218/255,227/255,243/255)

# データを読み込み、箱ひげ図を作成
data = read_csv(file_path)
create_boxplot(data, whisker_color=whisker_color, mean_color=mean_color,fill_color=fill_color, titles=titles)