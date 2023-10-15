import pandas as pd

DATA_PATH = 'C:/Users/SpaceSpace\PycharmProjects\Anomaly_Reseach\data/1group_2017_01-2017_04.csv'

df = pd.read_csv(DATA_PATH, index_col = 'timestamp', parse_dates=True)
df = df.squeeze()
print(df)
# from adtk.visualization import plot
from adtk.detector import PcaAD
pca_ad = PcaAD(k=1)
anomalies = pca_ad.fit_detect(df)
# plot(df, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3, curve_group='all');
print(anomalies)