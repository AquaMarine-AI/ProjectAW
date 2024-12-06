import matplotlib.pyplot as plt
import pandas as pd

# 데이터를 데이터프레임으로 생성
data = {
    "MODEL": ["LSTM", "Sequential LSTM-Transformer", "Parallel LSTM-Transformer", "Temporal Fusion Transformer"],
    "RMSE": [0.0697, 0.2525, 0.1621, 0.051187],
    "R²": [0.9671, 0.5683, 0.8221, 0.8943],
    "MBD": [-0.0260, -0.0019, -0.0652, 0.018183],
    "MAEM": [0.4330, 0.3486, 0.4250, 0.033913]
}

df = pd.DataFrame(data)

# 표 그리기
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(df.columns))))

plt.show()
