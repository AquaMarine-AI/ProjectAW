모델 성능 지표 요약 (LSTM, Transformer, TFT)
모델	RMSE	R²	MAPE (%)	PCC	MBD	MAEM
LSTM	0.1001	0.9333	0.8072	0.9779	-0.0579	0.0579
Transformer	0.1535	0.8430	1.2370	0.9306	-0.0528	0.0528
TFT	0.2272	0.8104	1.5857	0.9017	-0.0139	0.0139
분석
RMSE, R², MAPE, PCC:

LSTM 모델이 이 지표에서 가장 높은 성능을 보입니다. 이는 LSTM이 Transformer와 TFT에 비해 예측의 정확도가 높고, 실제 값을 잘 따라가고 있음을 의미합니다.
Transformer 모델은 LSTM보다 성능이 다소 낮지만, TFT보다는 높습니다.
TFT 모델은 RMSE, R², MAPE, PCC 지표에서 가장 낮은 성능을 보입니다.
Mean Bias Deviation (MBD):

TFT 모델의 MBD가 -0.0139로 가장 0에 가깝습니다. 이는 TFT 모델이 장기적으로 과대 또는 과소 편향이 가장 적음을 나타냅니다.
Mean Absolute Error of Means (MAEM):

TFT 모델의 MAEM이 0.0139로 가장 낮아, 실제 값의 평균과 가장 가까운 예측을 하고 있음을 의미합니다.
결론
**가장 장기적인 추세를 잘 반영하는 모델은 TFT (Temporal Fusion Transformer)**입니다.

이유: MBD와 MAEM 지표에서 TFT 모델이 가장 낮은 값을 보이므로, 이 모델이 실제 데이터의 장기적인 평균 수준을 가장 잘 반영하고 있습니다. 이는 장기적인 평균 수준과 편향을 고려한 평가에서 TFT가 가장 우수하다는 것을 의미합니다.
비록 LSTM이 RMSE, R², MAPE, PCC와 같은 단기적인 예측 정확도 지표에서 더 높은 성능을 보이지만, 장기적인 추세와 평균 수준을 잘 반영하는 것이 중요한 경우에는 TFT 모델이 더 적합하다고 판단할 수 있습니다.