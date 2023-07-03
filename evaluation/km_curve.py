# Python code to create the above Kaplan Meier curve
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times

from matplotlib import pyplot as plt
from lifelines.datasets import load_waltons
df = load_waltons() # returns a Pandas DataFrame

T = df['T']
E = df['E']

# create a kmf object
kmf = KaplanMeierFitter()


# Fit the data into the model
kmf.fit(T, E, label='Kaplan Meier Estimate')
# kmf.fit(T, E, label='Kaplan Meier Estimate', timeline=range(0, 100, 2))  # 可以指定生存时间坐标的区间

# kmf.plot_survival_function(ci_show=False)  # 没有ci区间的曲线
kmf.plot_survival_function()  # 有ci区间的曲线
# kmf.plot_cumulative_density()  # 累积风险函数，同样可用ci_show=False

# 生存时间中位数
# median_ = kmf.median_survival_time_
# print(median_)
# median_confidence_interval_ = median_survival_times(kmf.confidence_interval_)
# print(median_confidence_interval_)

# 单因素分组
# groups = df['group']
# ix = (groups == 'miR-137')
# kmf.fit(T[~ix], E[~ix], label='control')
# ax = kmf.plot()
# kmf.fit(T[ix], E[ix], label='miR-137')
# ax = kmf.plot(ax=ax)

# 多因素分组
# ax = plt.subplot(111)
#
# for name, grouped_df in df.groupby('group'):
#     kmf.fit(grouped_df["T"], grouped_df["E"], label=name)
#     kmf.plot(ax=ax)

plt.show()
