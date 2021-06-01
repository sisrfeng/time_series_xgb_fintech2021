# %%
# import modin.pandas as pd
# from distributed import Client
# client = Client()

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
plt.close("all")

# %%
train_df=pd.read_csv('./data/train_wf.csv')
test_df=pd.read_csv('./data/wf_test_Nov_peri.csv')#按0.5h计算
wkd_df=pd.read_csv('./data/wkd_v1.csv')
wkd_df=wkd_df.rename(columns={'ORIG_DT':'date'})
train_df=train_df.merge(wkd_df)


# %%
#将A/B岗位拆分出来分别分析，以天为粒度
tmp=train_df[['date','post_id','amount']].groupby(['date','post_id'],sort=False).agg('sum')


# %%
train_day_df=tmp.reset_index()
train_day_df_A=train_day_df[train_day_df['post_id']=='A'].reset_index(drop=True)
train_day_df_B=train_day_df[train_day_df['post_id']=='B'].reset_index(drop=True)
train_day_df_A=train_day_df_A.merge(wkd_df)
train_day_df_B=train_day_df_B.merge(wkd_df)
#
def get_frt(df):
    df['WKD_TYP_CD']=df['WKD_TYP_CD'].map({'WN':0,'SN': 1, 'NH': 1, 'SS': 1, 'WS': 0})
    month=[]
    day=[]
    year=[]
    for date in df['date'].values:
        year.append(int(date.split('/')[0]))
        month.append(int(date.split('/')[1]))
        day.append(int(date.split('/')[2]))
    df['year']=year
    df['month']=month
    df['day']=day
    df.drop(['date','post_id'],axis=1,inplace=True)
    return df
train_day_df_A=get_frt(train_day_df_A)
train_day_df_B=get_frt(train_day_df_B)
train_day_df_A['amount']=train_day_df_A['amount']/1e4
train_day_df_B['amount']=train_day_df_B['amount']/1e4


# %%
watch_week = train_day_df_A[85:200]
pd.Series(watch_week['amount'].values, index=pd.date_range("2018-1-1", periods=len(watch_week['amount'].values))).plot()
watch_week = train_day_df_B[85:200]
pd.Series(watch_week['amount'].values, index=pd.date_range("2018-1-1", periods=len(watch_week['amount'].values))).plot()

# %% [markdown]
# 对于A：
# - 可以看到18/10--18/11月增加了(59.1890-54.6016)/54.6016=8.4%
# - 可以看到19/10--19/11月增加了(46.6275-44.9047)/44.9047=3.8%
# 
# 对于B:
# - 可以看到18/10--18/11月增加了(12.88-11.72)/11.72=9.9%
# - 可以看到19/10--19/11月增加了(3.793-3.625)/3.625=4.6%  
# 18-19年月份之间的趋势变化基本一致，20年因为疫情影响，数值的绝对值在变化，
# 但是趋势并未变化，如果以年份来看的话周期性十分明显。

# %%
#A: 看看每一年中每一个月的变化
group_year_month_A=train_day_df_A[['year','month','amount']].groupby(['year','month'],sort=False).agg(['sum'])
group_year_month_A=pd.DataFrame(group_year_month_A).reset_index()
#agg(['sum'])会让'amount'变'amount sum'
group_year_month_A.columns=['year','month','amount']


# %%
ts = pd.Series(group_year_month_A['amount'].values, index=pd.date_range("1/2018", periods=len(group_year_month_A['amount'].values),freq="M"))
ts.plot()


# %%
#B: 看看每一年中每一个月的变化
group_year_month_B=train_day_df_B[['year','month','amount']].groupby(['year','month'],sort=False).agg(['sum'])
group_year_month_B=pd.DataFrame(group_year_month_B).reset_index()
group_year_month_B.columns=['year','month','amount']
ts = pd.Series(group_year_month_B['amount'].values, index=pd.date_range("1/2018", periods=len(group_year_month_B['amount'].values),freq="M"))
ts.plot()

# %% [markdown]
# ## 分析18/19两年11月份的每一天的趋势
# 
# - 如果只看day和amount的趋势，18/19各自11月份的变化趋势和天数关系其实不大
# - 但通过分析WKD_TYP_CD可以明显看到，11月份只有两种WKD_TYP_CD:WN/SN，当为WN时，业务量明显高于SN.这也是符合常识的
# - 20年11月份的WKD_TYP_CD也只有WN/SN两种情形，所以我们只需要考虑WN和SN对amount的影响
#   - 这里最朴素的想法是统计18，19年11月0,1对应的每一天的amount占一个月的比例

# %%
nov_month=train_day_df_A[train_day_df_A['month']==11]
nov_month_18=nov_month[nov_month['year']==2018][['WKD_TYP_CD','amount']].reset_index(drop=True)
nov_month_19=nov_month[nov_month['year']==2019][['WKD_TYP_CD','amount']].reset_index(drop=True)
nov_month_18['amount']/=np.sum(nov_month_18['amount'])
nov_month_19['amount']/=np.sum(nov_month_19['amount'])
nov_month_18['amount'].plot()
nov_month_19['amount'].plot()
print(nov_month_18[nov_month_18['WKD_TYP_CD']==0]['amount'].mean())
print(nov_month_19[nov_month_19['WKD_TYP_CD']==0]['amount'].mean())
print(nov_month_18[nov_month_18['WKD_TYP_CD']==1]['amount'].mean())
print(nov_month_19[nov_month_19['WKD_TYP_CD']==1]['amount'].mean())

# %% [markdown]
# ## 结果
# 0.03926397564658053  
# 0.0415190392097041  
# 0.017024066971903562  
# 0.014233352955134861  
# 所以如果得到20年11月份的总业务量nov_2020  
# nov_2020*(0.039+0.041)/2 if 当前day的WKD_TYP_CD==0  
# nov_2020*(0.017+0.014)/2 if 当前day的WKD_TYP_CD==1
# 
# %% [markdown]
# ## 规则法建模
# 
# 经过上面的分析其实可以看出来数据规律性极强，辅以一定简单的模型和规则就可以取得很好的效果，这里点到为止咯，大家自己项怎么利用数据的规律
# 。目前利用**数据规律+一定的规则**可以得到的分数:
# - 任务一: 0.09
# - 任务二: 0.23

