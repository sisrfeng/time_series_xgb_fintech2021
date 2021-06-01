import pandas as pd


import pandas as pd
gt = pd.read_csv('./data/wf_test_Nov_peri.csv')#按0.5h计算
gt = gt.groupby(by=['date','post_id'], as_index=False)['amount'].agg('sum')
print(gt)
gt.to_csv('./data/wf_test_Nov_day.csv')