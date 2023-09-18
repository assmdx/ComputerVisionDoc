1. pandas 删除第一列
```
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
print(train_data.iloc[0:4,[0,1,2,3,-3,-2,-1]]) # 打印0-4列
all_features = pd.concat((train_data.iloc[:,1:-1], test_data.iloc[:,1:])) # 删除第一列
```

