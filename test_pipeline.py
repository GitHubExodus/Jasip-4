

from main import StockLibrary
from xgboostClass import XGBoostLibrary


stock_data = StockLibrary(
    bucket="stocks-data",
    start_date="2025-01-01",
    end_date="2100-01-01",
    train_percent=0.70,
    validation_percent=0.15,
    local_dir="data"
)
train, validation, testing, profits = stock_data.run()

print("Data processing complete")
print("Training:", train.shape)
print("Validation:", validation.shape)
print("Testing:", testing.shape)
print("Profits:", profits.shape)


model = XGBoostLibrary(
    train_file="data/training.parquet",
    validation_file="data/validation.parquet",
    test_file="data/testing.parquet",
    stats_dir="data/xgboost"
)

model.train_model(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)

print("\nModel statistics:")
print(model.evaluate())

print("\nFeature usefulness:")
print(model.feature_usefulness().head(20))

model.save()

print("\nFull process complete")