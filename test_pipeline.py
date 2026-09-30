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


# Save all processed data to R2 ONCE
for file in [
    "training.parquet",
    "validation.parquet",
    "testing.parquet",
    "profits.parquet"
]:
    stock_data.upload(
        f"data/{file}",
        f"jasip4/{file}"
    )

print("Processed data saved to R2")


# Train XGBoost
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


# Save model locally AND upload everything to R2
model.save()

model.upload_all(
    bucket="stocks-data",
    cloud_folder="jasip4/xgboost"
)

print("\nFull process complete")