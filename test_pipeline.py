from main import StockLibrary
from splitRanking import SplitRanking


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


ranking = SplitRanking(
    train,
    validation,
    testing,
    local_dir="data/splits",
    bucket="stocks-data",
    cloud_folder="jasip4/splits"
)

ranking.run()
ranking.save()

ranking.print_rankings(30)

print("\nSplit ranking saved locally and to R2")