from main import StockLibrary


library = StockLibrary(
    bucket="stocks-data",
    start_date="2025-01-01",
    end_date="2100-01-01",
    train_percent=0.70,
    validation_percent=0.15,
    local_dir="data"
)

train, validation, testing, profits = library.run()

print("Training:", train.shape)
print("Validation:", validation.shape)
print("Testing:", testing.shape)
print("Profits:", profits.shape)


print("\nTRAINING DATA")
print(train.head(10))

print("\nVALIDATION DATA")
print(validation.head(10))

print("\nTESTING DATA")
print(testing.head(10))

print("\nPROFIT DATA")
print(profits.head(10))



print("\nTRAINING DATA")
print(train)

print("\nVALIDATION DATA")
print(validation)

print("\nTESTING DATA")
print(testing)

print("\nPROFIT DATA")
print(profits)