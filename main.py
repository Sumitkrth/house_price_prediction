import argparse
from Src.train_model import preprocess_and_train
from Src.predict_price import predict_house_price

# Train only if needed
try:
    model, r2, rmse = preprocess_and_train()
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("Make sure 'Data/house_data.csv' exists in your repo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="House Price Predictor")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", nargs=3, metavar=('AREA', 'BEDROOMS', 'LOCATION'),
                        help="Predict house price")
    args = parser.parse_args()

    if args.train:
        preprocess_and_train()
    elif args.predict:
        area = float(args.predict[0])
        bedrooms = int(args.predict[1])
        location = args.predict[2]
        price = predict_house_price(area, bedrooms, location)
        print(f"Predicted House Price: {price:.2f}")
    else:
        parser.print_help()
