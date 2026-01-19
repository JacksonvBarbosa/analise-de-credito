# Libs
import pandas as pd
from pathlib import Path

# Modulos
from src.pipeline.pipeline_ml import pipeline


def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "dados" / "processed" / "df_analises_models.csv"

    print(data_path)

    df = pd.read_csv(data_path)
    pipeline(df, "xgboost", save=True, nome='xgb')


if __name__ == "__main__":
    main()