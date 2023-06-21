import pandas as pd


def preprocess_data(hearts: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for companies.

    Args:
        hearts: Raw data.
    Returns:
        Preprocessed data, with `company_rating` converted to a float and
        `iata_approved` converted to boolean.
    """
    hearts['Sex'] = hearts['Sex'].astype('category')
    hearts['ChestPainType'] = hearts['ChestPainType'].astype('category')
    hearts['FastingBS'] = hearts['FastingBS'].astype('category')
    hearts['RestingECG'] = hearts['RestingECG'].astype('category')
    hearts['ExerciseAngina'] = hearts['ExerciseAngina'].astype('category')
    hearts['ST_Slope'] = hearts['ST_Slope'].astype('category')
    hearts = hearts.dropna()
    hearts_prepared = hearts

    return hearts_prepared

