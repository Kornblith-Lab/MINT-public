import pandas as pd
import numpy as np

def create_age_category(age):
    """
    Convert age (in years) to categorical age group.

    Categories:
    - Infant (0 – <1)
    - Toddler (1 – <3)
    - Early childhood (3 – <6)
    - Middle childhood (6 – <12)
    - Adolescent (12 – <18)
    """
    if pd.isna(age):
        return np.nan
    if age < 1:
        return 'Infant (0-<1)'
    elif age < 3:
        return 'Toddler (1-<3)'
    elif age < 6:
        return 'Early childhood (3-<6)'
    elif age < 12:
        return 'Middle childhood (6-<12)'
    elif age < 18:
        return 'Adolescent (12-<18)'
    else:
        return np.nan

def combine_race_ethnicity(
    df: pd.DataFrame,
    race_col: str = "FirstRace",
    ethnicity_col: str = "Ethnicity",
    multiracial_col: str = "MultiRacial",
) -> pd.Series:
    """
    Create a combined race-ethnicity category.

    Output categories:
      - Multiracial
      - Hispanic White
      - Hispanic Non-White
      - Black
      - Asian
      - Other
    """
    race = df[race_col].astype("string").str.strip().str.lower()
    eth = df[ethnicity_col].astype("string").str.strip().str.lower()
    multiracial = pd.to_numeric(df[multiracial_col], errors="coerce").fillna(0).eq(1)

    # Only rows POSITIVELY identified as Hispanic/Latino become Latino. Missing
    # ethnicity (Declined / Unknown / *Unspecified / NaN) is NOT assumed Hispanic;
    # those rows fall through to their race bucket (e.g. unknown-ethnicity White
    # -> Non-Hispanic White) rather than being mislabeled Latino.
    is_hispanic = eth.str.contains("hispanic", na=False) & ~eth.str.contains("not hispanic", na=False)
    is_white = race.str.contains(r"^white$", na=False)
    is_black = race.str.contains("black", na=False)
    is_asian = race.str.contains("asian", na=False)

    out = pd.Series("Other", index=df.index, dtype="string")

    out[is_hispanic] = "Latino"
    out[~is_hispanic & is_white] = "Non-Hispanic White"

    out[is_black] = "Black"
    out[is_asian] = "Asian"
    out[multiracial] = "Multiracial"

    return out
