from enum import Enum


class FeatureType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class Feature:
    def __init__(self, name: str, type: FeatureType):
        self.name = name
        self.type = type

concentration = Feature("concentration", FeatureType.CATEGORICAL)
socio_econ_cluster_2015 = Feature("socio_econ_cluster_2015", FeatureType.CATEGORICAL)
is_arabic_settlement = Feature("is_arabic_settlement", FeatureType.CATEGORICAL)
is_stable = Feature("is_stable", FeatureType.CATEGORICAL)
settlement_name = Feature("settlement_name", FeatureType.CATEGORICAL)
valid_share = Feature("valid_share", FeatureType.NUMERIC)
voters = Feature("voters", FeatureType.NUMERIC)
invalid_votes = Feature("invalid_votes", FeatureType.NUMERIC)
valid_votes = Feature("valid_votes", FeatureType.NUMERIC)

all_features = [
    concentration,
    socio_econ_cluster_2015,
    is_arabic_settlement,
    voters,
    valid_votes,
    valid_share,
]

all_features_names = [feature.name for feature in all_features]
categorical_features_names = [feature.name for feature in all_features if feature.type == FeatureType.CATEGORICAL]
numeric_features_names = [feature.name for feature in all_features if feature.type == FeatureType.NUMERIC]
