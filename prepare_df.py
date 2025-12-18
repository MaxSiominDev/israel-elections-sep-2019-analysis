import pandas as pd

import arabic
import feature
import hebrew
import party

def calculate_valid_share(row):
    total = row['invalid_votes'] + row['valid_votes']
    return row['valid_votes'] / total if total > 0 else 0

def prepare_df() -> pd.DataFrame:
    df = pd.read_csv('votes per booth 2019b.csv', encoding='utf-8')

    df = df.drop(columns=hebrew.ignored_columns_list)
    df = df.rename(columns=hebrew.columns_translation)
    df = df.rename(columns=hebrew.winning_parties_translation)
    df['settlement_name'] = df['settlement_name'].replace(hebrew.settlements_translation)
    df[feature.is_arabic_settlement.name] = df['settlement_name'].apply(arabic.is_arabic_settlement_name)

    df[party.other_parties.name] = df[hebrew.losing_parties_list].sum(axis=1)
    df = df.drop(columns=hebrew.losing_parties_list)

    df[feature.valid_share.name] = df.apply(calculate_valid_share, axis=1)

    def sort_parties_by_desc(df_to_sort: pd.DataFrame) -> pd.DataFrame:
        non_party_columns = list(hebrew.columns_translation.values()) + [feature.is_arabic_settlement.name, feature.valid_share.name]
        party_columns = list(hebrew.winning_parties_translation.values())

        party_totals = df_to_sort[party_columns].sum().sort_values(ascending=False)

        sorted_columns = non_party_columns + list(party_totals.index)
        sorted_columns.append(party.other_parties.name)

        return df_to_sort[sorted_columns]

    df = sort_parties_by_desc(df)

    df['winner_party'] = df[party.party_columns].idxmax(axis=1)

    return df