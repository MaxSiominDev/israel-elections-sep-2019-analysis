import pandas as pd
import matplotlib.pyplot as plt

from party import party_columns, all_parties
import prepare_df

def visualize_votes_distribution_per_party_by_feature(df: pd.DataFrame, feature: str):
    party_cols = party_columns[::-1]
    df_clustered = df.dropna(subset=[feature])
    cluster_votes = (
        df_clustered
        .groupby(feature)[party_cols]
        .sum()
    )
    cluster_shares = cluster_votes.div(cluster_votes.sum(axis=1), axis=0)
    for cluster in cluster_shares.index.astype(int):
        shares = cluster_shares.loc[cluster]

        fig, ax = plt.subplots()
        bottom = 0
        handles = []

        for party in party_cols:
            value = shares[party]
            bar = ax.bar(
                cluster,
                value,
                bottom=bottom,
                label=party
            )
            bottom += value
            handles.append(bar[0])

        ax.set_title(f"Vote distribution - {feature} = {cluster}")
        ax.set_ylabel("Vote share")
        ax.set_xticks([cluster])
        ax.set_xticklabels([str(cluster)])

        ax.legend(
            handles[::-1],
            party_cols[::-1],
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )

        plt.tight_layout()
        plt.show()

def visualize_votes_distribution_per_ideology_by_feature(df: pd.DataFrame, feature: str):
    party_to_ideology = {
        p.name: p.ideology.name.lower()
        for p in all_parties
    }

    party_cols = party_columns[::-1]
    df_clustered = df.dropna(subset=[feature])

    cluster_party_votes = (
        df_clustered
            .groupby(feature)[party_cols]
            .sum()
    )

    cluster_party_votes.columns = [
        party_to_ideology[col] for col in cluster_party_votes.columns
    ]

    cluster_ideology_votes = (
        cluster_party_votes
            .groupby(axis=1, level=0)
            .sum()
    )

    cluster_shares = cluster_ideology_votes.div(
        cluster_ideology_votes.sum(axis=1),
        axis=0,
    )

    ideology_order = ["left", "center", "right"]

    for cluster in sorted(cluster_shares.index.astype(int)):
        shares = cluster_shares.loc[cluster]

        fig, ax = plt.subplots()
        bottom = 0

        for ideology in ideology_order:
            value = shares.get(ideology, 0)
            ax.bar(cluster, value, bottom=bottom, label=ideology)
            bottom += value

        ax.set_title(
            f"Vote distribution by ideology\n"
            f"socio_econ_cluster = {cluster}"
        )
        ax.set_ylabel("Vote share")
        ax.set_xticks([cluster])
        ax.set_xticklabels([str(cluster)])
        ax.legend()

        plt.tight_layout()
        plt.show()

def visualize(df: pd.DataFrame):
    df['voters'].hist(bins=200)
    plt.title('Voters per booth')
    plt.show()

    df['socio_econ_cluster_2015'].value_counts().sort_index().plot(kind='bar')
    plt.title('Social economical clusters')
    plt.show()

def analyze_df(df: pd.DataFrame):
    df.info()
    df.describe()
    visualize(df)
    visualize_votes_distribution_per_ideology_by_feature(df, "socio_econ_cluster_2015")

if __name__ == "__main__":
    analyze_df(prepare_df.prepare_df())
