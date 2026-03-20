import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re


def clean_text(text):
    """Converts text to lowercase and removes punctuation and numbers."""
    text = text.lower()
    text = re.sub(r'[\W_]+', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text


def extract_merchant_signals(cleaned_text, merchant_keywords):
    """Identifies the presence of specified merchant keywords in a cleaned text string.

    Returns a dict of boolean flags e.g. {'has_rent': True, 'has_tesco': False}.
    """
    return {f'has_{kw}': kw in cleaned_text for kw in merchant_keywords}


def transaction_type(df):
    """Returns one row per customer with boolean merchant-category flags."""
    merchant_keywords = ['rent', 'tesco', 'restaurant', 'salary', 'utility', 'transport', 'airbnb', 'payroll', 'bonus']
    subset = df[['transaction_id', 'customer_id', 'description']].copy()
    signals_df = subset['description'].apply(
        lambda x: pd.Series(extract_merchant_signals(x, merchant_keywords))
    )
    subset = pd.concat([subset, signals_df], axis=1)
    has_cols = [c for c in subset.columns if c.startswith('has_')]
    return subset[['customer_id'] + has_cols].groupby('customer_id').any().reset_index()


def analyze_transactions(df):
    """Calculates per-customer numeric features."""
    customer_features = df.groupby('customer_id').agg(
        num_transactions=('transaction_id', 'count'),
        total_debit=('amount', lambda x: x[df.loc[x.index, 'txn_type'] == 'debit'].sum()),
        total_credit=('amount', lambda x: x[df.loc[x.index, 'txn_type'] == 'credit'].sum()),
        average_transaction_amount=('amount', 'mean'),
    ).fillna(0)
    return customer_features


def merge_dataframes(transactions_df, labels_df):
    """Left-joins labels with transactions on customer_id."""
    return pd.merge(labels_df, transactions_df, on='customer_id', how='left')


if __name__ == '__main__':
    # --- Load ---
    transactions_df = pd.read_csv('data/transactions.csv')
    labels_df = pd.read_csv('data/labels.csv')
    print("Data loaded.")

    # Data quality check
    print(f"\nTransactions shape: {transactions_df.shape}")
    print(f"Null counts:\n{transactions_df.isnull().sum()}")
    print(f"Duplicate transaction_ids: {transactions_df['transaction_id'].duplicated().sum()}")

    # --- Clean text ---
    transactions_df['description'] = transactions_df['description'].apply(clean_text)
    print("\nDescriptions cleaned.")

    # --- Feature engineering ---
    merged_df = merge_dataframes(transactions_df, labels_df)
    customer_features_df = analyze_transactions(merged_df)
    transaction_type_df = transaction_type(merged_df)

    # --- Build training set ---
    training_df = pd.merge(labels_df, customer_features_df, on='customer_id', how='left')
    training_df = pd.merge(training_df, transaction_type_df, on='customer_id', how='left')
    print("\nTraining set created.")
    print(training_df.head())

    # --- Save ---
    os.makedirs('artifacts', exist_ok=True)
    training_df.to_csv('artifacts/training_set.csv', index=False)
    print("\nSaved to artifacts/training_set.csv")

    # --- EDA 1: Default distribution ---
    plt.figure(figsize=(6, 4))
    sns.countplot(x='defaulted_within_90d', data=training_df)
    plt.title('Distribution of Default Within 90 Days')
    plt.xlabel('Defaulted Within 90 Days')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['No Default (0)', 'Default (1)'])
    plt.tight_layout()
    plt.savefig('artifacts/eda_default_distribution.png')
    plt.close()

    # --- EDA 2: Merchant category presence ---
    has_cols = [c for c in training_df.columns if c.startswith('has_')]
    has_counts = training_df[has_cols].sum()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=has_counts.index, y=has_counts.values, hue=has_counts.index,
                palette='viridis', legend=False)
    plt.title("Customers per Merchant Category")
    plt.xlabel('Category')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('artifacts/eda_merchant_categories.png')
    plt.close()

    print("EDA plots saved to artifacts/")
