import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from ingest_data import ingest_data

def explore_data(train_dir='cs-train', test_input='cs-production',output_dir='data_exploration_output'):
    '''
    Explores the training dataset by generating lineplots of revenue distribution for each country and a correlation heatmap of numerical features. 
    The generated visualizations are saved in the specified output directory.

    Args:
        train_dir (str): Directory path for the training dataset.
        test_input (str): Directory path for the test dataset.
    Rerturns:
        output_dir (str): Directory path to save the generated visualizations.
    '''
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Ingest the datasets
    datasets = ingest_data(train=train_dir, test=test_input)
    train_dict = datasets['train']
    test_dict = datasets['test']
    fig, ax = plt.subplots(figsize=(12,8))
    # Generate lineplots of revenue distribution for each country and save them individually
    for label,df in train_dict.items():
        tick_df = df.groupby('year_month')['date'].first().reset_index()
        plt.figure(figsize=(12,8))
        sns.lineplot(x='date',y='revenue',data=df,label=label) 
        plt.title(f'Revenue Distribution for {label} Dataset')
        plt.xticks(ticks=tick_df['date'], labels=tick_df['year_month'], rotation=45)
        plt.xlabel('Year Month')
        plt.ylabel('Revenue')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,f'{label}_revenue_distribution.png'))
        plt.close()
        if label == 'all':
            continue
        # Generate a combined lineplot of revenue distribution for all countries annd save it
        sns.lineplot(x='date',y='revenue',data=df,label=label,ax=ax) 
    ax.set_title('Revenue Distribution by Countries')
    ax.set_xticks(ticks=tick_df['date'], labels=tick_df['year_month'], rotation=45)
    ax.legend(title='Countries')
    ax.set_xlabel('Year Month')
    ax.set_ylabel('Revenue')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,'countries_revenue_distribution.png'))
    plt.close()

    # Generate a correlation heatmap of numerical features and save it
    corr = train_dict['all'][['purchases','unique_invoices','unique_streams','total_views','revenue']].corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,'numerical_feature_correlation_heatmap.png'))
    plt.close()

    print(f"Explotatory Data Analysis completed. Visualizations saved in '{output_dir}' directory.")


if __name__ == "__main__":
    explore_data()
