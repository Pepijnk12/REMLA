import os
import sys
import tensorflow_data_validation as tfdv
import pandas as pd

df_train = pd.read_csv('data/external/train.tsv', sep='\t')
# Only keep the feature columns for the training data
df_train = df_train.drop(columns=['tags'])
df_test = pd.read_csv('data/external/test.tsv', sep='\t')

train_stats = tfdv.generate_statistics_from_dataframe(df_train)
schema = tfdv.infer_schema(train_stats)
test_stats = tfdv.generate_statistics_from_dataframe(df_test)
anomalies = tfdv.validate_statistics(test_stats, schema=schema)

if len(anomalies.anomaly_info) == 0:
    print('Data validation passed. No anomalies found between the testing data and the training data.')
    sys.exit(0)
else:
    print('Data validation failed. Found following anomalies:')
    print(anomalies)
    if not os.path.exists('output'):
        os.makedirs(os.getcwd() + '/output', exist_ok=True)
    tfdv.write_anomalies_text(anomalies, output_path='output/anomalies.txt')
    sys.exit(1)
