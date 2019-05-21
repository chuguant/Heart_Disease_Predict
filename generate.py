import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


column_names = [
    'age',
    'sex',
    'chest pain type',
    'resting blood pressure',
    'serum cholestoral',
    'fasting blood sugar',
    'resting electrocardiographic results',
    'maximum heart rate',
    'angina',
    'oldpeak = ST depression',
    'the slope of the peak exercise ST segment',
    'number of major vessels (0-3) colored by flourosopy',
    'thal',
    'target'
]


df = pd.read_csv('./processed.cleveland.data', header=None,
                 na_values='?', names=column_names)


def stat(df, col='resting blood pressure', stat='mean'):
    df_temp = df[['age', 'sex', col]].groupby(['age', 'sex'])
    if stat == 'mean':
        df_temp = df_temp.mean().reset_index()
    elif stat == 'min':
        df_temp = df_temp.min().reset_index()
    elif stat == 'max':
        df_temp = df_temp.max().reset_index()
    else:
        assert False, 'no stat'

    df_male = df_temp[df_temp['sex'] == 1.0].set_index(
        'age').drop(columns='sex')
    df_female = df_temp[df_temp['sex'] == 0.0].set_index(
        'age').drop(columns='sex')
    df_total = df_male.join(df_female, how='outer',
                            lsuffix=' (male)', rsuffix=' (female)', sort=True)

    return df_total.rename(columns={
        col + ' (male)': stat + ' ' + col + ' (male)',
        col + ' (female)': stat + ' ' + col + ' (female)',
    })


if not os.path.exists('images'):
    os.makedirs('images')

print('***** Task1 *****')

numerical_columns = [
    'resting blood pressure',
    'serum cholestoral',
    'maximum heart rate',
    'oldpeak = ST depression'
]

for col in numerical_columns:
    print(col)
    df_total = stat(df, col, 'mean')

    fig, axs = plt.subplots(3, 1, figsize=(15, 15))
    df_total[['mean '+col+' (male)', 'mean '+col+' (female)']].plot(ax=axs[0])
    axs[0].set_ylabel('mean '+col)

    df_total[['mean '+col+' (male)', 'mean '+col +
              ' (female)']].plot.bar(ax=axs[1])
    axs[1].set_ylabel('mean '+col)

    df_total[['mean '+col+' (male)', 'mean '+col +
              ' (female)']].boxplot(ax=axs[2])
    axs[2].set_ylabel('mean '+col)

    plt.savefig('images/{}.jpg'.format(col))


categorical_columns = {
    'chest pain type': {
        1: 'typical angin',
        2: 'atypical angina',
        3: 'non-anginal pain',
        4: 'asymptomatic',
    },
    'fasting blood sugar': {
        0: 'No',
        1: 'Yes',
    },
    'resting electrocardiographic results': {
        0: 'normal',
        1: 'having ST-T wave abnormality',
        2: 'showing probable or definite left ventricular hypertrophy',
    },
    'angina': {
        0: 'No',
        1: 'Yes',
    },
    'the slope of the peak exercise ST segment': {
        1: '1',
        2: '2',
        3: '3',
    },
    'number of major vessels (0-3) colored by flourosopy': {
        0: '0',
        1: '1',
        2: '2',
        3: '3',
    },
    'thal': {
        3: 'normal',
        6: 'fixed defect',
        7: 'reversable defect',
    },
}

for col_name, maps in categorical_columns.items():
    print(col_name)

    dfs = {
        'total': df.groupby(['age', 'sex'])[col_name].count()
    }
    for key, value in maps.items():
        dfs[value] = df.groupby(['age', 'sex'])[col_name].agg(lambda x: sum(x==key))

    df2 = pd.DataFrame(dfs)

    for name in maps.values():
        df2[name] = df2[name] / df2['total'] * 100

    df2 = df2.drop(columns='total')

    df2 = df2.reset_index()
    df2_male = df2[df2['sex']==1].drop(columns='sex').set_index('age')
    df2_female = df2[df2['sex']==0].drop(columns='sex').set_index('age')

    fig, axs = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    df2_male.plot.bar(ax=axs[0])
    axs[0].set_title('male')
    axs[0].set_ylabel('percentage (%)')
    df2_female.plot.bar(ax=axs[1])
    axs[1].set_title('female')
    axs[1].set_ylabel('percentage (%)')
    plt.savefig('images/{}.jpg'.format(col_name))


# task2
print('Start ')
df_new = df.copy()
df_new.loc[df_new['target'] != 0, 'target'] = 1
df_new = df_new.dropna()
df_new.head()

print('***** Task2 *****')
for col in df_new.columns:
    if col == 'target':
        continue
    print(col, np.corrcoef(df_new[col].values, df_new['target'].values)[1, 0])


print('---- Statistical method ----')
X = df_new.iloc[:, 0:13]  # independent columns
y = df_new.iloc[:, -1]  # target column
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']
print(featureScores.nlargest(10, 'Score'))


print('---- Decision Tree Feature Importance ----')
X = df_new.iloc[:, 0:13]  # independent columns
y = df_new.iloc[:, -1]  # target column
model = ExtraTreesClassifier()
model.fit(X, y)
print(list(zip(X.columns, model.feature_importances_)))

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10)[::-1].plot(kind='barh')
plt.title('Feature Importance')
plt.savefig('images/feature_importance.jpg')


print('---- Correlation Matrix with Heatmap ----')
X = df_new.iloc[:, 0:13]  # independent columns
y = df_new.iloc[:, -1]  # target column

# get correlations of each features in dataset
corrmat = df_new.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
# plot heat map
g = sns.heatmap(df_new[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.title('Correlation heat map')
plt.savefig('images/corr.jpg')
