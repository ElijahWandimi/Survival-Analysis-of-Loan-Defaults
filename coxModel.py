# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from lifelines import CoxPHFitter
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# %%
loans_df = pd.read_csv('loan_level_500k.csv')
fets = ['CREDIT_SCORE','ORIGINAL_COMBINED_LOAN_TO_VALUE','ORIGINAL_DEBT_TO_INCOME_RATIO','DELINQUENT']
loans_df = loans_df[fets]
print(loans_df.shape)
loans_df.head(2)

# %%
numeric_types = ['float64', 'int64']
for col in loans_df.columns:
    if loans_df[col].dtype in numeric_types:
        loans_df[col].fillna(loans_df[col].median(), inplace=True)
    if loans_df[col].dtype not in numeric_types:
        loans_df[col].fillna(loans_df[col].mode()[0], inplace=True)

# %%
loans_df.isna().sum()

# %%
# create a new column for the default rate based on all three
dfrate = loans_df['DELINQUENT'].groupby([loans_df['CREDIT_SCORE'],loans_df['ORIGINAL_COMBINED_LOAN_TO_VALUE'],loans_df['ORIGINAL_DEBT_TO_INCOME_RATIO']]).mean()
loans_df['DFRATE_ALL'] = loans_df[['CREDIT_SCORE','ORIGINAL_COMBINED_LOAN_TO_VALUE','ORIGINAL_DEBT_TO_INCOME_RATIO']].apply(tuple, axis=1).map(dfrate)
loans_df.head(2)

# %%
# visualize the default rate  all
plt.figure(figsize=(10,6))
plt.scatter(loans_df['DFRATE_ALL'],loans_df['CREDIT_SCORE'],alpha=0.1)
plt.xlabel('Default Rate')
plt.ylabel('Credit Score')
plt.show()

# %%
loans_df.drop(['DELINQUENT'],axis=1,inplace=True)

# %% [markdown]
# ## CoxModel

# %%
# cox regression model
cph = CoxPHFitter()
cph.fit(loans_df, duration_col='CREDIT_SCORE', event_col='DFRATE_ALL', show_progress=True)
cph.print_summary()

# %%


# %%
# install sklearn
# %%
# split the data into training and testing y=default rate
X = loans_df.drop(['DFRATE_ALL'],axis=1)
y = loans_df['DFRATE_ALL']

# split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fit the model
lr = LinearRegression()
lr.fit(X_train,y_train)

# predict the model
y_pred = lr.predict(X_test)

# evaluate the model
print('Coefficients: ', lr.coef_)
print('Mean squared error: %.2f'
        % mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f'
        % r2_score(y_test, y_pred))


# %%
# visualize the linear regression model
fig, ax = plt.subplots(figsize=(10,6))
sns.regplot(y_test, y_pred, ax=ax)
ax.set_xlabel('Default Rate')
ax.set_ylabel('Predicted Default Rate')
plt.show()


