# -*- coding: utf-8 -*-
"""House Prediction in Amsterdam.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17lT1VVRp6wY219dJ7J_Cuom8u19mzKVa

### **Prediksi Harga Rumah di Kota Amsterdam Mennggunakan Alogritma KNN, RF dan Bost**
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

from sklearn.preprocessing import  OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

"""Melakukan import Library yang dibutuhkan
Selanjutnya import dataset yang sudah di download
"""

df = pd.read_csv('/content/HousingPrices-Amsterdam-August-2021.csv')
df.head(10)

"""### Exploratory Data Analysis


"""

df.info()

df = df.dropna()

postal_code_district = pd.DataFrame(np.array([
    [1011, 1018, 1], [1019, 1019, 2], [1020, 1029, 3],
    [1030, 1039, 4], [1040, 1049, 5], [1050, 1059, 6], 
    [1060, 1069, 7], [1070, 1083, 8], [1086, 1099, 9], 
    [1100, 1108, 10], [1109, 1109, 11]]), 
    columns = ['under', 'upper', 'dstrct_id'])

postal_code_district

df['temp'] = df['Zip'].apply( lambda x: int(float((x.split(' ', 1)[0]))))
df['district'] = df['temp'].apply( lambda x: postal_code_district.loc[(postal_code_district['under'] <= x) & (postal_code_district['upper'] >= x), 'dstrct_id'].values[0])

"""membuat kolum baru 'district' dengan menggabungkan kolum zip dan temp"""

df['Price'] = df['Price']//1000
df = df.drop(['Zip', 'Address', 'temp', 'Unnamed: 0'], axis=1)
df.head(10)

"""merubah harga rumah kedalam ratus ribu (Dolar) karena jikta tidak kita rubah akan sulid melihat hasil akhirnya nanti. lalu membuang kolum yang tidak diperlukan"""

df[['district', 'Room']] = df[['district', 'Room']].astype(str).astype(object)

"""mengganti tipe data menjadi object pada kolum (Room dan district)"""

df.describe()

"""### **Mencari Missing Value dan Outlier**"""

df.isnull().sum()

sns.boxplot(x=df['Price'])

sns.boxplot(x=df['Area'])

sns.boxplot(x=df['Lon'])

sns.boxplot(x=df['Lat'])

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR=Q3-Q1
df=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]
 
df.shape

"""### **Univariate Analysis**"""

df.hist(bins=50, figsize=(15,10))
plt.show()

"""### **Categorical Features**"""

categorical_features = ['Room', 'district']

feature = categorical_features[0]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
new_df = pd.DataFrame({'Total Ruangan': count,
                   'persentase': percent.round(1)})
print(new_df)
count.plot(kind='bar', title=feature)

feature = categorical_features[1]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
new_df = pd.DataFrame({'Total Ruangan': count,
                   'persentase': percent.round(1)})
print(new_df)
count.plot(kind='bar', title=feature)

cat_features = df.select_dtypes(include='object').columns.to_list()
 
for col in cat_features:
  sns.catplot(x=col, y="Price", kind="bar", dodge=False, height = 3, aspect = 3,  data=df, palette="Set2")
  plt.title("Rata-rata 'Harga' Relatif terhadap - {}".format(col))

"""### **Multivariate Anlysis**"""

sns.pairplot(df, diag_kind = 'kde')

"""### **Corelation Matrix**"""

plt.figure(figsize=(12, 8))
correlation_matrix = df.corr().round(2)
 

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=12)

"""### **Encoding**"""

df = pd.concat([df, pd.get_dummies(df['Room'], prefix='Room')],axis=1)
df = pd.concat([df, pd.get_dummies(df['district'], prefix='district')],axis=1)
df.drop(['Room', 'district'], axis=1, inplace=True)
df.head(10)

sns.pairplot(df[['Lon', 'Lat']], plot_kws={"s": 2});

"""### **Data Preparation**"""

X = df.drop(["Price"],axis =1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""train test split

### **Standarisasi**
"""

numerical_features = ['Lon', 'Lat']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

X_train[numerical_features].describe().round(2)

"""### **Model Development**"""

models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=['Boosting', 'RandomForest', 'knn'])

"""**Random Forest**"""

RF = RandomForestRegressor(n_estimators=150, max_depth=16, random_state=100)
RF.fit(X_train, y_train)
 
models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

"""**Boosting Algortihm**"""

boosting = AdaBoostRegressor(random_state=100, learning_rate=0.001, n_estimators=50)
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)

"""**KN Neighbors**"""

knn = KNeighborsRegressor(n_neighbors=13)
knn.fit(X_train, y_train)
 
models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

"""### **Evaluasi Model**"""

mse = pd.DataFrame(columns=['train', 'test'], index=['RF','Boosting', 'KNN'])
 

model_dict = {'RF': RF, 'Boosting': boosting, 'KNN': knn}
 
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e4
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e4

mse

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

"""Nilai akurasi Model"""

knn_accuracy = knn.score(X_test, y_test)*100
rf_accuracy = RF.score(X_test, y_test)*100
boosting_accuracy = boosting.score(X_test, y_test)*100

list_evaluasi = [[knn_accuracy],
            [rf_accuracy],
            [boosting_accuracy]]
evaluasi = pd.DataFrame(list_evaluasi,
                        columns=['Accuracy (%)'],
                        index=['K-Nearest Neighbor', 'Random Forest', 'Boosting'])
evaluasi

prediksi = X_test.iloc[:20].copy()
pred_dict = {'y_true':y_test[:20]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)
 
pd.DataFrame(pred_dict)

def render_predict_table(data, col_width=7.0, row_height=0.625, font_size=12,
                     header_color='#82072c', row_colors=['#bad9db', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    predict_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    predict_table.auto_set_font_size(False)
    predict_table.set_fontsize(font_size)

    for k, cell in predict_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax.get_figure(), ax

fig,ax = render_predict_table(pd.DataFrame(pred_dict), header_columns=0, col_width=2.5)
fig.savefig("table_predict.png")