#!/usr/bin/env python
# coding: utf-8

# # DIAMONDS PROJECT

# Quirates (0.2--5.01)Partimos de un dataset que contiene el precio y las principales características de, aproximadamente, 54.000 diamantes. El objetivo de este proyecto es crear un modelo de regresión lineal con el fin de predicir el precio de un diamante teniendo en consideración las siguientes variables:
# 
# *   Price: Precio en USD (\$326--\$18,823)
# *   Carat: Quirates (0.2--5.01)
# *   Cut: Calidad del corte (Fair, Good, Very Good, Premium, Ideal)
# *   Color: color (de J (peor) a D (mejor)).
# *   Clarity: calidad (I1 (pitjor), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (millor))
# *   X: Longitud X en mm (0--10.74)
# *   Y: Anchura Y en mm (0--58.9)
# *   Z: Profundidad z en mm (0--31.8)
# *   Table: Porcentaje de la profundidad total (43--79).
# *   Depth: Anchura de la parte superior del diamante en relación con el punto más ancho (43-95)

# ## Importar librerias

# In[578]:


#Importar las librerias necesarias
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import scipy.stats as stats
import numpy as np    # numerical python, algebra library
import matplotlib.pyplot as plt
import mpl_toolkits
import pycountry_convert
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.linear_model import Lasso       # L1
from sklearn.linear_model import Ridge       # L2
from sklearn.linear_model import ElasticNet  # L1+L2
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_squared_error as ms
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from xgboost import XGBRegressor as XGBR
from lightgbm import LGBMRegressor as LGBMR
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
import warnings 
warnings.simplefilter('ignore')


# In[579]:


#Primero, importamos el dataset
diamonds = pd.read_csv("./diamonds.csv")
print(diamonds.head(10)) #Se muestras las 10 primas filas del dataset
print("Tamaño del dataset: " + str(diamonds.shape)) #filas x columnas


# ## DATA ANALYSIS

# In[580]:


#El precio será la variable a precedir. Quiero saber cual es el precio medio de los diamantes del dataset
diamonds['price'].mean()


# In[581]:


#Cambio de nombre la variable Y por anchura para no confundirme con la variable target en el proceso de predicción
diamonds.rename(columns = {'y':'anchura'}, inplace = True)  


# In[582]:


diamonds


# In[583]:


#Qué información nos proporciona el dataset?
diamonds.info()


# In[584]:


#Analizar mejor la información que nos proporciona el dataset
diamonds.describe()


# In[585]:


#Cuantos diamantes hay según el número de quirates (carat)?
diamonds.hist(column="carat",        # Column to plot
              figsize=(8,8),         # Plot size
              bins=20,
              color="black");         # Plot color


# In[586]:


#Analizo las variables categóricas
diamonds['color'].value_counts()


# In[587]:


diamonds['cut'].value_counts()


# In[588]:


diamonds['clarity'].value_counts()


# In[589]:


#Separamos variables categoricas y numericas
data_for= diamonds
num_cols=[c for c in data_for.columns if (data_for[c].dtype!='object')]
catego_cols=[c for c in data_for.columns if (data_for[c].dtype=='object')]


# In[590]:


#Analizamos las variables numericas a partir de boxplot
for i in range(len(num_cols)):
    plt.figure()
    sns.boxplot(x=diamonds[num_cols[i]])
    plt.title(num_cols[i])
    plt.show()


# ## DATA CLEANING

# In[591]:


#Búqueda de null-values
pd.set_option('display.max_columns',None)
for i in diamonds.columns:
    print(i)
    print(i,diamonds[i].isna().sum())
#Se concluye que no hay null-values.


# In[592]:


#Teniendo en consideración la tabla anterior donde aparecen los valores mínimos y máximos de las variables. Se observa que hay diamantes cuyos valores de X,Y,Z es 0. 
#Lo analizo:
diamonds.loc[(diamonds['x']==0) | (diamonds['anchura']==0) | (diamonds['z']==0)]


# In[593]:


#No tiene sentido que un diamante no tenga anchura, profundidad o longitud. Por lo tanto, elimino todos esos diamantes cuyos valores de X,y,Z sea 0.
diamonds =  diamonds[(diamonds[['x','anchura','z']] != 0).all(axis=1)]


# In[594]:


diamonds.iloc[:,:]


# In[595]:


#Como hemos observado hay outlieres en algunas variables, por lo tanto, se decide eliminar outliers en todas las caracteristicas con el fin de reducir la dispersión de valores
#find absolute value of z-score for each observation
z = np.abs(stats.zscore(diamonds.select_dtypes(np.number))) #Todas las filas y las columnas de la 2 a la última


# In[596]:


#only keep rows in dataframe with all z-scores less than absolute value of 3 
diamonds_clean = diamonds[(z<3).all(axis=1)]

#find how many rows are left in the dataframe 
print(diamonds_clean.shape)
print(diamonds.shape)


# In[597]:


#Una vez hecho el data cleaning procedo analizar la correlación
#Qué variables están más correlacionadas entre ellas?
plt.figure(figsize=(10,10))
corr = diamonds_clean.corr()
sns.heatmap(data=corr, square=True , annot=True, cbar=True,linewidth=2)


# Concluciones extraidas a partir de la matriz de correlación:
# 
# *   Correlacion muy alta (0,92) entre price y carat.
# *   Correlació alta (0,89, 0,87 y 0,87 respectivamente) entre price y X/Y/Z.
# 
# También, la correlación entre las variables X, Y y Z es alta... Podríamos unir las 3 variables en una sola "feature" que indique el volumen del diamante? 
# 

# In[598]:


#Mostrar gráficamente la relación entre "carat" y "price".
plt.scatter(diamonds.carat,diamonds.price)
plt.xlabel("Quirates")
plt.ylabel("Precio")


# In[599]:


#Para pasar de variables categoricas a numericas no he hecho onehotencoding sino que he decidido crear un diccionario para ponderar las variables, y a posteriori, convertir un unico valor de test para la demostración en tiempo real.
ordinal_dict = {
    'color': {
        'D': 0,
        'E': 1,
        'F': 2,
        'G': 3,
        'H': 4,
        'I': 5,
        'J': 6},
    'cut': {
        'Ideal': 0,
        'Premium': 1,
        'Very Good': 2,
        'Good': 3,
        'Fair':4},
    'clarity':{
        'IF': 0,
        'VVS1': 1,
        'VVS2': 2,
        'VS1': 3,
        'VS2': 4,
        'SI1': 5,
        'SI2': 6,
        'I1': 7}
        }


# In[600]:


#Transformamos las variables categoricas en numericas.
for column in catego_cols:
    diamonds_clean[column] = diamonds_clean[column].map(ordinal_dict[column])

diamonds_clean_nonorm = diamonds_clean
diamonds_clean_nonorm.sample()


# ## PREDICTION

# In[601]:


diamonds_clean = diamonds_clean_nonorm.copy()
scalers = {}
for c in diamonds_clean:  
    scaler = StandardScaler()
    diamonds_clean[c]=scaler.fit_transform(diamonds_clean[c].values.reshape(-1, 1))
    scalers.update({c: scaler})


# In[602]:


#Primero, hemos de separar las variables entre train y test. Para hacerlo, separamos la variable "target" (price) del resto. 
X = diamonds_clean.drop(['price'],axis=1)
y = diamonds_clean_nonorm['price']


# In[603]:


#Dividimos las muestras en conjunto de entrenamiento ( se utilizarán el 90%) y de test el 10%.
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.1, random_state=42)  # random state fixed sample
X_train.head(5)


# In[604]:


linreg=LinReg()    # model
linreg.fit(X_train, y_train)   # model train
y_pred_linreg=linreg.predict(X_test)   # model prediction

# Lasso L1

lasso=Lasso()
lasso.fit(X_train, y_train)

y_pred_lasso = lasso.predict(X_test)

# Ridge L2

ridge=Ridge()
ridge.fit(X_train, y_train)

y_pred_ridge = ridge.predict(X_test)

# ElasticNet L1+L2

elastic=ElasticNet()
elastic.fit(X_train, y_train)

y_pred_elastic = elastic.predict(X_test)

rfr=RFR()
rfr.fit(X_train, y_train)

y_pred_rfr = rfr.predict(X_test)


# In[605]:


xgbr=XGBR()
xgbr.fit(X_train, y_train)

y_pred_xgbr = xgbr.predict(X_test)


# In[606]:


lgbmr=LGBMR()
lgbmr.fit(X_train, y_train)
LGBMR_MODEL = lgbmr
y_pred_lgbmr = lgbmr.predict(X_test)


# In[607]:


models = [linreg, lasso, ridge, elastic, rfr, xgbr, lgbmr]
model_names = ['linreg', 'lasso', 'ridge', 'elastic', 'rfr', 'xgbr', 'lgbmr']
preds=[y_pred_linreg, y_pred_lasso, y_pred_ridge, y_pred_elastic, y_pred_rfr, y_pred_xgbr, y_pred_lgbmr]


# In[644]:


train_score = [0]*7
test_score = [0]*7
train_mse = [0]*7
test_mse = [0]*7
train_rmse = [0]*7
test_rmse = [0]*7
train_mae = [0]*7
test_mae = [0]*7

for i in range(len(models)):

    train_score[i] = models[i].score(X_train, y_train) #R2
    test_score[i] = models[i].score(X_test, y_test)
    train_mse[i]=mse(models[i].predict(X_train), y_train) #MSE
    test_mse[i]=mse(preds[i], y_test)
    train_rmse[i]=mse(models[i].predict(X_train), y_train)**0.5 #RMSE
    test_rmse[i]=mse(preds[i], y_test)**0.5
    train_mae[i]=mae(models[i].predict(X_train), y_train) #MAE
    test_mae[i]=mae(preds[i], y_test)
    print ('Model: {}, train MSE: {} -- test MSE: {}'.format(model_names[i], train_mse[i], test_mse[i]))
    print ('Model: {}, train R2: {} -- test R2: {}'.format(model_names[i], train_score[i], test_score[i]))
    print ('Model: {}, train RMSE: {} -- test RMSE: {}'.format(model_names[i], train_rmse[i], test_rmse[i]))
    print ('Model: {}, train MAE: {} -- test MAE: {}'.format(model_names[i], train_mae[i], test_mae[i]))


# In[609]:


courses = list(model_names)
values = list(test_score)

plt.bar(courses, values, color ='maroon',
        width = 0.4)
plt.ylim([0.8,1])
plt.title('R2')

plt.figure()
values = list(test_rmse)

plt.bar(courses, values, color ='maroon',
        width = 0.4)
plt.title('RMSE')

plt.figure()
values = list(test_mae)

plt.bar(courses, values, color ='maroon',
        width = 0.4)
plt.title('MAE')


# Se concluye que el mejor modelo de predicción es el rfr con un R2 de 0,984 y MAE:230,11.

# In[610]:


Result= pd.DataFrame({'Actual Price':y_test,'Predicción Lgbmr': y_pred_lgbmr})
Result


# In[611]:


Result['Actual Price'].mean()


# In[612]:


#Para finalizar el ejercicio, me ha parecido interesante mostrar de forma gráfica el pequeño margen de error que hemos obtenido con nuestro modelo.
#De color naraja es nuestra prediccion, y de color azul son las variables "test" del modelo.
plt.plot(y_pred_lgbmr) 
plt.plot(y_test.values)

plt.xlabel("Predicted price ($)")
plt.ylabel("Real price ($)")


# ### K-FOLD CROSS VALIDATION

# **K-FOLD CROSS VALIDATION**
# Para estar segura del score obtenido, he decidio hacer el análisis cross-validation ya que es mucho más robusto en comparación con la predicción anterior. Pues, dicha tecnica sirve para evaluar los resultados de un análisis estadístico y garantizar que son independientes de la partición entre los datos de entrenamiento y prueba. Por lo tanto, me aseguro que el score obtenido no es debido a una selección azarosa de los conjuntos `train` / `test` que benecifie el resultado. 

# In[613]:


# Validación cruzada k fold
modelo = RFR()
kfold_validacion = KFold(10) # indicamos cuantos fold queremos. En nuestro caso elegimos 10


# In[614]:


from sklearn.model_selection import cross_val_score
resultados = cross_val_score(modelo, X,y,cv = kfold_validacion,scoring='neg_mean_absolute_error')
print(resultados)
pd.Series(resultados).median() # para ver la mediana de los resultados 


# In[615]:


plt.boxplot(resultados)


# Es la mediana del error del modelo elegido para predecir el precio de mercado de los diamantes. 

# #### UNSUPERVISED LEARNING: K-MEANS
# Se decidie realizar K-means para observar si las variables del dataset se pueden agrupar, y en consecuencia, obtener un mejor análisis de las mismas

# In[616]:


scalers = {}
for c in diamonds_clean.drop(columns='price'):  
    scaler = StandardScaler()
    diamonds_clean[c]=scaler.fit_transform(diamonds_clean[c].values.reshape(-1, 1))
    scalers.update({c: scaler})


# In[617]:


# Plot the random blub data
plt.figure(figsize=(6, 6))
plt.scatter(X.values[:, 3], X.values[:, 5], s=5)
plt.title(f"No Clusters Assigned")


# In[618]:


# Plot the data and color code based on clusters
# changing the number of clusters 
random_state = 42

for i in range(1,9):
    plt.figure(figsize=(6, 6))
    # Predicting the clusters
    y_pred = KMeans(n_clusters=i, random_state=random_state).fit_predict(X)
# plotting the clusters
    plt.scatter(X.values[:, 3], X.values[:, 5], c=y_pred, s=5)
    plt.title(f"Number of Clusters: {i}")

plt.show();


# In[619]:


km = KMeans(n_clusters=i, random_state=random_state)
km.fit(X)
km.inertia_


# In[620]:


# Calculating the inertia and silhouette_score
inertia = []
sil = []
# changing the number of clusters 
for k in range(2,11):
    
    km = KMeans(n_clusters=k, random_state=random_state)
    km.fit(X)
    y_pred = km.predict(X)
    
    inertia.append((k, km.inertia_))
    sil.append((k, silhouette_score(X, y_pred)))


# In[621]:


fig, ax = plt.subplots(1,2, figsize=(12,4))
# Plotting Elbow Curve
x_iner = [x[0] for x in inertia]
y_iner  = [x[1] for x in inertia]
ax[0].plot(x_iner, y_iner)
ax[0].set_xlabel('Number of Clusters')
ax[0].set_ylabel('Intertia')
ax[0].set_title('Elbow Curve')
# Plotting Silhouetter Score
x_sil = [x[0] for x in sil]
y_sil  = [x[1] for x in sil]
ax[1].plot(x_sil, y_sil)
ax[1].set_xlabel('Number of Clusters')
ax[1].set_ylabel('Silhouetter Score')
ax[1].set_title('Silhouetter Score Curve')


# Se concluye que los datos analizados no se pueden agrupar. Pues, tal y como se observa en los gráficos anteriores no hay codo en la gráfica de la izquierda ("Elbow curve") y no hay máximo en el gráfico de la derecha ("Silhouetter Score Curve")..

# Hacemos K-means con la variable volume

# In[622]:


#Creamos la variable volumen a partir de X,Y y Z ya que estan muy correlacionadas.
diamondscluster = diamonds_clean.copy()
diamondscluster['volume'] = diamondscluster['x']*diamondscluster['anchura']*diamondscluster['z']
diamondscluster.drop(['x','anchura','z'], axis=1, inplace= True)
diamondscluster.head(5)


# In[623]:


scalers2 = {}
for c in diamondscluster.drop(columns='price'): 
    scaler2 = StandardScaler()
    diamondscluster[c]=scaler2.fit_transform(diamondscluster[c].values.reshape(-1, 1))
    scalers2.update({c: scaler2})


# In[624]:


diamondscluster.head()


# In[625]:


# Plot the random blub data
plt.figure(figsize=(6, 6))
plt.scatter(X.values[:, 1], X.values[:, 3], s=5)
plt.title(f"No Clusters Assigned")


# In[626]:


X


# In[627]:


# Plot the data and color code based on clusters
# changing the number of clusters 
random_state = 42

for i in range(1,9):
    plt.figure(figsize=(6, 6))
    # Predicting the clusters
    y_pred2 = KMeans(n_clusters=i, random_state=random_state).fit_predict(X)
# plotting the clusters
    plt.scatter(X.values[:, 1], X.values[:, 3], c=y_pred2, s=5)
    plt.title(f"Number of Clusters: {i}")

plt.show();


# In[628]:


km2 = KMeans(n_clusters=i, random_state=random_state)
km2.fit(X)
km2.inertia_


# In[629]:


# Calculating the inertia and silhouette_score
inertia2 = []
sil2 = []
# changing the number of clusters 
for k in range(2,11):
    
    km2 = KMeans(n_clusters=k, random_state=random_state)
    km2.fit(X)
    y_pred2 = km2.predict(X)
    
    inertia2.append((k, km2.inertia_))
    sil2.append((k, silhouette_score(X, y_pred2)))


# In[630]:


fig, ax = plt.subplots(1,2, figsize=(12,4))
# Plotting Elbow Curve
x_iner2 = [x[0] for x in inertia2]
y_iner2  = [x[1] for x in inertia2]
ax[0].plot(x_iner2, y_iner2)
ax[0].set_xlabel('Number of Clusters')
ax[0].set_ylabel('Intertia')
ax[0].set_title('Elbow Curve')
# Plotting Silhouetter Score
x_sil2 = [x[0] for x in sil2]
y_sil2  = [x[1] for x in sil2]
ax[1].plot(x_sil2, y_sil2)
ax[1].set_xlabel('Number of Clusters')
ax[1].set_ylabel('Silhouetter Score')
ax[1].set_title('Silhouetter Score Curve')


# Se observa un codo en el Elbow Curve en el número 4. Por lo tanto, se decide agrupar las variables en 4 clusters. 

# In[631]:


km2 = KMeans(n_clusters=4, random_state=random_state)
km2.fit(X)
y_pred_k4 = km2.predict(X)


# In[632]:


np.unique(y_pred_k4)


# In[633]:


X


# In[634]:


X['y'] = y
X['cluster'] = y_pred_k4


# In[635]:


y


# In[636]:


#Separamos las variables de train y de test
X0 = X[X['cluster']==0]
X1 = X[X['cluster']==1]
X2 = X[X['cluster']==2]
X3 = X[X['cluster']==3]

y0 = X[X['cluster']==0]['y']
y1 = X[X['cluster']==1]['y']
y2 = X[X['cluster']==2]['y']
y3 = X[X['cluster']==3]['y']


# In[637]:


#Mostrar gráficamente las características de los diamantes para cada cluster
for i in range(len(X.columns)):
    plt.figure()
    sns.scatterplot(data=X, x=X.columns[i], y="cluster", hue="cluster")


# In[638]:


# Vuelvo a realizar la validación cruzada k fold
lgbmr=LGBMR()
kfold_validacion = KFold(10) # indicamos cuantos fold queremos. En nuestro caso elegimos 10
from sklearn.model_selection import cross_val_score
resultados = cross_val_score(modelo, X,y,cv = kfold_validacion,scoring='neg_mean_absolute_error')
print(resultados)
pd.Series(resultados).median() # para ver el promedio de los resultados 


# # Demo

# In[639]:


#Determinar las características del diamante para determinar su precio.
datauser = pd.DataFrame()

datauser['carat']=[float(input('Introduce cuántos quirates tiene el diamante (0.2--5.01)'))]
datauser['cut'] =[str(input('Introduce la calidad del corte (Fair, Good, Very Good, Premium, Ideal)'))]
datauser['color'] =[str(input('Introduce el tipo de color (De D (mejor) a J (peor)'))]
datauser['clarity'] =[str(input('Introduce la calidad del diamante ( IF (mejor),VVS1,VVS2,VS1,VS2,SI1,SI2,I1(peor))'))]
datauser['depth']=[float(input('Introduce la anchura de la parte superior del diamante en relación con el punto más ancho del mismo (45--95)'))]
datauser['table']=[float(input('Introduce el porcentaje de la profundidad total del diamante (43--79)'))]
datauser['x']=[float(input('Introduce la longitud en mm (0--10.74)'))]
datauser['y']=[float(input('Introduce la anchura en mm (0--58.9)'))]
datauser['z']=[float(input('Introduce la profundidad en mm (0--31.8)'))]


# In[645]:


for column in catego_cols:
    datauser[column] = datauser[column].map(ordinal_dict[column])


# In[646]:


vector_topredict = scaler.transform(datauser.values.reshape(-1, 1))
for i in range(len(datauser.columns)):
    datauser[datauser.columns[i]] = vector_topredict[i]


# In[647]:


price = LGBMR_MODEL.predict(datauser)
print('The price to be bought is:',np.round(price[0],2),'$')


# In[648]:


datauser.to_csv('Diamantesfinal.csv', index = False)


# In[ ]:




