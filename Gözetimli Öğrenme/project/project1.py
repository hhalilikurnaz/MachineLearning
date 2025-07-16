#Univariate variable analysis => her bir değişkenin tek başına incelenmesi demek
#Basic Data Analysis
#Outlier Detection

#Visualization
#feature engineering
#modeling

#####------------Aşamalarımız bunlar---------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
import warnings
import os 
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid") 


#Load and check data 

df=pd.read_csv("tested.csv")
print(df.columns)

print(df.head(4))

print(df.describe())

print(df.info())





#Univariate variable analysis => her bir değişkenin tek başına incelenmesi demek

#==>> Kategorik veriler için barplot çizdiriyorum
''' 1111111111111111111

def bar_plot(variable):

    """
    input : variable ex : "sex"
    output : bar plot & value count
    """

    #get feature 
    var=df[variable]
    #count number of categorical variable(value/samples)
    varValue=var.value_counts()

    #visualize
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Frequency")

    plt.title(variable)
    plt.show()
    print("{}:\n {}".format(variable,varValue))


category1=["Survived","Sex","Pclass","Embarked","SibSp","Parch"]
for c in category1:
        bar_plot(c)



category2=["Cabin","Name","Ticket"]
for c in category2:
      print("{} \n".format(df[c].value_counts()))

11111111111111111111111111'''
#Numerical Variable ,
#the best way for analysis of numerical value is hsytogram
'''11111111111111111111111111111
def plot_hist(variable):
      plt.figure(figsize=(9,3))
      plt.hist(df[variable],bins=50)
      plt.xlabel(variable)
      plt.ylabel("Frequency")
      plt.title("{} distribution with hist".format(variable))
      plt.show() #==>>> histogram function

numeriVar=["Fare","Age"]
for n in numeriVar:
      plot_hist(n)
       111111111111111'''


#Basic Data Analysis

#yolcuların sınıflarına göre hayatta kalma olasılıkları
 #Pclass vs Survived
pcl_vs_survived=df[['Pclass',"Survived"]].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
print(pcl_vs_survived)

#faturenın önemli olup olmadıgına baktık
#Sex vs survived
sex_vs_survived=df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Sex',ascending=False)
print(sex_vs_survived)

#Parch vs Survied 
parch_vs_survied=df[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Parch',ascending=False)
print(parch_vs_survied)

#SiSp vs Survived
sibSp_vs_survied=df[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='SibSp',ascending=False)
print(sibSp_vs_survied)


#Outlier Detection => Aykırı Değer
#Outlier detection, veri setindeki uçuk kaçık, ortalamadan sapmış, garip değerleri tespit etmeye yarayan işlemdir.

'''   
Yöntem	                                    Açıklama
IQR (Interquartile Range)	--- %25-%75 arası verileri alır, dışındakileri uç değer sayar
Z-score	                  --      Ortalama ve standart sapma kullanır (
Boxplot	                   --           Görsel olarak aykırı değerleri hızlıca gösterir
Isolation Forest, DBSCAN	    --        Makine öğrenmesi ile daha akıllı outlier tespiti

'''

def detect_outliers(df,features):
    
    
    outlier_indices=[]



    for c in features:
        
        #1'st quartile
        Q1=np.percentile(df[c],25)
        #3'rd quartile
        Q3=np.percentile(df[c],75)
        #IQR
        IQR=Q3-Q1
        #Qutlier step
        outlier_step=IQR * 1.5  #dış sınır hesaplama 
        '''
        1.5 IQR → güvenli ve yaygın

        3 IQR → daha sert filtre

        1 IQR → çok hassas, fazla değer uçmuş gibi görünür
        '''
        #detect outlier and their indeces
        outlier_list_col=df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index # bu alt ve üst sınırın koşulları
        #store indeces
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices=Counter(outlier_indices)
    
    multiple_outliers=list(i for i,v in outlier_indices.items() if v >= 1)

    return multiple_outliers


outliers=df.loc[detect_outliers(df,["Age","SibSp","Parch","Fare"])]
print(outliers)

#outliers'ları çıkardım
df=df.drop(detect_outliers(df,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True)

#Missing Value 
#missing values 
x=df.isnull().sum()
print(x)

#Fare
print(df[df["Fare"].isnull()])

df.boxplot(column="Fare", by="Pclass")
plt.xticks(rotation=100)  # Çünkü kabin isimleri yatayda sığmaz
plt.title("Pclassa Göre Fare Dağılımı")
plt.suptitle("")  # Pandas otomatik başlık koymasın
plt.show()

#df['Fare']=df["Fare"].fillna(np.mean(df[df['Pclass']]==3)["Fare"])

#Visualization
list1=["SibSp","Parch","Age","Fare","Survived"]
sns.heatmap(df[list1].corr(), annot=True, fmt=".2f") 

plt.show()

g=sns.catplot(x="SibSp",y="Survived",data=df,kind="bar")
g.set_ylabels("Survived Probality")
plt.show()

'''h=sns.FacetGrid(df,col="Survived")
h.map(sns.displot,"Age",bins=25)
plt.show()


'''

age_missing=df[df['Age'].isnull()]
print(age_missing)

sns.catplot(x='Sex',y='Age',data=df,kind="box")
plt.show()


sns.catplot(x="Sex",y="Age",hue="Pclass",data=df,kind="box")
plt.show()



index_nan_age = list(df[df["Age"].isnull()].index)

# Her bir eksik yaş değeri için doldurma işlemi
for i in index_nan_age:
    sibsp_val = df.loc[i, "SibSp"]
    parch_val = df.loc[i, "Parch"]
    
    # Aynı SibSp ve Parch değerlerine sahip kişilerin yaş medyanı
    age_pred = df["Age"][
        (df["SibSp"] == sibsp_val) & 
        (df["Parch"] == parch_val) & 
        (df["Age"].notnull())
    ].median()
    
    # Genel yaş medyanı (yedek)
    age_med = df["Age"].median()

    # Eğer tahmin edilebildiyse onu yaz, yoksa genel medyanı kullan
    df.loc[i, "Age"] = age_pred if not np.isnan(age_pred) else age_med


#Feature Engineering 
print(df['Name'].head(10))
#insanların ünvanlarını isimlerinden ayırsam
#diyorum ki bu ünvanlar insanların hayatta kalıp kalmadığını ilgilendiriyor mu 

name=df['Name']
df['Title']=[i.split(".")[0].split(",")[-1].strip() for i in name]
print(df["Title"].head(10))

sns.countplot(x="Title",data=df)
plt.xticks(rotation=60)
plt.show()


#convert to categorical

df['Title']=df['Title'].replace(['Lady', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 
     'Sir', 'Jonkheer', 'Dona'],"other")


df['Title']=[0 if i == 'Master' else 1 if i == "Ms" or i =="Mlle" or i == 'Mrs' else 2 if i =='Mr' else 3 for i in df['Title']]
print(df['Title'].head(10))

sns.countplot(x='Title',data=df)
plt.xticks(rotation=70)
plt.show()

#artık name sutununa ihtiyacım yok title sutunu oluşturdum
df.drop(labels=["Name"],axis=1,inplace=True)
print(df.head(10))

#kategorik değerleri eğitime hazırlamak için dummies fonkisyonunu oluşturuyorum
df=pd.get_dummies(df,columns=['Title'])


#SibSp ve Parch topluyarak yeni bir feature ekliyoruz ve buna +1 ekliyoruz çünkü kendiside olucak boş olamaz

df['Fsize']=df['SibSp']+df['Parch']

a=sns.catplot(x="Fsize",y="Survived",data=df,kind="bar")
a.set_ylabels("Survival")
plt.show()

df=pd.get_dummies(df,columns=['Embarked'])

#Ticket bazıları numeric bazıları kategorik

tickets=[]
for i in list(df.Ticket):
    if not i.isdigit():
        tickets.append(i.replace(".","").replalace("/","").strip().split(" ")[0])
    else:
        tickets.append("x")
df['Ticket']=tickets


df=pd.get_dummies(df,columns='Ticket',prefix="T")
#başına T ekledik karışmasın 


df=pd.get_dummies(df,columns='Pclass')
df=pd.get_dummies(df,columns="Sex")


#PassangerId ile Cabin gereksiz 
df.drop(labels=['PasangerId','Cabin'],axis=1,inplace=True)


