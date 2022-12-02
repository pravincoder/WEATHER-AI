#Cleaning the data

#import the libraries 
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
'''
#Read the data in pandas dataframe
data = pd.read_csv("weatherHistory.csv")

#Drop(delete) unwanted columns in data
data= data.drop(['Formatted Date','Summary','Loud Cover','Daily Summary','Precip Type','Visibility (km)','Wind Bearing (degrees)'],axis=1)
data.to_csv('weather_final.csv')
'''
#read the data
data = pd.read_csv("weather_final.csv")

#Co-ordinates ,Label = Temp
X = data.drop(['Temperature (C)'],axis=1)
#Output Label
Y = data['Temperature (C)']
#Reshape into 2D vector
Y = Y.values.reshape(-1,1)

day_index = 45654
days = [i for i in range(Y.size)]

#LinearReg
clf = LinearRegression()
clf.fit(X,Y)

#Test with sample data
inp = np.array([[9.35],[7.22],[0.86],[14.2646],[1015.63]])
inp = inp.reshape(1,-1)

#Print Sample Output 
print('The Temperature in Celcius is:', clf.predict(inp))

#Graph
print('Temperature Graph: ')
plt.rcParams['lines.linestyle'] = '--'
plt.rc('lines', linewidth=1, linestyle='-.')
plt.plot(days,Y,color='g')
plt.scatter(days[day_index],Y[day_index],color='r')
plt.title("Temperature")
plt.xlabel('Days')
plt.ylabel('Temperature')

plt.show()

#Filtered Temperature using Apprant temp
x_f = X.filter(['Apparent Temperature (C)'],axis=1)
print('Apparent Temperature (C): ')
for i in range(x_f.columns.size):
    plt.rcParams['lines.linestyle'] = '--'
    plt.rc('lines', linewidth=1, linestyle='-.')
    plt.plot(3,2,i+1)
    plt.scatter(days,x_f[x_f.columns.values[i][:100]],color='g')
    plt.scatter(days[day_index],x_f[x_f.columns.values[i]][day_index],color='r')
    plt.title(x_f.columns.values[i])

#plot a graph with a Apprant Temp vs Temperature to observe the trends
plt.show()


#Filtered Temperature using humidity
x_f = X.filter(['Humidity'],axis=1)
print('Humidity:')
for i in range(x_f.columns.size):
    plt.rcParams['lines.linestyle'] = '--'
    plt.rc('lines', linewidth=1, linestyle='-.')
    plt.plot(3,2,i+1)
    plt.scatter(days,x_f[x_f.columns.values[i][:100]],color='g')
    plt.scatter(days[day_index],x_f[x_f.columns.values[i]][day_index],color='r')
    plt.title(x_f.columns.values[i])

#plot a graph with a Humidity vs Temperature to observe the trends
plt.show()


