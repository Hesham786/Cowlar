import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans

# Function for Reading Data Set
from sklearn.preprocessing import StandardScaler


def read_csv():
    data = pd.read_csv('movies_dataset.csv')
    new_data = data.drop(['title'], axis=1)
    data['average_rating'] = new_data.mean(axis=1)

    return data


def k_means(data2):
    inertias = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data2[['average_rating']])
        inertias.append(kmeans.inertia_)

    plt.plot(range(1, 11), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data2[['average_rating']])

    data2['clusters'] = kmeans.fit_predict(data2[['average_rating']])

    plt.scatter(data2['title'], data2['average_rating'], c=kmeans.labels_)
    plt.show()

    return data2


def recommended_movie(data2):
    data = k_means(data2)

    movie = input("Enter Movie Name:")
    recommendation = ''
    for index, row in data.iterrows():
        if movie in row['title']:
            print('Following is the List of Recommended Movies')
            recommendation = data.loc[data['clusters'] == row['clusters'], 'title'].values
    for index, i in enumerate(recommendation):
        print(index, i)


data1 = read_csv()
recommended_movie(data1)
