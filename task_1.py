import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.tree import export_graphviz
import pickle


# Function for Reading Data Set

def read_csv():
    df = pd.read_csv('car_data.csv')

    # Dropping all the NaN values as well as Call for price from the dataset

    df = df.dropna()
    df = df.drop(df[df['Price'] == 'Call for price'].index)

    return df


def model(data_frame):
    # Encoding the Categorical Values using Label Encoding
    label_encoder = preprocessing.LabelEncoder()

    columns_to_encode = ['Make', 'Model', 'Version', 'Assembly', 'Registered City']
    for column in columns_to_encode:
        data_frame[column] = label_encoder.fit_transform(data_frame[column])

    dependent_variable = data_frame['Price']
    independent_variable = data_frame[
        ['Make', 'Model', 'Version', 'Make_Year', 'CC', 'Assembly', 'Mileage', 'Registered City']]

    # Splitting the dataset into Training and Testing with 80% of the data as training and 20% as Testing

    X_train, X_test, y_train, y_test = train_test_split(independent_variable, dependent_variable, random_state=0,
                                                        test_size=0.2, shuffle=False)
    '''regression = DecisionTreeRegressor()
    regression.fit(X_train, y_train)

    with open('result_task_1.pkl', 'wb') as f:
        pickle.dump(regression, f)'''

    with open('result_task_1.pkl', 'rb') as f:
        res = pickle.load(f)
    print(y_test)
    print(res.predict(X_test))
    print('Score: %.2f' % res.score(X_test, y_test))
    export_graphviz(res, out_file='tree.dot',
                    feature_names=['Make', 'Model', 'Version', 'Make_Year', 'CC', 'Assembly', 'Mileage',
                                   'Registered City'])


data = read_csv()
model(data)
