import pandas
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay



# Press the green button in the gutter to run the script.

def lin_reg_project():
    league_data = pandas.read_csv('League_of_Legends_stats.csv')
    league_data['Gold'] = league_data['Gold'].map(lambda s: int(s.replace(",", "")))

    league_no_support = league_data[league_data['CS'] >= 125]
    league_no_support = league_no_support.sort_values(by='CS', ascending=True)
    plot = league_no_support.plot.scatter('CS', 'Gold', s=10, figsize=(15, 15))
    lin_reg = LinearRegression()
    lin_reg.fit(league_no_support['CS'].values.reshape(-1, 1), league_no_support['Gold'].values.reshape(-1, 1))
    line = lin_reg.predict(league_no_support['CS'].values.reshape(-1, 1))
    print("Gold =", lin_reg.coef_[0][0], "cs +", lin_reg.intercept_[0])
    plt.xlabel("Creep Score")
    plt.title("Relationship between Creep Score and Gold")
    league_no_support['Gold Prediction'] = line
    plot.add_line(matplotlib.lines.Line2D(league_no_support['CS'], league_no_support['Gold Prediction']))
    plt.show()
    league_no_support['diff'] = abs(league_no_support['Gold'] - league_no_support['Gold Prediction'])
    league_no_support['diff percentage'] = league_no_support['diff'] / league_no_support['Gold'] * 100
    print("Is on average off by", np.average(league_no_support['diff']), "gold")
    print("Is on average off by", np.average(league_no_support['diff percentage']), "%")




def outlier_project():
    league_data = pandas.read_csv('League_Worlds_2022.csv')
    sns.set_theme()
    sns.displot(data=league_data['Avg deaths']).set(title="Average Deaths", xlabel="Average Deaths",
                                                          ylabel="Number of Players")

    plt.show()
    mean = league_data['Avg deaths'].mean()
    std = league_data['Avg deaths'].std()
    bot = mean - 3 * std
    top = mean + 3 * std
    print("Mean:", mean)
    print("Standard Dev:", std)
    print("Upper Limit:", top)
    print("Lower Limit:", bot)
    outliers = league_data[(league_data['Avg deaths'] < bot) | (league_data['Avg deaths'] > top)]
    print(outliers)


def fake_news_project():
    news = pandas.read_csv('News.csv', index_col=0)
    news = news.sample(frac=1)
    news.reset_index(inplace=True)
    news.drop(['title'], axis=1)
    print(news.shape)
    sns.countplot(data=news,
                  x='label',
                  order=news['label'].value_counts().index)
    plt.xlabel("Type of News")
    plt.title("Distribution of Data")
    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(news['text'],
                                                        news['label'],
                                                        test_size=0.25, random_state=10)
    vectorizer = TfidfVectorizer(stop_words='english', max_df=.9)
    train = vectorizer.fit_transform(x_train)
    test = vectorizer.transform(x_test)
    classifier = PassiveAggressiveClassifier(max_iter=100)
    classifier.fit(train, y_train)
    pred = classifier.predict(test)
    print('Accuracy', accuracy_score(y_test, pred) * 100)
    cm = confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['FAKE', 'REAL'])
    disp.plot()
    plt.show()


if __name__ == '__main__':
    fake_news_project()
    #lin_reg_project()
    #outlier_project()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
