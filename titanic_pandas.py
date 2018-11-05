import pandas
import numpy as np

data = pandas.read_csv('data/titanic.csv', index_col='PassengerId')


def calc_sex(d):
    """
    Calculate male and female passengers
    :param d: data frame
    :return: counters
    """
    sex_count = d['Sex'].value_counts()
    return sex_count.index[0], sex_count[0], sex_count.index[1], sex_count[1]


def calc_surv_perc(d):
    """
    Calculate percentage of survived passengers
    :param d: data frame
    :return: rounded percents
    """
    surv = np.sum(d['Survived'] == 1)
    return round(surv / d['Survived'].count() * 100, 2)


def calc_first_class(d):
    """
    Calculate percentage of first-class passengers
    :param d: data frame
    :return: rounded percents
    """
    fst = np.sum(d['Pclass'] == 1)
    return round(fst / d['Pclass'].count() * 100, 2)


def get_mean_and_med_age(d):
    """
    Calculate mean and median age of passengers
    :param d: data frame
    :return: rounded mean amd median
    """
    return round(d['Age'].mean(), 2), d['Age'].median()


def get_corr(d):
    """
    Calculate correlation between SibSp and Parch
    :param d: data frame
    :return: rounded correlation
    """
    return round(d.filter(items=['SibSp', 'Parch']).corr(method='pearson'), 2)


def get_most_popular_female_name(d):
    """
    Calculate the most popular female first name
    :param d: data frame
    :return: first name and number of usage
    """

    females = d[d['Sex'] == 'female']['Name']
    names = []
    for elem in females:
        elem = elem.split()
        if 'Mrs.' in elem:
            names += elem[elem.index("Mrs.") + 1:]

        if 'Miss.' in elem:
            names += elem[elem.index("Miss.") + 1:]

    for i in range(len(names)):
        if '(' in names[i]:
            names[i] = names[i].replace('(', '')

        if ')' in names[i]:
            names[i] = names[i].replace(')', '')

    unic_names = set(names)
    names_count = {}
    for un in unic_names:
        count = names.count(un)
        names_count[un] = count
    v = list(names_count.values())
    k = list(names_count.keys())
    return k[v.index(max(v))], max(v)


print(get_most_popular_female_name(data))
