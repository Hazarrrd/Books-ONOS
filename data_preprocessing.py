import pandas as pd


def preprocessing(file_name):
    columns = ["ISBN", "ID", "TITLE", "COUNTRY", "YEAR", "MONTH", "RATE", "PAGES", "POPULARITY", "SHELVE", "AUTHOR"]
    df = pd.read_csv(file_name, error_bad_lines=False, sep=", ", names=columns)
    df = df[df.ISBN != "None"]

    df.drop('MONTH', axis=1, inplace=True)
    df.drop('SHELVE', axis=1, inplace=True)
    df.drop('ISBN', axis=1, inplace=True)
    df.drop('COUNTRY', axis=1, inplace=True)

    df = df.drop_duplicates(subset="ID", keep="first")
    df = df.drop_duplicates(subset="TITLE", keep="first")
    return df


def shelves_to_dict(file_name):
    dic = {}
    with open(file_name, 'r') as f:
        for line in f:
            id, shelve = line.strip().split(', ')
            if id in dic.keys() and shelve not in dic[id]:
                dic[id].append(shelve)
            else:
                dic[id] = [shelve]
    return dic

def delete_rare_tags(dic, limit):
    all_tags = []
    for key, value in dic.items():
        all_tags += value
    dic_freq_tags = {}
    for i in all_tags:
        if i in dic_freq_tags.keys():
            dic_freq_tags[i] += 1
        else:
            dic_freq_tags[i] = 1

    list_uniqe_tags = []
    for key, value in dic_freq_tags.items():
        if value > limit:
            list_uniqe_tags.append(key)

    for key, value in dic.items():
        dic[key] = list(set(value) & set(list_uniqe_tags))
    return dic

def get_one():
    file_name = 'data/books1000_10000.txt'
    file_name_shelves = 'data/10000-11000.txt'
    return preprocessing(file_name), shelves_to_dict(file_name_shelves)


def get_second():
    file_name = "data/books10000-11000.txt"
    file_name_shelves = 'data/100000-102000.txt'
    return preprocessing(file_name), shelves_to_dict(file_name_shelves)


def get_third():
    file_name = "data/books100000-102000.txt"
    file_name_shelves = 'data/tags1000-10000.txt'
    return preprocessing(file_name), shelves_to_dict(file_name_shelves)


def get_all_data(limit=200):
    df1, shelv1 = get_one()
    df2, shelv2 = get_second()
    df3, shelv3 = get_third()

    frames = [df1, df2, df3]
    df = pd.concat(frames)
    df = df.drop_duplicates(subset="ID", keep="first")
    df = df.drop_duplicates(subset="TITLE", keep="first")
    shelv = {**shelv1, **shelv2, **shelv3}
    shelv = delete_rare_tags(shelv, limit=limit)

    return df, shelv