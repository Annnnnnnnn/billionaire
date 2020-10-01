import pandas as pd

NEWS_PATH = "news.csv"

def convert_level2number(level):
    """
    将level中的评价转换为数字
    标准为:
    super negative        negative        neural          positive        super positive
    -1                    -0.5            0               0.5             1
    """
    number = ""
    for i in level:
        if i == 'super negative':
            number= -1
        elif i == 'negative':
            number= -0.5
        elif i == 'neural':
            number= 0
        elif i == 'positive':
            number= 0.5
        elif i == 'super positive':
            number= 1
        else:
            number = None
    return number

def delete_duplicated(df):
    """
    删除重复的新闻，每一种新闻只保留一个
    尽量保留绝对值大的新闻
    """
    data = {'categories':[],
            'level':[]}
    news_categories = ['business','health','entertainment','science and technology']
    for _, row in df.iterrows():
        if row['category'] in news_categories:
            data['categories'].append(row['category'])
            data['level'].append(row['level'])
            news_categories.remove(row['category'])
        else:
            continue
    dataframe = pd.DataFrame(data)
    return dataframe

def obtain_news(date):
    """
    根据给定的具体日期返回新闻向量
    日期格式为字符串 year/month/day
    """
    news_vector = []
    df = pd.read_csv(NEWS_PATH)
    # 找到该日期对应的所有新闻，每种只保留一个
    news_line = df[df['date'] == date]
    news_line = delete_duplicated(news_line)
    news_vector.append(convert_level2number(news_line[news_line['categories'] == 'business']['level']))
    news_vector.append(convert_level2number(news_line[news_line['categories'] == 'health']['level']))
    news_vector.append(convert_level2number(news_line[news_line['categories'] == 'entertainemnt']['level']))
    news_vector.append(convert_level2number(news_line[news_line['categories'] == 'science and technology']['level']))
    # news_vector = list(map(int, news_vector))
    news_vector = [10 if x=='' else x for x in news_vector]
    return news_vector

def convert_number2onehot(number):
    if number == -1:
        return [1,0,0,0,0]
    elif number == -0.5:
        return [0,1,0,0,0]
    elif number == 0:
        return [0,0,1,0,0]
    elif number == 0.5:
        return [0,0,0,1,0]
    elif number == 1:
        return [0,0,0,0,1]
    elif number == 10:
        return [0,0,0,0,0]

def one_hot_encoding(news_vector):
    """
    将新闻向量进行onehot编码
    """
    onehot_vector = []
    for i in news_vector:
        onehot_vector += convert_number2onehot(i)
    onehot_vector += [0,0,0]
    return onehot_vector

news_vector = obtain_news("2019/4/24")
print(one_hot_encoding(news_vector))