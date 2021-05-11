"""
Part 1: Data Acquistion

Download dataset
"""
# Importing urlib
import os
import urllib.request

# Creating the data folder
if not os.path.exists('./data'):
    os.makedirs('./data')

# Obtaining the dataset using the url that hosts it
kaggle_url = 'https://github.com/sundeepblue/movie_rating_prediction/raw/master/movie_metadata.csv'
if not os.path.exists('./data/kaggle_dataset.csv'):  # avoid downloading if the file exists
    response = urllib.request.urlretrieve(kaggle_url, './data/kaggle_dataset.csv')

import gzip

# Obtaining IMDB's text files
imdb_url_prefix = 'ftp://ftp.funet.fi/pub/mirrors/ftp.imdb.com/pub/frozendata/'
imdb_files_list = ['genres.list.gz', 'ratings.list.gz']
for name in imdb_files_list:
    if not os.path.exists('./data/' + name):
        response = urllib.request.urlretrieve(imdb_url_prefix + name, './data/' + name)
        urllib.request.urlcleanup()  # urllib fails to download two files from a ftp source. This fixes the bug!
        with gzip.open('./data/' + name) as comp_file, open('./data/' + name[:-3], 'w') as reg_file:
            file_content = comp_file.read()
            reg_file.write(file_content)

imdb_url = 'https://anaconda.org/BigGorilla/datasets/1/download/imdb_dataset.csv'
if not os.path.exists('./data/imdb_dataset.csv'):  # avoid downloading if the file exists
    response = urllib.request.urlretrieve(kaggle_url, './data/imdb_dataset.csv')

with open("./data/ratings.list", encoding="ISO-8859-1") as myfile:
    head = [next(myfile) for x in range(38)]
print(''.join(head[28:38]))  # skipping the first 28 lines as they are descriptive headers

with open("./data/genres.list", encoding="ISO-8859-1") as myfile:
    head = [next(myfile) for x in range(392)]
print(''.join(head[382:392]))  # skipping the first 382 lines as they are descriptive header

"""
Part 2: Data Extraction
"""
import re
import pandas as pd

with open("./data/genres.list", encoding="ISO-8859-1") as genres_file:
    raw_content = genres_file.readlines()
    genres_list = []
    content = raw_content[382:]
    for line in content:
        m = re.match(r'"?(.*[^"])"? \(((?:\d|\?){4})(?:/\w*)?\).*\s((?:\w|-)+)', line.strip())
        if m is not None:
            genres_list.append([m.group(1), m.group(2), m.group(3)])
genres_data = pd.DataFrame(genres_list, columns=['movie', 'year', 'genre'])

with open("./data/ratings.list", encoding="ISO-8859-1") as ratings_file:
    raw_content = ratings_file.readlines()
    ratings_list = []
    content = raw_content[28:]
    for line in content:
        m = re.match(r'(?:\d|\.|\*){10}\s+\d+\s+(1?\d\.\d)\s"?(.*[^"])"? \(((?:\d|\?){4})(?:/\w*)?\)', line.strip())
        if m is None: continue
        ratings_list.append([m.group(2), m.group(3), m.group(1)])
    ratings_data = pd.DataFrame(ratings_list, columns=['movie', 'year', 'rating'])

"""
Part 3: Data Profiling & Cleaning
"""

import pandas as pd

"""
Step 1: Loading the “Kaggle 5000 Movie Dataset”
"""
# Loading the Kaggle dataset from the .csv file (kaggle_dataset.csv)
kaggle_data = pd.read_csv('./data/kaggle_dataset.csv')

"""
Step 2: Calculating Some Basic Statistics (Profiling)
"""

print('Number of movies in kaggle_data: {}'.format(kaggle_data.shape[0]))
print('Number of movies in genres_data: {}'.format(genres_data.shape[0]))
print('Number of movies in ratings_data: {}'.format(ratings_data.shape[0]))

print('Number of duplicates in kaggle_data: {}'.format(
    sum(kaggle_data.duplicated(subset=['movie_title', 'title_year'], keep=False))))
print('Number of duplicates in genres_data: {}'.format(
    sum(genres_data.duplicated(subset=['movie', 'year'], keep=False))))
print('Number of duplicates in ratings_data: {}'.format(
    sum(ratings_data.duplicated(subset=['movie', 'year'], keep=False))))

"""
Step 3: Dealing with duplicates (cleaning)
"""

kaggle_data = kaggle_data.drop_duplicates(subset=['movie_title', 'title_year'], keep='first').copy()
genres_data = genres_data.drop_duplicates(subset=['movie', 'year'], keep='first').copy()
ratings_data = ratings_data.drop_duplicates(subset=['movie', 'year'], keep='first').copy()

"""
Step 4: Normalizing the text (cleaning)
"""


def preprocess_title(title):
    title = title.lower()
    title = title.replace(',', ' ')
    title = title.replace("'", '')
    title = title.replace('&', 'and')
    title = title.replace('?', '')
    # title = title.decode('utf-8', 'ignore')
    return title.strip()


kaggle_data['norm_movie_title'] = kaggle_data['movie_title'].map(preprocess_title)
genres_data['norm_movie'] = genres_data['movie'].map(preprocess_title)
ratings_data['norm_movie'] = ratings_data['movie'].map(preprocess_title)

sample = kaggle_data.sample(3, random_state=0)

print("\n\n\n")
print(sample)


def preprocess_year(year):
    if pd.isnull(year):
        return '?'
    else:
        return str(int(year))


kaggle_data['norm_title_year'] = kaggle_data['title_year'].map(preprocess_year)
sample = kaggle_data.head()

print("\n\n\n")
print(sample)

"""
Part 4: Data Matching & Merging
"""

"""
Step 1: Integrating the “IMDB Plain Text Data” files
"""

brief_imdb_data = pd.merge(ratings_data, genres_data, how='inner', on=['norm_movie', 'year'])
sample = brief_imdb_data.head()
print("\n\n\n")
print(sample)

# reading the new IMDB dataset
imdb_data = pd.read_csv('./data/imdb_dataset.csv')
# let's normlize the title as we did in Part 3 of the tutorial
imdb_data['norm_title'] = imdb_data['title'].map(preprocess_title)
imdb_data['norm_year'] = imdb_data['year'].map(preprocess_year)
imdb_data = imdb_data.drop_duplicates(subset=['norm_title', 'norm_year'], keep='first').copy()
shape = imdb_data.shape
print(shape)

"""
Step 2: Integrating the Kaggle and IMDB datasets
"""

data_attempt1 = pd.merge(imdb_data, kaggle_data, how='inner', left_on=['norm_title', 'norm_year'],
                         right_on=['norm_movie_title', 'norm_title_year'])
print(data_attempt1.shape)

import py_stringsimjoin as ssj
import py_stringmatching as sm

imdb_data['id'] = range(imdb_data.shape[0])
kaggle_data['id'] = range(kaggle_data.shape[0])
similar_titles = ssj.edit_distance_join(imdb_data, kaggle_data, 'id', 'id', 'norm_title',
                                        'norm_movie_title', l_out_attrs=['norm_title', 'norm_year'],
                                        r_out_attrs=['norm_movie_title', 'norm_title_year'], threshold=1)
# selecting the entries that have the same production year
data_attempt2 = similar_titles[similar_titles.r_norm_title_year == similar_titles.l_norm_year]
print(data_attempt2.shape)

head = data_attempt2[data_attempt2.l_norm_title != data_attempt2.r_norm_movie_title].head()
print("\n\n\n")
print(head)

"""
Step 3: Using Magellan for Data Matching
"""

# transforming the "budget" column into string and creating a new **mixture** column
ssj.utils.converter.dataframe_column_to_str(imdb_data, 'budget', inplace=True)
imdb_data['mixture'] = imdb_data['norm_title'] + ' ' + imdb_data['norm_year'] + ' ' + imdb_data['budget']

# repeating the same thing for the Kaggle dataset
ssj.utils.converter.dataframe_column_to_str(kaggle_data, 'budget', inplace=True)
kaggle_data['mixture'] = kaggle_data['norm_movie_title'] + ' ' + kaggle_data['norm_title_year'] + \
                         ' ' + kaggle_data['budget']

C = ssj.overlap_coefficient_join(kaggle_data, imdb_data, 'id', 'id', 'mixture', 'mixture', sm.WhitespaceTokenizer(),
                                 l_out_attrs=['norm_movie_title', 'norm_title_year', 'duration',
                                              'budget', 'content_rating'],
                                 r_out_attrs=['norm_title', 'norm_year', 'length', 'budget', 'mpaa'],
                                 threshold=0.65)

shape = C.shape
print(shape)

import py_entitymatching as em

em.set_key(kaggle_data, 'id')  # specifying the key column in the kaggle dataset
em.set_key(imdb_data, 'id')  # specifying the key column in the imdb dataset
em.set_key(C, '_id')  # specifying the key in the candidate set
em.set_ltable(C, kaggle_data)  # specifying the left table
em.set_rtable(C, imdb_data)  # specifying the right table
em.set_fk_rtable(C, 'r_id')  # specifying the column that matches the key in the right table
em.set_fk_ltable(C, 'l_id')  # specifying the column that matches the key in the left table

head = C[['l_norm_movie_title', 'r_norm_title', 'l_norm_title_year', 'r_norm_year',
          'l_budget', 'r_budget', 'l_content_rating', 'r_mpaa']].head()
print("\n\n\n")
print(head)

# Sampling 500 pairs and writing this sample into a .csv file
sampled = C.sample(500, random_state=0)
sampled.to_csv('./data/sampled.csv', encoding='utf-8')

# If you would like to avoid labeling the pairs for now, you can download the labled.csv file from
# BigGorilla using the following command (if you prefer to do it yourself, command the next line)

if not os.path.exists('./data/labeled.csv'):  # avoid downloading if the file exists
    response = urllib.request.urlretrieve('https://anaconda.org/BigGorilla/datasets/1/download/labeled.csv',
                                          './data/labeled.csv')

labeled = em.read_csv_metadata('data/labeled.csv', ltable=kaggle_data, rtable=imdb_data,
                               fk_ltable='l_id', fk_rtable='r_id', key='_id')
head = labeled.head()

print("\n\n\n")
print(head)

split = em.split_train_test(labeled, train_proportion=0.5, random_state=0)
train_data = split['train']
test_data = split['test']

dt = em.DTMatcher(name='DecisionTree', random_state=0)
svm = em.SVMMatcher(name='SVM', random_state=0)
rf = em.RFMatcher(name='RF', random_state=0)
lg = em.LogRegMatcher(name='LogReg', random_state=0)
ln = em.LinRegMatcher(name='LinReg')
nb = em.NBMatcher(name='NaiveBayes')

attr_corres = em.get_attr_corres(kaggle_data, imdb_data)
attr_corres['corres'] = [('norm_movie_title', 'norm_title'),
                         ('norm_title_year', 'norm_year'),
                         ('content_rating', 'mpaa'),
                         ('budget', 'budget'),
                         ]

l_attr_types = em.get_attr_types(kaggle_data)
r_attr_types = em.get_attr_types(imdb_data)

tok = em.get_tokenizers_for_matching()
sim = em.get_sim_funs_for_matching()

F = em.get_features(kaggle_data, imdb_data, l_attr_types, r_attr_types, attr_corres, tok, sim)

train_features = em.extract_feature_vecs(train_data, feature_table=F, attrs_after='label', show_progress=False)
import numpy

train_features = em.impute_table(train_features, exclude_attrs=['_id', 'l_id', 'r_id', 'label'], strategy='mean',
                                 missing_val=numpy.nan)

result = em.select_matcher([dt, rf, svm, ln, lg, nb], table=train_features,
                           exclude_attrs=['_id', 'l_id', 'r_id', 'label'], k=5,
                           target_attr='label', metric_to_select_matcher='f1', random_state=0)
print("\n\n\n")
print(result['cv_stats'])

best_model = result['selected_matcher']
best_model.fit(table=train_features, exclude_attrs=['_id', 'l_id', 'r_id', 'label'], target_attr='label')

test_features = em.extract_feature_vecs(test_data, feature_table=F, attrs_after='label', show_progress=False)
test_features = em.impute_table(test_features, exclude_attrs=['_id', 'l_id', 'r_id', 'label'], strategy='mean',
                                missing_val=numpy.nan)

# Predict on the test data
predictions = best_model.predict(table=test_features, exclude_attrs=['_id', 'l_id', 'r_id', 'label'],
                                 append=True, target_attr='predicted', inplace=False)

# Evaluate the predictions
eval_result = em.eval_matches(predictions, 'label', 'predicted')
em.print_eval_summary(eval_result)

candset_features = em.extract_feature_vecs(C, feature_table=F, show_progress=True)
candset_features = em.impute_table(candset_features, exclude_attrs=['_id', 'l_id', 'r_id'], strategy='mean',
                                   missing_val=numpy.nan)
predictions = best_model.predict(table=candset_features, exclude_attrs=['_id', 'l_id', 'r_id'],
                                 append=True, target_attr='predicted', inplace=False)
matches = predictions[predictions.predicted == 1]

from py_entitymatching.catalog import catalog_manager as cm

matches = matches[['_id', 'l_id', 'r_id', 'predicted']]
matches.reset_index(drop=True, inplace=True)
cm.set_candset_properties(matches, '_id', 'l_id', 'r_id', kaggle_data, imdb_data)
matches = em.add_output_attributes(matches,
                                   l_output_attrs=['norm_movie_title', 'norm_title_year', 'budget', 'content_rating'],
                                   r_output_attrs=['norm_title', 'norm_year', 'budget', 'mpaa'],
                                   l_output_prefix='l_', r_output_prefix='r_',
                                   delete_from_catalog=False)
matches.drop('predicted', axis=1, inplace=True)
sample = matches.head()

print("\n\n\n")
print(sample)
