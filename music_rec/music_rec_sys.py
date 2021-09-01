# Music Recommender Systems 1.0

import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from music_rec.___________crawler import path_manage, concat_data


def main():
    print("------- Music Recommender Systems 1.0 -------")
    # input format
    #   target_user_profile: | User_ID | Song_ID |
    #   candidate_user_profile: | User_ID | Song_ID |
    #   song_profile: | Song_ID | Title | Artist | Genre | Release_Date | Plays_count | Likes_count
    function_activate = input(
        "1: User-based Collaborative Filtering, UserCF\n"
        "2: Content-based Filtering, CBF\n"
        "3: Top Popular Play, TopPlay\n"
        "please input command>>")

    # 1: User-based Collaborative Filtering, UserCF
    if function_activate == '1':
        print("{0} procedure is preparing...".format("UserCF: User-based Collaborative Filtering"))
        top_k_sim_neighbor = input(
            "top k most similar users: How many users used to make rating predictions for target user?\n"
            "please input integer k (default suggestion as 30)>>")
        top_n_recommend_item = input(
            "top n most similar songs: How many songs used to recommend for target user?\n"
            "please input integer n (default suggestion as 40)>>")

        # read & concat all candidate_user_profile csv table
        candidate_user_profile = concat_data("ALL_USER_PROFILE", 'candidate_user_profile')

        # read & concat all candidate user song_profile csv table
        candidate_song_profile = concat_data("ALL_SONG_PROFILE", 'song_profile')

        # read all target_user_profile csv table
        for target_user_profile_file in path_manage("", 'target_user_profile', 'r', 0):
            print("{0}:\n{1}".format('target_user_profile', target_user_profile_file))
            user_id = target_user_profile_file.stem.replace("_tar_user_profile", "").split("_", 1)[1]

            # check recommend file exist or not
            check_recommend_exist = list(
                path_manage("{0}_{1}".format(user_id, "UserCF"), 'recommendation_list', 'r', 0))
            if not check_recommend_exist:
                target_user_profile = pd.read_csv(
                    target_user_profile_file, sep=',', encoding='utf8', dtype={'User_ID': str})

                # concat candidate user song_profile & target user add_song_profile csv table
                add_song_profile_file = list(pathlib.Path.cwd().joinpath("Song Information Profile").glob(
                    "*{0}_add_song_profile_full.csv".format(user_id)))[0]
                print("{0}:\n{1}".format(user_id + ' add_song_profile', add_song_profile_file))
                add_song_profile = pd.read_csv(add_song_profile_file, sep=',', encoding='utf8')
                song_profile = pd.concat([candidate_song_profile, add_song_profile], axis=0, join='outer', sort=False)

                UserCF_List = UserCF_RecSys(
                    target_user_profile, candidate_user_profile, song_profile, user_id,
                    top_k_sim_neighbor, top_n_recommend_item)
                UserCF_List.to_csv(path_manage("{0}_{1}".format(user_id, "UserCF"), 'recommendation_list', 'w', 0),
                                   header=True, mode='w', sep=',', index=True, encoding='utf8')

    # 2: Content-based Filtering, CBF
    elif function_activate == '2':
        print("{0} procedure is preparing...".format("CBF: Content-based Filtering"))
        top_n_recommend_item = input(
            "top n most similar songs: How many songs used to recommend for target user?\n"
            "please input integer n (default suggestion as 40)>>")

        # read & concat all candidate user song_profile csv table
        candidate_song_profile = concat_data("ALL_SONG_PROFILE", 'song_profile')

        # read all target_user_profile csv table
        for target_user_profile_file in path_manage("", 'target_user_profile', 'r', 0):
            print("{0}:\n{1}".format('target_user_profile', target_user_profile_file))
            user_id = target_user_profile_file.stem.replace("_tar_user_profile", "").split("_", 1)[1]

            # check recommend file exist or not
            check_recommend_exist = list(
                path_manage("{0}_{1}".format(user_id, "CBF"), 'recommendation_list', 'r', 0))
            if not check_recommend_exist:
                target_user_profile = pd.read_csv(
                    target_user_profile_file, sep=',', encoding='utf8', dtype={'User_ID': str})

                # concat candidate user song_profile & target user add_song_profile csv table
                add_song_profile_file = list(pathlib.Path.cwd().joinpath("Song Information Profile").glob(
                    "*{0}_add_song_profile_full.csv".format(user_id)))[0]
                print("{0}:\n{1}".format(user_id + ' add_song_profile', add_song_profile_file))
                add_song_profile = pd.read_csv(add_song_profile_file, sep=',', encoding='utf8')
                song_profile = pd.concat([candidate_song_profile, add_song_profile], axis=0, join='outer', sort=False)

                CBF_List = CBF_RecSys(target_user_profile, song_profile, user_id, top_n_recommend_item)
                CBF_List.to_csv(path_manage("{0}_{1}".format(user_id, "CBF"), 'recommendation_list', 'w', 0),
                                header=True, mode='w', sep=',', index=True, encoding='utf8')

    # 3: Top Popular Play, TopPlay
    elif function_activate == '3':
        print("{0} procedure is preparing...".format("TopPlay: Top Popular Play"))
        top_n_recommend_item = input(
            "top n most similar songs: How many songs used to recommend for target user?\n"
            "please input integer n (default suggestion as 40)>>")

        # read & concat all candidate user song_profile csv table
        candidate_song_profile = concat_data("TOP_PLAY", 'song_profile')

        # read all target_user_profile csv table
        for target_user_profile_file in path_manage("", 'target_user_profile', 'r', 0):
            print("{0}:\n{1}".format('target_user_profile', target_user_profile_file))
            user_id = target_user_profile_file.stem.replace("_tar_user_profile", "").split("_", 1)[1]

            # check recommend file exist or not
            check_recommend_exist = list(
                path_manage("{0}_{1}".format(user_id, "TopPlay"), 'recommendation_list', 'r', 0))
            if not check_recommend_exist:
                target_user_profile = pd.read_csv(
                    target_user_profile_file, sep=',', encoding='utf8', dtype={'User_ID': str})

                TopPlay_List = TopPlay_RecSys(
                    target_user_profile, candidate_song_profile, user_id, top_n_recommend_item)
                TopPlay_List.to_csv(path_manage("{0}_{1}".format(user_id, "TopPlay"), 'recommendation_list', 'w', 0),
                                    header=True, mode='w', sep=',', index=True, encoding='utf8')


# 1: User-based Collaborative Filtering, UserCF
def UserCF_RecSys(target_user_profile, candidate_user_profile, song_profile, user_id,
                  top_k_sim_neighbor, top_n_recommend_item):
    # concat DataFrame from target_user_profile & candidate_user_profile
    all_user_profile = pd.concat([target_user_profile, candidate_user_profile],
                                 axis=0, join='outer', keys=['target', 'candidate'], sort=False)
    all_user_profile.index.names = ['User_Type', 'index_no']
    all_user_profile.reset_index(level=['User_Type'], inplace=True)

    # import song_profile ['Likes_count'] & ['Plays_count']
    song_profile.set_index(['Song_ID'], inplace=True)
    all_user_profile = all_user_profile.merge(song_profile, left_on='Song_ID', right_on='Song_ID', right_index=True)

    # given Rating = log((['Likes_count'] / max(['Likes_count'])) / (['Plays_count'] / max(['Plays_count'])))
    # if user had observed the song
    all_user_profile['Rating'] = \
        ((all_user_profile['Likes_count'].replace(0, np.nan)).div(all_user_profile['Likes_count'].max())).div(
            (all_user_profile['Plays_count'].replace(0, np.nan)).div(all_user_profile['Plays_count'].max())).fillna(0)
    all_user_profile['Rating'] = all_user_profile['Rating'].apply(lambda x: np.log10(x) if x > 0 else 0)

    # transform to user_song_like_matrix & fill up Nan Rating if the song unknown to user
    user_song_like_matrix = all_user_profile.pivot_table(
        index=['User_ID', 'User_Type'], columns='Song_ID', values='Rating').fillna(value=0)

    # UserSim(target_user, candidate_user)
    # = cosine_similarity(target_user, candidate_user)
    user_sim_list = cosine_similarity(
        user_song_like_matrix.loc[(slice(None), 'target'), :], user_song_like_matrix, dense_output=False)

    # reform MultiIndex
    user_song_like_matrix.columns = pd.MultiIndex.from_product(
        [['Song_ID'], user_song_like_matrix.columns.to_numpy()], names=['Data_Type', 'item'])

    # prediction_rating_target_user
    # = Σ_top_k_sim_neighbor(UserSim(target_user, candidate_users) * rating_candidate_user)
    # / Σ_top_k_sim_neighbor(|UserSim(target_user, candidate_users)|)
    user_song_like_matrix = user_song_like_matrix.mul(user_sim_list[0], axis=0)
    user_song_like_matrix['User_Sim'] = user_sim_list[0]
    user_song_like_matrix.sort_values(by='User_Sim', ascending=False, inplace=True)

    prediction_rating_nume = user_song_like_matrix.loc[(slice(None), 'candidate'), :][0:int(top_k_sim_neighbor)].sum()
    prediction_rating_deno = user_song_like_matrix.loc[(slice(None), 'candidate'), 'User_Sim'][
                             0:int(top_k_sim_neighbor)].sum()
    prediction_rating = prediction_rating_nume / prediction_rating_deno

    # prediction_rating_target_user
    prediction_rating_target_user = pd.DataFrame([prediction_rating.to_numpy()],
                                                 index=user_song_like_matrix.loc[(slice(None), 'target'), :].index,
                                                 columns=user_song_like_matrix.loc[(slice(None), 'target'), :].columns)
    prediction_rating_target_user.rename(index={prediction_rating_target_user.index[0][1]: 'prediction'}, inplace=True)

    # sort by prediction values
    prediction_rating_target_user.sort_values(
        by=prediction_rating_target_user.loc[(slice(None), 'prediction'), :].index[0],
        axis=1, ascending=False, inplace=True)

    # filter out target user unknown rating
    filter_unknown_rating = \
        prediction_rating_target_user.loc[(slice(None), 'prediction'), :]['Song_ID'].columns.difference(
            pd.Index(target_user_profile['Song_ID']), sort=False)

    # recommend top prediction rating item
    recommend_item = \
        prediction_rating_target_user.loc[:, ('Song_ID', filter_unknown_rating)].T[:int(top_n_recommend_item)]
    recommend_item.index.names = ['User_ID', 'Song_ID']
    recommend_item.rename(index={'Song_ID': user_id}, inplace=True)
    recommend_item.columns.names = [None, None]
    recommend_item.columns = ['Prediction_Rating']
    recommend_item['Strategy_Type'] = 'UserCF'
    recommend_item = pd.merge(recommend_item, song_profile, right_index=True, on='Song_ID')
    print(recommend_item)

    return recommend_item


# 2: Content-based Filtering, CBF
def CBF_RecSys(target_user_profile, song_profile, user_id, top_n_recommend_item):
    song_profile.set_index(['Song_ID'], inplace=True)

    # artist keywords authority control
    artist_idx = pd.DataFrame({'keyword': pd.unique(song_profile['Artist'])})
    artist_idx.rename('{0}_Art'.format, inplace=True)
    artist_idx.reset_index(level=0, inplace=True)
    artist_idx.set_index('keyword', drop=True, inplace=True)
    artist_idx.rename(columns={'index': 'Art_Idx'}, inplace=True)

    # genre keywords authority control
    genre_idx = pd.DataFrame({'keyword': pd.unique(song_profile['Genre'])})
    genre_idx.rename('{0}_Gen'.format, inplace=True)
    genre_idx.reset_index(level=0, inplace=True)
    genre_idx.set_index('keyword', drop=True, inplace=True)
    genre_idx.rename(columns={'index': 'Gen_Idx'}, inplace=True)

    # artist & genre keywords authority control merge into song_profile
    song_profile = song_profile.merge(artist_idx, left_on='Artist', right_on='keyword', right_index=True)
    song_profile = song_profile.merge(genre_idx, left_on='Genre', right_on='keyword', right_index=True)
    song_profile['Keywords'] = song_profile.apply(
        lambda x: "{0}, {1}".format(str(x['Art_Idx']), str(x['Gen_Idx'])), axis=1)

    # song information from song_profile merge into target_user_profile
    target_user_song_profile = target_user_profile.merge(
        song_profile, left_on='Song_ID', right_on='Song_ID', right_index=True)

    # compress target_user_song_profile(from 'Title' to 'Keywords') to single row
    # ['User_ID', 'Song_ID',
    # 'Title', 'Artist', 'Genre', 'Release_Date', 'Plays_count', 'Likes_count', 'Art_Idx', 'Gen_Idx', 'Keywords']
    compress_row = []
    for column in list(target_user_song_profile.columns)[2:]:
        field = target_user_song_profile.groupby('User_ID')[column].apply(
            lambda x: ', '.join(map(str, x))).reset_index()
        compress_row.append(field)

    # create single row target_user_song_profile & drop duplicates ['User_ID']
    target_user_song_profile = pd.concat(compress_row, axis=1, sort=False).T.drop_duplicates().T
    target_user_song_profile.set_index('User_ID', inplace=True)

    # concat song_profile & target_user_song_profile
    all_profile = pd.concat([song_profile, target_user_song_profile], keys=['song', 'user'], axis=0, sort=False)
    all_profile.index.names = ['Profile_Type', 'Profile_ID']

    # TF-IDF matrix
    vectorizer = TfidfVectorizer(lowercase=False)
    data_tf_idf = vectorizer.fit_transform(all_profile['Keywords'])
    tf_idf = pd.DataFrame(data_tf_idf.toarray(), columns=vectorizer.get_feature_names(), index=all_profile.index)

    # ProfileSim(target_user_song_profile, candidate_song_profile)
    # = cosine_similarity(target_user_song_profile, candidate_song_profile)
    profile_sim_list = cosine_similarity(
        tf_idf.loc[('user', slice(None)), :], tf_idf, dense_output=False)
    all_profile['Profile_Sim'] = profile_sim_list[0]

    # sort by prediction values
    all_profile.sort_values(by='Profile_Sim', ascending=False, inplace=True)

    # filter out target user unknown rating
    filter_unknown_rating = all_profile.loc['song'].index.difference(
        pd.Index(target_user_profile['Song_ID']), sort=False)

    # recommend top prediction rating item
    recommend_item = all_profile.loc[('song', filter_unknown_rating), :][:int(top_n_recommend_item)]
    recommend_item.index.names = ['User_ID', 'Song_ID']
    recommend_item.rename(index={'song': user_id}, inplace=True)
    recommend_item['Strategy_Type'] = 'CBF'
    print(recommend_item)

    return recommend_item


# 3: Top Popular Play, TopPlay
def TopPlay_RecSys(target_user_profile, candidate_song_profile, user_id, top_n_recommend_item):
    candidate_song_profile = candidate_song_profile.set_index('Song_ID', drop=True)

    # sort candidate_song_profile by ['Plays_count']
    candidate_song_profile.sort_values(by='Plays_count', ascending=False, inplace=True)

    # filter out target user unknown rating
    filter_unknown_rating = candidate_song_profile.index.difference(
        pd.Index(target_user_profile['Song_ID']), sort=False)

    # recommend top Plays_count item
    recommend_item = candidate_song_profile.loc[filter_unknown_rating, :][:int(top_n_recommend_item)]
    recommend_item['User_ID'] = user_id
    recommend_item.set_index(['User_ID', recommend_item.index], inplace=True)
    recommend_item['Strategy_Type'] = 'TopPlay'
    print(recommend_item)

    return recommend_item


def cos_sim(vector1, vector2):
    user_song_like_matrix = pd.DataFrame()
    cos_sim_list = []
    for i in range(0, user_song_like_matrix.shape[0]+1):
        user_sim_h = cos_sim(
            user_song_like_matrix.loc[user_song_like_matrix.index.get_level_values('User_Type') == 'target'].values[0],
            user_song_like_matrix.iloc[[i]].values[0])
        cos_sim_list.append(user_sim_h)
        print(i, user_sim_h)
    print(cos_sim_list)
    # ------- hand made cos_sim -------
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB) ** 0.5)


if __name__ == '__main__':
    main()
