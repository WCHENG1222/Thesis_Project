# User Preference Structure & Recommendation Effectiveness Analysis 3.5

import pathlib
import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
import re
import matplotlib as mpl
import matplotlib.pyplot as plt


'''
# input format
#   candidate_user_profile: | User_ID | Song_ID |
#   song_profile: | Song_ID | Title | Artist | Genre | Release_Date | Plays_count | Likes_count
#   target_user_profile: | User_ID | Song_ID |
#   scale_data: | User_ID | Pref_Ques01:scale_score |...| Pref_Ques20:scale_score | 生理性別 | 年齡 | 教育程度 | 專業領域 |
#   evaluation_data: | User_ID | Rec_Type | Latin_Square_Order
#                    | Song_01_Known:known_or_unknown_statement | Song_01_Satisfy:rating |...
#                    | Song_30_Known:known_or_unknown_statement | Song_30_Satisfy:rating |
#                    | Song_ID | Title | Artist | Genre | Release_Date | Plays_count | Likes_count |
'''


def profile_processing(profile):
    # artist keywords authority control
    artist_idx = pd.DataFrame({'keyword': pd.unique(profile['Artist'])})
    artist_idx.rename('{0}_Art'.format, inplace=True)
    artist_idx.reset_index(level=0, inplace=True)
    artist_idx.set_index('keyword', drop=True, inplace=True)
    artist_idx.rename(columns={'index': 'Art_Idx'}, inplace=True)

    # genre keywords authority control
    genre_idx = pd.DataFrame({'keyword': pd.unique(profile['Genre'])})
    genre_idx.rename('{0}_Gen'.format, inplace=True)
    genre_idx.reset_index(level=0, inplace=True)
    genre_idx.set_index('keyword', drop=True, inplace=True)
    genre_idx.rename(columns={'index': 'Gen_Idx'}, inplace=True)

    # artist & genre keywords authority control merge into song_profile
    profile = profile.merge(artist_idx, left_on='Artist', right_on='keyword')
    profile = profile.merge(genre_idx, left_on='Genre', right_on='keyword')
    profile['Keywords'] = profile.apply(
        lambda x: "{0}, {1}".format(str(x['Art_Idx']), str(x['Gen_Idx'])), axis=1)
    return profile


def user_evaluation(evaluation_data):
    evaluation_data.sort_values(by='User_ID', inplace=True)
    evaluation_data.set_index(['User_ID', 'Rec_Type'], inplace=True)

    # split [Song_ID: known_val] to ['Song_ID', 'Known'] & [Song_ID: satisfy_val] to ['Song_ID', 'Rating']
    known = evaluation_data.filter(regex='Known').stack().str.split(':', expand=True)
    rating = evaluation_data.filter(regex='Satisfy').stack().str.split(':', expand=True).iloc[:, 1].astype(float)

    # drop & reset deprecated index
    known.reset_index(level=2, drop=True, inplace=True)
    rating.reset_index(level=2, drop=True, inplace=True)

    # concat ['Known'] & ['Satisfy']
    evaluation_data = pd.concat([known, rating], axis=1, sort=False)
    evaluation_data.columns = ['Song_ID', 'Known', 'Satisfy']
    evaluation_data['Song_ID'] = evaluation_data['Song_ID'].astype(int)
    return evaluation_data


def profile_description(profile):
    p_song_count = len(pd.unique(profile['Song_ID']))
    p_genre_count = len(pd.unique(profile['Genre']))
    p_artist_count = len(pd.unique(profile['Artist']))
    p_plays_count = profile['Plays_count'].describe().map("{:.2f}".format)
    p_likes_count = profile['Likes_count'].describe().map("{:.2f}".format)
    p_release_date = profile['Release_Date'].describe()
    return print(
        "Song Count: {0}\nGenre Count: {1}\nArtist Count: {2}\n"
        "Plays Count:\n{3}\nLikes Count:\n{4}\nRelease Date:\n{5}\n".format(
            p_song_count, p_genre_count, p_artist_count, p_plays_count, p_likes_count, p_release_date
        ))


def preference_structure_scale(scale_data):
    scale_data.set_index('User_ID', inplace=True)
    scale_list = []
    # split [Pref_ID: pref_val] to ['Pref_ID', 'Pref_Val'] & transform to {'User_ID': 'Pref_Val'}
    for user_scale in scale_data.filter(regex='Pref').iterrows():
        scale_values = user_scale[1].sort_values().str.split(':', expand=True)
        scale_values = pd.DataFrame(
            scale_values[1].astype(float).to_numpy(), index=scale_values[0].to_numpy(), columns=[user_scale[0]]).T
        scale_list.append(scale_values)
    scale_data = pd.concat(scale_list, sort=False)
    return scale_data


def unknown_filter(input_data, mask, original_col, revised_col, revised_value):
    input_data[revised_col] = input_data[original_col]
    input_data.loc[~mask, revised_col] = revised_value
    return


def metrics_calc(metric_name, input_data, key_col, data_col, metric_func):
    data = input_data.groupby(key_col)[data_col].apply(metric_func)
    data.rename(metric_name, inplace=True)
    return data


def metrics_df(metrics_param, metrics_index):
    data_df = []
    for metric, param in metrics_param.items():
        data = metrics_calc(metric, param[0], param[1], param[2], param[3])
        if data.shape[0] != metrics_index.groupby(param[1]).count().shape[0]:
            data = data.reindex(metrics_index.groupby(param[1]).count().index, fill_value=0)
        else:
            pass
        data_df.append(data)
    data_df = pd.concat(data_df, axis=1)
    return data_df


def metrics_interact(input_data, variable_matrix, metrics_col_list, judgement_metric):
    if judgement_metric == 'Satisfaction_RE':
        judgement_col = 'Satisfy'
    elif judgement_metric == 'Satisfaction_Unknown_RE':
        judgement_col = 'Unknown_Satisfy_all'
    elif judgement_metric == 'Serendipity_RE':
        judgement_col = 'Unknown_Satisfy_3'
    else:
        judgement_col = judgement_metric
    judgement_sum = metrics_calc(
        '{0}_sum'.format(judgement_col), input_data, ['User_ID', 'Rec_Type'],
        judgement_col, lambda rec: rec.sum())
    for metric in metrics_col_list:
        interact_col = '{0}_X_{1}'.format(metric, judgement_metric)
        variable_matrix[interact_col] = variable_matrix[metric] * judgement_sum
    return


def entropy_based_diversity(profile):
    label_list = pd.concat([profile['Gen_Idx'], profile['Art_Idx']], axis=0, ignore_index=True, sort=True)
    value, counts = np.unique(list(label_list), return_counts=True)
    return entropy(counts, base=10)


def similarity_based_diversity(profile, calc_func):
    # Keywords Count matrix
    vectorizer = CountVectorizer(lowercase=False)
    text_matrix = vectorizer.fit_transform(profile['Keywords'])
    text_matrix = pd.DataFrame(text_matrix.toarray(), columns=vectorizer.get_feature_names(), index=profile.index)
    profile_sim_list = cosine_distances(text_matrix)
    return calc_func(profile_sim_list)


def minmax_popularity_based_novelty(profile, min_pop, max_pop, calc_func):
    return profile.apply(lambda x: (max_pop - x) / (max_pop - min_pop)).agg(calc_func)


def log_popularity_based_novelty(profile, calc_func):
    return profile.apply(lambda x: -np.log10(x)).agg(calc_func)


def minmax_time_aware_novelty(profile, first_release, latest_release, calc_func):
    return profile.apply(lambda x: (x - first_release) / (latest_release - first_release)).agg(calc_func)


def log_time_aware_novelty(profile, first_release, calc_func):
    return profile.apply(lambda x: np.log10((x - first_release) / np.timedelta64(1, 'D'))).agg(calc_func)


def distance_based_novelty(profile, user_profile, calc_func):
    # get target user's User Profile
    user_profile = user_profile[user_profile['User_ID'] == pd.unique(profile['User_ID'])[0]]
    # unify Keywords index from User Profile & Recommendation Profile
    all_profile = pd.concat([profile, user_profile], axis=0, ignore_index=True, sort=False)
    all_profile.drop(columns=['Art_Idx', 'Gen_Idx', 'Keywords'], inplace=True)
    all_profile = profile_processing(all_profile)
    # distinguish between User Profile & Recommendation Profile
    all_profile.rename(columns={'Rec_Type': 'Profile_Type'}, inplace=True)
    all_profile.sort_values(by='Profile_Type', inplace=True)
    all_profile['Profile_Type'].fillna(value='UserProfile', inplace=True)
    all_profile.set_index(['Profile_Type', 'Song_ID'], inplace=True)
    # Keywords Count matrix
    vectorizer = CountVectorizer(lowercase=False)
    text_matrix = vectorizer.fit_transform(all_profile['Keywords'])
    text_matrix = pd.DataFrame(text_matrix.toarray(), columns=vectorizer.get_feature_names(), index=all_profile.index)
    profile_sim_list = cosine_distances(
        text_matrix.loc[text_matrix.index.get_level_values(0) == 'UserProfile', :],
        text_matrix.loc[text_matrix.index.get_level_values(0) != 'UserProfile', :]
    )
    return calc_func(profile_sim_list)


def long_tail_plots(candidate_likes):
    candidate_count = pd.DataFrame(candidate_likes.value_counts().to_numpy(), columns=['Likes'])
    candidate_count['Size'] = pd.cut(candidate_count['Likes'].to_numpy(), bins=10, labels=range(1, 11)).codes
    fig_candidate_count, axs_candidate_count = plt.subplots(dpi=200, constrained_layout=True)
    axs_candidate_count.scatter(
        candidate_count.index, candidate_count['Likes'].to_numpy(),
        c=candidate_count['Size'].to_numpy(), s=np.sqrt(candidate_count['Likes'].to_numpy())*3)
    return


def main():
    # ----------------------------------------------------------------------
    # Initialization

    print("------- User Preference Structure & Recommendation Effectiveness Analysis 3.0 -------")
    # read 2000 Candidate User Profile
    candidate_user_profile = pd.read_csv(
        pathlib.Path.cwd().joinpath("Candidate User Profile", "ALL_USER_PROFILE_can_user_profile.csv"),
        sep=',', encoding='utf8')

    print("------- Candidate Profile Count -------")
    print("Candidate Profile include:\n  {0} users & {1} songs\n  every profile has {2} songs at least".format(
        len(pd.unique(candidate_user_profile['User_ID'])), len(pd.unique(candidate_user_profile['Song_ID'])),
        pd.unique(candidate_user_profile['Count']).min()))

    # read User List for mapping
    user_list = pd.read_csv(
        pathlib.Path.cwd().joinpath("User Evaluation", "user_list.csv"), sep=',', encoding='utf8', dtype=str)

    # read Song Evaluation Form Data
    evaluation_data = pd.read_csv(
        pathlib.Path.cwd().joinpath("User Evaluation", "SongEvalForm_Value.csv"), sep=',', encoding='utf8', dtype=str)

    # read Preference Structure Scale Data
    scale_data = pd.read_csv(
        pathlib.Path.cwd().joinpath("User Evaluation", "PrefStruScale_Value.csv"), sep=',', encoding='utf8', dtype=str)

    # read 126 Target User Profile
    print("------- 126 Target User Profile description -------")
    target_user_profile = pd.read_csv(
        pathlib.Path.cwd().joinpath("User Evaluation", "TARGET_USER_user_profile.csv"),
        sep=',', encoding='utf8', dtype={'User_ID': str})
    target_user_profile = profile_processing(target_user_profile)
    target_user_profile['Release_Date'] = pd.to_datetime(target_user_profile['Release_Date'],
                                                         infer_datetime_format=True)
    profile_description(target_user_profile)

    # read 15333 Candidate Songs Profile
    print("------- 15333 Candidate Songs Profile description -------")
    candidate_song_profile = pd.read_csv(
        pathlib.Path.cwd().joinpath("Song Information Profile", "ALL_SONG_PROFILE_song_profile_full.csv"),
        sep=',', encoding='utf8')
    candidate_song_profile['Release_Date'] = pd.to_datetime(
        candidate_song_profile['Release_Date'], infer_datetime_format=True)
    profile_description(candidate_song_profile)

    # read all Recommendation List
    recommendation_list = pd.read_csv(
        pathlib.Path.cwd().joinpath("User Evaluation", "ALL_USER_rec_list.csv"),
        sep=',', encoding='utf8', dtype={'User_ID': str})
    recommendation_list.drop(columns=['Art_Idx', 'Gen_Idx', 'Keywords'], inplace=True)
    recommendation_list = profile_processing(recommendation_list)
    recommendation_list['Release_Date'] = pd.to_datetime(recommendation_list['Release_Date'],
                                                         infer_datetime_format=True)

    # Recommendation Evaluation Profile processing
    recommendation_evaluation = user_evaluation(evaluation_data)
    recommendation_evaluation_profile = pd.merge(
        recommendation_evaluation.reset_index(), recommendation_list,
        left_on=['User_ID', 'Rec_Type', 'Song_ID'], right_on=['User_ID', 'Strategy_Type', 'Song_ID'])

    # concat global song profile (Target User + Recommendation Candidate)
    print("------- global song profile description -------")
    global_profile = pd.concat([target_user_profile, candidate_song_profile], axis=0, sort=False)
    # get global Plays_count (drop_duplicates by chosen largest Plays_count)
    global_profile['Plays_count'] = global_profile['Plays_count'].replace(0, 1)  # avoid 0 Plays_count
    global_plays = global_profile.groupby('Song_ID')['Plays_count'].max()
    global_min_plays = global_plays.min()
    global_max_plays = global_plays.max()
    log_global_plays = -np.log10(global_plays)
    print("global Plays_count:\n{0}\nglobal log(Plays_count):\n{1}".format(
        global_plays.describe().map("{:.2f}".format), log_global_plays.describe().map("{:.5f}".format)))
    # get global Likes_count (drop_duplicates by chosen largest Likes_count)
    global_likes = global_profile.groupby('Song_ID')['Likes_count'].max()
    global_min_likes = global_likes.min()
    global_max_likes = global_likes.max()
    log_global_likes = -np.log10(global_likes)
    print("global Likes_count:\n{0}\nglobal log(Likes_count):\n{1}".format(
        global_likes.describe().map("{:.2f}".format), log_global_likes.describe().map("{:.5f}".format)))
    # get global Release_Date
    global_release = global_profile.groupby('Song_ID')['Release_Date'].max()
    global_first_release = global_profile['Release_Date'].min() - np.timedelta64(1, 'D')  # avoid 0 Release_Date_period
    global_latest_release = global_profile['Release_Date'].max()
    global_release_period = (global_release - global_first_release) / np.timedelta64(1, 'D')  # type conversion to int
    log_global_release_period = np.log10(global_release_period)
    print("global Release_Date:\n{0}\nRelease_period:\n{1}\nglobal log(Release_period):\n{2}".format(
        global_release.describe(),
        global_release_period.describe().map("{:.2f}".format),
        log_global_release_period.describe().map("{:.5f}".format)))

    # ----------------------------------------------------------------------
    # Preference Structure (Moderator Variables)

    # Preference Scale: Diversity; Openness; Identity; Involvement
    scale_up = preference_structure_scale(scale_data)
    scale_up['rev_Invol_05'] = 6 - scale_up['Invol_05']
    scale_up['rev_Invol_06'] = 6 - scale_up['Invol_06']
    scale_up['Diversity_Scale_UP'] = (scale_up['Dive_04'] + scale_up['Dive_03']) / 2
    scale_up['Openness_Scale_UP'] = (scale_up['Open_05'] + scale_up['Open_06'] + scale_up['Open_03']) / 3
    scale_up['Identity_Scale_UP'] = (scale_up['Iden_04'] + scale_up['Iden_03']) / 2
    scale_up['Involvement_Scale_UP'] = (scale_up['rev_Invol_06'] + scale_up['Invol_04'] + scale_up['Invol_02']) / 3
    scale_up = scale_up

    # Demographic Statistics data
    print("------- Demographic Statistics -------")
    demog = scale_data.filter(regex='生理性別|年齡|教育程度|專業領域')
    print(pd.value_counts(demog['生理性別']))
    print(pd.value_counts(demog['年齡']))
    print(pd.value_counts(demog['教育程度']))
    print(pd.value_counts(demog['專業領域']))

    # User Profile Metrics
    # 1. Diversity
    up_diversity_param = {
        # entropy_based
        'Div_Entropy_UP': [
            target_user_profile, 'User_ID', target_user_profile.columns, entropy_based_diversity],
        # similarity_based
        'Div_AvgSim_UP': [
            target_user_profile, 'User_ID', target_user_profile.columns,
            lambda up: similarity_based_diversity(
                up, lambda song_sim: song_sim.sum() / (song_sim.shape[0] * (song_sim.shape[1] - 1)))],
        'Div_SumSim_UP': [
            target_user_profile, 'User_ID', target_user_profile.columns,
            lambda up: similarity_based_diversity(
                up, lambda song_sim: np.tril(song_sim).sum())],
        # genre_count
        'Div_GenreCount_UP': [
            target_user_profile, 'User_ID', 'Gen_Idx', lambda genre_keyword: genre_keyword.nunique()],
        # artist_count
        'Div_ArtistCount_UP': [
            target_user_profile, 'User_ID', 'Art_Idx', lambda artist_keyword: artist_keyword.nunique()],
    }
    # 2. Openness
    up_openness_param = {
        # min_max_normalization_plays_popularity_based
        'Opn_MMAvgPopPlays_UP': [
            target_user_profile, 'User_ID', 'Plays_count',
            lambda up: minmax_popularity_based_novelty(up, global_min_plays, global_max_plays, 'mean')],
        'Opn_MMSumPopPlays_UP': [
            target_user_profile, 'User_ID', 'Plays_count',
            lambda up: minmax_popularity_based_novelty(up, global_min_plays, global_max_plays, 'sum')],
        'Opn_MMMedPopPlays_UP': [
            target_user_profile, 'User_ID', 'Plays_count',
            lambda up: minmax_popularity_based_novelty(up, global_min_plays, global_max_plays, 'median')],
        # log_plays_popularity_based
        'Opn_LogAvgPopPlays_UP': [
            target_user_profile, 'User_ID', 'Plays_count', lambda up: log_popularity_based_novelty(up, 'mean')],
        'Opn_LogSumPopPlays_UP': [
            target_user_profile, 'User_ID', 'Plays_count', lambda up: log_popularity_based_novelty(up, 'sum')],
        'Opn_LogMedPopPlays_UP': [
            target_user_profile, 'User_ID', 'Plays_count', lambda up: log_popularity_based_novelty(up, 'median')],
        # min_max_normalization_likes_popularity_based
        'Opn_MMAvgPopLikes_UP': [
            target_user_profile, 'User_ID', 'Likes_count',
            lambda up: minmax_popularity_based_novelty(up, global_min_likes, global_max_likes, 'mean')],
        'Opn_MMSumPopLikes_UP': [
            target_user_profile, 'User_ID', 'Likes_count',
            lambda up: minmax_popularity_based_novelty(up, global_min_likes, global_max_likes, 'sum')],
        'Opn_MMMedPopLikes_UP': [
            target_user_profile, 'User_ID', 'Likes_count',
            lambda up: minmax_popularity_based_novelty(up, global_min_likes, global_max_likes, 'median')],
        # log_likes_popularity_based
        'Opn_LogAvgPopLikes_UP': [
            target_user_profile, 'User_ID', 'Likes_count', lambda up: log_popularity_based_novelty(up, 'mean')],
        'Opn_LogSumPopLikes_UP': [
            target_user_profile, 'User_ID', 'Likes_count', lambda up: log_popularity_based_novelty(up, 'sum')],
        'Opn_LogMedPopLikes_UP': [
            target_user_profile, 'User_ID', 'Likes_count', lambda up: log_popularity_based_novelty(up, 'median')],
        # min_max_normalization_time_aware
        'Opn_MMAvgTime_UP': [
            target_user_profile, 'User_ID', 'Release_Date',
            lambda up: minmax_time_aware_novelty(up, global_first_release, global_latest_release, 'mean')],
        'Opn_MMSumTime_UP': [
            target_user_profile, 'User_ID', 'Release_Date',
            lambda up: minmax_time_aware_novelty(up, global_first_release, global_latest_release, 'sum')],
        'Opn_MMMedTime_UP': [
            target_user_profile, 'User_ID', 'Release_Date',
            lambda up: minmax_time_aware_novelty(up, global_first_release, global_latest_release, 'median')],
        # log_time_aware
        'Opn_LogAvgTime_UP': [
            target_user_profile, 'User_ID', 'Release_Date',
            lambda up: log_time_aware_novelty(up, global_first_release, 'mean')],
        'Opn_LogSumTime_UP': [
            target_user_profile, 'User_ID', 'Release_Date',
            lambda up: log_time_aware_novelty(up, global_first_release, 'sum')],
        'Opn_LogMedTime_UP': [
            target_user_profile, 'User_ID', 'Release_Date',
            lambda up: log_time_aware_novelty(up, global_first_release, 'median')],
        # count_user_profile
        'Opn_UserProfCount_UP': [
            target_user_profile, 'User_ID', 'Song_ID', lambda like_song: like_song.nunique()]
    }
    # metrics output
    up_diversity_df = metrics_df(up_diversity_param, target_user_profile)
    up_openness_df = metrics_df(up_openness_param, target_user_profile)

    # Moderator Variables
    mo_matrix = pd.concat([scale_up, up_diversity_df, up_openness_df, demog], axis=1, sort=False)
    mo_matrix = mo_matrix.round(5)
    # user data De-identification & Anonymization
    mo_matrix.index.names = ['User_ID']
    mo_matrix = pd.merge(mo_matrix, user_list, on=['User_ID'], right_index=True, sort=False)
    mo_matrix.set_index('User_SN', inplace=True)
    print("------- User Preference Structure (Moderator Variables) -------")
    mo_matrix.to_csv(pathlib.Path.cwd() / "ModeratorVar.csv", sep='\t', encoding='utf8')
    print(mo_matrix)

    # ----------------------------------------------------------------------
    # Recommendation Effectiveness (Dependent Variables)

    # Coverage
    recommendation_total_count = len(pd.unique(recommendation_evaluation_profile['Song_ID']))
    recommendation_each_count = recommendation_evaluation_profile.groupby(['Rec_Type'])['Song_ID'].apply(
        lambda x: len(pd.unique(x)))
    print("------- Recommendation Coverage -------")
    print(
        "All Strategies totally recommend: {0} songs\n"
        "  UserCF recommend: {1} songs\n"
        "  CBF recommend: {2} songs\n"
        "  TopPlay recommend: {3} songs".format(
            recommendation_total_count, recommendation_each_count['UserCF'],
            recommendation_each_count['CBF'], recommendation_each_count['TopPlay']))

    # User Judgement Metrics
    # Serendipity masks & filters
    # unknown & satisfaction rated all
    unknown_satisfy_all_mask = \
        (recommendation_evaluation_profile['Known'] == 'unknown') & (recommendation_evaluation_profile['Satisfy'] > -1)
    unknown_filter(recommendation_evaluation_profile, unknown_satisfy_all_mask, 'Satisfy', 'Unknown_Satisfy_all', 0)
    # unknown & satisfaction rated 3~4
    unknown_satisfy_3_mask = \
        (recommendation_evaluation_profile['Known'] == 'unknown') & (recommendation_evaluation_profile['Satisfy'] > 2)
    unknown_filter(recommendation_evaluation_profile, unknown_satisfy_3_mask, 'Satisfy', 'Unknown_Satisfy_3', 0)
    # strict version: unknown & satisfaction rated 4
    unknown_satisfy_4_mask = \
        (recommendation_evaluation_profile['Known'] == 'unknown') & (recommendation_evaluation_profile['Satisfy'] > 3)
    unknown_filter(recommendation_evaluation_profile, unknown_satisfy_4_mask, 'Satisfy', 'Unknown_Satisfy_4', 0)
    re_judgement_param = {
        # 1. Unknown
        # count
        'Unknown_Count_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Known',
            lambda rec: rec.value_counts(dropna=False)['unknown']
            if rec.value_counts(dropna=False).index.isin(['unknown']).any() else 0],
        # ratio
        'Unknown_Ratio_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Known',
            lambda rec: rec.value_counts(dropna=False)['unknown'] / rec.value_counts(dropna=False).sum()
            if rec.value_counts(dropna=False).index.isin(['unknown']).any() else 0],
        # 2. Satisfaction
        'Satisfaction_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Satisfy',
            lambda rec: rec.mean()],
        # 3. Serendipity
        # unknown & satisfaction rated 3~4
        'Serendipity_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Unknown_Satisfy_3',
            lambda rec: rec.mean()],
        # unknown & satisfaction rated 4
        'strict_Serendipity_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Unknown_Satisfy_4',
            lambda rec: rec.mean()],
    }

    # Recommendation Evaluation Metrics
    # 1. Diversity
    re_diversity_param = {
        # entropy_based
        'Div_Entropy_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], recommendation_evaluation_profile.columns,
            entropy_based_diversity],
        # similarity_based
        'Div_AvgSim_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], recommendation_evaluation_profile.columns,
            lambda rec: similarity_based_diversity(
                rec, lambda song_sim: song_sim.sum() / (song_sim.shape[0] * (song_sim.shape[1] - 1)))],
        'Div_SumSim_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], recommendation_evaluation_profile.columns,
            lambda rec: similarity_based_diversity(
                rec, lambda song_sim: np.tril(song_sim).sum())]
    }
    # 2. Novelty(unknown)
    # Novelty(unknown) masks & filters
    unknown_mask = recommendation_evaluation_profile['Known'] == 'unknown'
    # plays_popularity_based(unknown)
    unknown_filter(recommendation_evaluation_profile, unknown_mask,
                   'Plays_count', 'Unknown_Plays_count', global_max_plays)
    # likes_popularity_based(unknown)
    unknown_filter(recommendation_evaluation_profile, unknown_mask,
                   'Likes_count', 'Unknown_Likes_count', global_max_likes)
    # time_aware(unknown)
    unknown_filter(recommendation_evaluation_profile, unknown_mask,
                   'Release_Date', 'Unknown_Release_Date', (global_first_release + np.timedelta64(1, 'D')))
    re_novelty_unknown_param = {
        # min_max_normalization_plays_popularity_based(unknown)
        'Nov_MMAvgPopPlays_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Unknown_Plays_count',
            lambda rec: minmax_popularity_based_novelty(rec, global_min_plays, global_max_plays, 'mean')],
        'Nov_MMSumPopPlays_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Unknown_Plays_count',
            lambda rec: minmax_popularity_based_novelty(rec, global_min_plays, global_max_plays, 'sum')],
        'Nov_MMMedPopPlays_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Unknown_Plays_count',
            lambda rec: minmax_popularity_based_novelty(rec, global_min_plays, global_max_plays, 'median')],
        # log_plays_popularity_based(unknown)
        'Nov_LogAvgPopPlays_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Unknown_Plays_count',
            lambda rec: log_popularity_based_novelty(rec, 'mean')],
        'Nov_LogSumPopPlays_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Unknown_Plays_count',
            lambda rec: log_popularity_based_novelty(rec, 'sum')],
        'Nov_LogMedPopPlays_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Unknown_Plays_count',
            lambda rec: log_popularity_based_novelty(rec, 'median')],
        # min_max_normalization_likes_popularity_based(unknown)
        'Nov_MMAvgPopLikes_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Unknown_Likes_count',
            lambda rec: minmax_popularity_based_novelty(rec, global_min_likes, global_max_likes, 'mean')],
        'Nov_MMSumPopLikes_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Unknown_Likes_count',
            lambda rec: minmax_popularity_based_novelty(rec, global_min_likes, global_max_likes, 'sum')],
        'Nov_MMMedPopLikes_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Unknown_Likes_count',
            lambda rec: minmax_popularity_based_novelty(rec, global_min_likes, global_max_likes, 'median')],
        # log_likes_popularity_based(unknown)
        'Nov_LogAvgPopLikes_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Unknown_Likes_count',
            lambda rec: log_popularity_based_novelty(rec, 'mean')],
        'Nov_LogSumPopLikes_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Unknown_Likes_count',
            lambda rec: log_popularity_based_novelty(rec, 'sum')],
        'Nov_LogMedPopLikes_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Unknown_Likes_count',
            lambda rec: log_popularity_based_novelty(rec, 'median')],
        # min_max_normalization_time_aware(unknown)
        'Nov_MMAvgTime_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Unknown_Release_Date',
            lambda rec: minmax_time_aware_novelty(rec, global_first_release, global_latest_release, 'mean')],
        'Nov_MMSumTime_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Unknown_Release_Date',
            lambda rec: minmax_time_aware_novelty(rec, global_first_release, global_latest_release, 'sum')],
        'Nov_MMMedTime_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Unknown_Release_Date',
            lambda rec: minmax_time_aware_novelty(rec, global_first_release, global_latest_release, 'median')],
        # log_time_aware(unknown)
        'Nov_LogAvgTime_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Unknown_Release_Date',
            lambda rec: log_time_aware_novelty(rec, global_first_release, 'mean')],
        'Nov_LogSumTime_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Unknown_Release_Date',
            lambda rec: log_time_aware_novelty(rec, global_first_release, 'sum')],
        'Nov_LogMedTime_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Unknown_Release_Date',
            lambda rec: log_time_aware_novelty(rec, global_first_release, 'median')],
        # distance_based(unknown)
        'Nov_AvgDist_RE': [
            recommendation_evaluation_profile[unknown_mask], ['User_ID', 'Rec_Type'],
            recommendation_evaluation_profile.columns,
            lambda rec: distance_based_novelty(
                rec, target_user_profile, lambda song_sim: song_sim.sum() / (song_sim.shape[0] * 30))],
        'Nov_SumDist_RE': [
            recommendation_evaluation_profile[unknown_mask], ['User_ID', 'Rec_Type'],
            recommendation_evaluation_profile.columns,
            lambda rec: distance_based_novelty(
                rec, target_user_profile, lambda song_sim: song_sim.sum())]
    }
    # 3. Novelty(all)
    re_novelty_all_param = {
        # min_max_normalization_plays_popularity_based(all)
        'NovAll_MMAvgPopPlays_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Plays_count',
            lambda rec: minmax_popularity_based_novelty(rec, global_min_plays, global_max_plays, 'mean')],
        'NovAll_MMSumPopPlays_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Plays_count',
            lambda rec: minmax_popularity_based_novelty(rec, global_min_plays, global_max_plays, 'sum')],
        'NovAll_MMMedPopPlays_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Plays_count',
            lambda rec: minmax_popularity_based_novelty(rec, global_min_plays, global_max_plays, 'median')],
        # log_plays_popularity_based(all)
        'NovAll_LogAvgPopPlays_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Plays_count',
            lambda rec: log_popularity_based_novelty(rec, 'mean')],
        'NovAll_LogSumPopPlays_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Plays_count',
            lambda rec: log_popularity_based_novelty(rec, 'sum')],
        'NovAll_LogMedPopPlays_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Plays_count',
            lambda rec: log_popularity_based_novelty(rec, 'median')],
        # min_max_normalization_likes_popularity_based(all)
        'NovAll_MMAvgPopLikes_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Likes_count',
            lambda rec: minmax_popularity_based_novelty(rec, global_min_likes, global_max_likes, 'mean')],
        'NovAll_MMSumPopLikes_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Likes_count',
            lambda rec: minmax_popularity_based_novelty(rec, global_min_likes, global_max_likes, 'sum')],
        'NovAll_MMMedPopLikes_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Likes_count',
            lambda rec: minmax_popularity_based_novelty(rec, global_min_likes, global_max_likes, 'median')],
        # log_likes_popularity_based(all)
        'NovAll_LogAvgPopLikes_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Likes_count',
            lambda rec: log_popularity_based_novelty(rec, 'mean')],
        'NovAll_LogSumPopLikes_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Likes_count',
            lambda rec: log_popularity_based_novelty(rec, 'sum')],
        'NovAll_LogMedPopLikes_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Likes_count',
            lambda rec: log_popularity_based_novelty(rec, 'median')],
        # min_max_normalization_time_aware(all)
        'NovAll_MMAvgTime_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Release_Date',
            lambda rec: minmax_time_aware_novelty(rec, global_first_release, global_latest_release, 'mean')],
        'NovAll_MMSumTime_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Release_Date',
            lambda rec: minmax_time_aware_novelty(rec, global_first_release, global_latest_release, 'sum')],
        'NovAll_MMMedTime_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Release_Date',
            lambda rec: minmax_time_aware_novelty(rec, global_first_release, global_latest_release, 'median')],
        # log_time_aware(all)
        'NovAll_LogAvgTime_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Release_Date',
            lambda rec: log_time_aware_novelty(rec, global_first_release, 'mean')],
        'NovAll_LogSumTime_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Release_Date',
            lambda rec: log_time_aware_novelty(rec, global_first_release, 'sum')],
        'NovAll_LogMedTime_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], 'Release_Date',
            lambda rec: log_time_aware_novelty(rec, global_first_release, 'median')],
        # distance_based(all)
        'NovAll_AvgDist_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], recommendation_evaluation_profile.columns,
            lambda rec: distance_based_novelty(
                rec, target_user_profile, lambda song_sim: song_sim.sum() / (song_sim.shape[0] * 30))],
        'NovAll_SumDist_RE': [
            recommendation_evaluation_profile, ['User_ID', 'Rec_Type'], recommendation_evaluation_profile.columns,
            lambda rec: distance_based_novelty(
                rec, target_user_profile, lambda song_sim: song_sim.sum())]
    }
    # metrics output
    re_judgement = metrics_df(re_judgement_param, recommendation_evaluation_profile)
    re_diversity = metrics_df(re_diversity_param, recommendation_evaluation_profile)
    re_novelty_unknown = metrics_df(re_novelty_unknown_param, recommendation_evaluation_profile)
    re_novelty_all = metrics_df(re_novelty_all_param, recommendation_evaluation_profile)

    # Dependent Variables
    dv_matrix = pd.concat([re_judgement, re_diversity, re_novelty_unknown, re_novelty_all], axis=1, sort=False)
    # metrics interaction
    novelty_unknown_list = [
        'Nov_MMAvgPopPlays_RE', 'Nov_LogAvgPopPlays_RE', 'Nov_MMAvgPopLikes_RE', 'Nov_LogAvgPopLikes_RE',
        'Nov_MMAvgTime_RE', 'Nov_LogAvgTime_RE', 'Nov_AvgDist_RE'
    ]
    novelty_all_list = [
        'NovAll_MMAvgPopPlays_RE', 'NovAll_LogAvgPopPlays_RE', 'NovAll_MMAvgPopLikes_RE', 'NovAll_LogAvgPopLikes_RE',
        'NovAll_MMAvgTime_RE', 'NovAll_LogAvgTime_RE', 'NovAll_AvgDist_RE'
    ]
    # Novelty(unknown) X Satisfaction_RE
    metrics_interact(recommendation_evaluation_profile, dv_matrix, novelty_unknown_list, 'Satisfaction_Unknown_RE')
    # Novelty(unknown) X Serendipity_RE
    metrics_interact(recommendation_evaluation_profile, dv_matrix, novelty_unknown_list, 'Serendipity_RE')
    # Novelty(all) X Satisfaction_RE
    metrics_interact(recommendation_evaluation_profile, dv_matrix, novelty_all_list, 'Satisfaction_RE')
    dv_matrix = dv_matrix.round(5)
    # user data De-identification & Anonymization
    dv_matrix = pd.merge(dv_matrix, user_list, on=['User_ID'], right_index=True, sort=False)
    dv_matrix.reset_index(level=0, drop=True, inplace=True)
    dv_matrix.set_index(['User_SN', dv_matrix.index], inplace=True)
    # get Dummy Variables
    dv_matrix['Rec_Type_SN'] = dv_matrix.rename(
        level='Rec_Type', index={'UserCF': 0, 'CBF': 1, 'TopPlay': 2}).index.get_level_values('Rec_Type')
    rec_type_dummy = \
        pd.get_dummies(dv_matrix.index.get_level_values('Rec_Type')).add_prefix('dummy_').set_index(dv_matrix.index)
    dv_matrix = dv_matrix.join(rec_type_dummy, sort=False)
    # get Latin Square Order
    user_a_pattern = r'[user]{4}[a]{1}\d{3}'
    user_b_pattern = r'[user]{4}[b]{1}\d{3}'
    user_c_pattern = r'[user]{4}[c]{1}\d{3}'
    strategy_order = ['UserCF', 'CBF', 'TopPlay']
    latin_square = dv_matrix.reset_index()['User_SN'].drop_duplicates().to_frame()
    latin_square['strategy_order'] = None
    for index, row in latin_square.iterrows():
        if re.search(user_a_pattern, row['User_SN']):
            latin_square.at[index, 'strategy_order'] = np.roll(strategy_order, 0)
        elif re.search(user_b_pattern, row['User_SN']):
            latin_square.at[index, 'strategy_order'] = np.roll(strategy_order, -1)
        elif re.search(user_c_pattern, row['User_SN']):
            latin_square.at[index, 'strategy_order'] = np.roll(strategy_order, -2)
    latin_square[['1', '2', '3']] = pd.DataFrame(
        latin_square['strategy_order'].to_numpy().tolist(), index=latin_square.index)
    latin_square = latin_square.drop(columns='strategy_order').set_index('User_SN')
    latin_square = latin_square.stack(0).reset_index()
    latin_square.columns = ['User_SN', 'LatinSquare_SN', 'Rec_Type']
    latin_square['LatinSquare_SN'] = latin_square['LatinSquare_SN'].astype(int)
    latin_square.set_index(['User_SN', 'Rec_Type'], inplace=True)
    dv_matrix = pd.merge(dv_matrix, latin_square, on=['User_SN', 'Rec_Type'])
    print("------- Recommendation Effectiveness (Dependent Variables) -------")
    dv_matrix.to_csv(pathlib.Path.cwd() / "DependentVar.csv", sep='\t', encoding='utf8')
    print(dv_matrix)

    # ----------------------------------------------------------------------
    # Long Format Variables combine

    dv_matrix.reset_index(level=1, drop=False, inplace=True)
    all_matrix_long = pd.merge(mo_matrix, dv_matrix, on=['User_SN'], sort=False)
    print("------- Long Format -------")
    all_matrix_long.to_csv(pathlib.Path.cwd() / "All_LongFormat.csv", sep='\t', encoding='utf8')
    print(all_matrix_long)

    # ----------------------------------------------------------------------
    # All Data combine

    # processing & clean
    recommendation_evaluation_profile.drop(columns=['Art_Idx', 'Gen_Idx', 'Keywords', 'Strategy_Type'], inplace=True)
    recommendation_evaluation_profile = profile_processing(recommendation_evaluation_profile)
    recommendation_evaluation_profile.rename(
        columns={'Prediction_Rating': 'UserCF_Pred', 'Profile_Sim': 'CBF_Pred'}, inplace=True)
    recommendation_evaluation_profile['UserCF_Pred'].fillna(np.NaN, inplace=True)
    recommendation_evaluation_profile['CBF_Pred'].fillna(np.NaN, inplace=True)
    recommendation_evaluation_profile = \
        pd.merge(recommendation_evaluation_profile, user_list, on=['User_ID'], sort=False)
    recommendation_evaluation_profile.set_index(['User_SN', 'Rec_Type'], inplace=True)
    recommendation_evaluation_profile.sort_index(inplace=True)
    recommendation_evaluation_profile['repeated_Song_SN'] = \
        recommendation_evaluation_profile.groupby(['User_SN']).cumcount() + 1
    # merge Recommendation Song Profile
    all_matrix = pd.merge(recommendation_evaluation_profile, all_matrix_long, on=['User_SN', 'Rec_Type'], sort=False)
    print("------- All Data -------")
    all_matrix.to_csv(pathlib.Path.cwd() / "All_Data.csv", sep='\t', encoding='utf8')
    print(all_matrix)

    # ----------------------------------------------------------------------
    # Descriptive Statistics

    # Moderator Variables
    print("------- Descriptive Statistics: User Preference Structure (Moderator Variables) -------")
    mo_matrix_ds = pd.concat(
        [mo_matrix.describe(), mo_matrix.agg([pd.DataFrame.kurtosis, pd.DataFrame.skew])],
        axis=0, join='outer', sort=False).T
    mo_matrix_ds.to_csv(pathlib.Path.cwd() / "ModeratorVar_StatsDesc.csv", sep='\t', encoding='utf8')
    print(mo_matrix_ds)

    # Dependent Variables
    print("------- Descriptive Statistics: Recommendation Effectiveness (Dependent Variables) -------")
    dv_matrix_ds = pd.concat(
        [dv_matrix.groupby('Rec_Type').describe(),
         dv_matrix.groupby('Rec_Type').agg([pd.DataFrame.kurtosis, pd.DataFrame.skew])],
        axis=1, join='outer', sort=False).T.sort_index(level=0, sort_remaining=False)
    dv_matrix_ds['ALL'] = pd.concat(
        [dv_matrix.describe(), dv_matrix.agg([pd.DataFrame.kurtosis, pd.DataFrame.skew])],
        axis=0, join='outer', sort=False).unstack()
    dv_matrix_ds = dv_matrix_ds.stack().unstack(1)
    dv_matrix_ds.to_csv(pathlib.Path.cwd() / "DependentVar_StatsDesc.csv", sep='\t', encoding='utf8')
    print(dv_matrix_ds)

    # ----------------------------------------------------------------------
    # Plots Area

    print("------- Plots -------")
    # plots font style
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Cambria']

    # Long Tail plots
    usercf_rec_list = recommendation_evaluation_profile.groupby(['Rec_Type'])['Song_ID'].apply(
        lambda x: pd.unique(x))['UserCF']
    usercf_candidate_likes = candidate_user_profile.loc[
        candidate_user_profile['Song_ID'].isin(usercf_rec_list), 'Song_ID']
    cbf_rec_list = recommendation_evaluation_profile.groupby(['Rec_Type'])['Song_ID'].apply(
        lambda x: pd.unique(x))['CBF']
    cbf_candidate_likes = candidate_user_profile.loc[
        candidate_user_profile['Song_ID'].isin(cbf_rec_list), 'Song_ID']
    topplay_rec_list = recommendation_evaluation_profile.groupby(['Rec_Type'])['Song_ID'].apply(
        lambda x: pd.unique(x))['TopPlay']
    topplay_candidate_likes = candidate_user_profile.loc[
        candidate_user_profile['Song_ID'].isin(topplay_rec_list), 'Song_ID']
    long_tail_plots(candidate_user_profile['Song_ID'])
    long_tail_plots(usercf_candidate_likes)
    long_tail_plots(cbf_candidate_likes)
    long_tail_plots(topplay_candidate_likes)
    plt.show()
    return


if __name__ == "__main__":
    main()
