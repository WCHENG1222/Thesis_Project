# User Preference Structure & Recommendation Effectiveness Analysis 3.0

import pathlib
import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
import re
import matplotlib as mpl
import matplotlib.pyplot as plt

# input format
#   candidate_user_profile: | User_ID | Song_ID |
#   song_profile: | Song_ID | Title | Artist | Genre | Release_Date | Plays_count | Likes_count
#   target_user_profile: | User_ID | Song_ID |
#   scale_data: | User_ID | Pref_Ques01:scale_score |...| Pref_Ques20:scale_score | 生理性別 | 年齡 | 教育程度 | 專業領域 |
#   evaluation_data: | User_ID | Rec_Type | Latin_Square_Order
#                    | Song_01_Known:known_or_unknown_statement | Song_01_Satisfy:rating |...
#                    | Song_30_Known:known_or_unknown_statement | Song_30_Satisfy:rating |
#                    | Song_ID | Title | Artist | Genre | Release_Date | Plays_count | Likes_count |


def main():
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
    #     get global Plays_count (drop_duplicates by chosen largest Plays_count)
    global_profile['Plays_count'] = global_profile['Plays_count'].replace(0, 1)  # avoid 0 Plays_count
    global_plays = global_profile.groupby('Song_ID')['Plays_count'].max()
    global_min_plays = global_plays.min()
    global_max_plays = global_plays.max()
    log_global_plays = -np.log10(global_plays)
    print("global Plays_count:\n{0}\nglobal log(Plays_count):\n{1}".format(
        global_plays.describe().map("{:.2f}".format), log_global_plays.describe().map("{:.5f}".format)))
    #     get global Likes_count (drop_duplicates by chosen largest Likes_count)
    global_likes = global_profile.groupby('Song_ID')['Likes_count'].max()
    global_min_likes = global_likes.min()
    global_max_likes = global_likes.max()
    log_global_likes = -np.log10(global_likes)
    print("global Likes_count:\n{0}\nglobal log(Likes_count):\n{1}".format(
        global_likes.describe().map("{:.2f}".format), log_global_likes.describe().map("{:.5f}".format)))
    #     get global Release_Date
    global_release = global_profile.groupby('Song_ID')['Release_Date'].max()
    global_first_release = global_profile['Release_Date'].min() - np.timedelta64(1, 'D')  # avoid 0 Release_Date_period
    global_latest_release = global_profile['Release_Date'].max()
    global_release_period = (global_release - global_first_release) / np.timedelta64(1, 'D')  # type conversion to int
    log_global_release_period = np.log10(global_release_period)
    print("global Release_Date:\n{0}\nRelease_period:\n{1}\nglobal log(Release_period):\n{2}".format(
        global_release.describe(),
        global_release_period.describe().map("{:.2f}".format),
        log_global_release_period.describe().map("{:.5f}".format)))

    # Preference Structure (Moderator Variables)
    # --- Preference Scale: Diversity; Openness; Identity; Involvement ---
    scale_up = preference_structure_scale(scale_data)
    scale_up['rev_Invol_05'] = 6 - scale_up['Invol_05']
    scale_up['rev_Invol_06'] = 6 - scale_up['Invol_06']
    scale_up['Diversity_Scale_UP'] = (scale_up['Dive_04'] + scale_up['Dive_03']) / 2
    scale_up['Openness_Scale_UP'] = (scale_up['Open_05'] + scale_up['Open_06'] + scale_up['Open_03']) / 3
    scale_up['Identity_Scale_UP'] = (scale_up['Iden_04'] + scale_up['Iden_03']) / 2
    scale_up['Involvement_Scale_UP'] = (scale_up['rev_Invol_06'] + scale_up['Invol_04'] + scale_up['Invol_02']) / 3
    scale_up = scale_up.round(5)
    # ------- Demographic Statistics data -------
    print("------- Demographic Statistics -------")
    demog = scale_data.filter(regex='生理性別|年齡|教育程度|專業領域')
    print(pd.value_counts(demog['生理性別']))
    print(pd.value_counts(demog['年齡']))
    print(pd.value_counts(demog['教育程度']))
    print(pd.value_counts(demog['專業領域']))
    # --- User Profile Metrics ---
    # --- Diversity ---
    # ------- entropy_based -------
    div_entropy_up = target_user_profile.groupby(['User_ID']).apply(entropy_based_diversity).round(5)
    div_entropy_up.rename('Div_Entropy_UP', inplace=True)
    # ------- similarity_based -------
    div_sim_up = target_user_profile.groupby(['User_ID']).apply(similarity_based_diversity)
    # ----------- averaged -----------
    div_avgsim_up = div_sim_up.apply(lambda x: x.sum() / (x.shape[0] * (x.shape[1] - 1))).round(5)
    div_avgsim_up.rename('Div_AvgSim_UP', inplace=True)
    # ----------- summation -----------
    div_sumsim_up = div_sim_up.apply(lambda x: np.tril(x).sum()).round(5)
    div_sumsim_up.rename('Div_SumSim_UP', inplace=True)
    # ------- genre_count_user_profile -------
    div_genrecount_up = target_user_profile.groupby(['User_ID'])['Gen_Idx'].nunique()
    div_genrecount_up.rename('Div_GenreCount_UP', inplace=True)
    # ------- artist_count_user_profile -------
    div_artistcount_up = target_user_profile.groupby(['User_ID'])['Art_Idx'].nunique()
    div_artistcount_up.rename('Div_ArtistCount_UP', inplace=True)
    # --- Openness ---
    # ------- min_max_normalization_plays_popularity_based -------
    opn_mmpopplays_up = target_user_profile.groupby(['User_ID'])['Plays_count'].apply(
        minmax_popularity_based_novelty, global_min_plays, global_max_plays)
    # ----------- averaged -----------
    opn_mmavgpopplays_up = opn_mmpopplays_up.apply(lambda x: np.mean(x)).round(5)
    opn_mmavgpopplays_up.rename('Opn_MMAvgPopPlays_UP', inplace=True)
    # ----------- summation -----------
    opn_mmsumpopplays_up = opn_mmpopplays_up.apply(lambda x: np.sum(x)).round(5)
    opn_mmsumpopplays_up.rename('Opn_MMSumPopPlays_UP', inplace=True)
    # ----------- median -----------
    opn_mmmedpopplays_up = opn_mmpopplays_up.apply(lambda x: np.median(x)).round(5)
    opn_mmmedpopplays_up.rename('Opn_MMMedPopPlays_UP', inplace=True)
    # ------- log_plays_popularity_based -------
    opn_logpopplays_up = target_user_profile.groupby(['User_ID'])['Plays_count'].apply(
        log_popularity_based_novelty)
    # ----------- averaged -----------
    opn_logavgpopplays_up = opn_logpopplays_up.apply(lambda x: np.mean(x)).round(5)
    opn_logavgpopplays_up.rename('Opn_LogAvgPopPlays_UP', inplace=True)
    # ----------- summation -----------
    opn_logsumpopplays_up = opn_logpopplays_up.apply(lambda x: np.sum(x)).round(5)
    opn_logsumpopplays_up.rename('Opn_LogSumPopPlays_UP', inplace=True)
    # ----------- median -----------
    opn_logmedpopplays_up = opn_logpopplays_up.apply(lambda x: np.median(x)).round(5)
    opn_logmedpopplays_up.rename('Opn_LogMedPopPlays_UP', inplace=True)
    # ------- min_max_normalization_likes_popularity_based -------
    opn_mmpoplikes_up = target_user_profile.groupby(['User_ID'])['Likes_count'].apply(
        minmax_popularity_based_novelty, global_min_likes, global_max_likes)
    # ----------- averaged -----------
    opn_mmavgpoplikes_up = opn_mmpoplikes_up.apply(lambda x: np.mean(x)).round(5)
    opn_mmavgpoplikes_up.rename('Opn_MMAvgPopLikes_UP', inplace=True)
    # ----------- summation -----------
    opn_mmsumpoplikes_up = opn_mmpoplikes_up.apply(lambda x: np.sum(x)).round(5)
    opn_mmsumpoplikes_up.rename('Opn_MMSumPopLikes_UP', inplace=True)
    # ----------- median -----------
    opn_mmmedpoplikes_up = opn_mmpoplikes_up.apply(lambda x: np.median(x)).round(5)
    opn_mmmedpoplikes_up.rename('Opn_MMMedPopLikes_UP', inplace=True)
    # ------- log_likes_popularity_based -------
    opn_logpoplikes_up = target_user_profile.groupby(['User_ID'])['Likes_count'].apply(
        log_popularity_based_novelty)
    # ----------- averaged -----------
    opn_logavgpoplikes_up = opn_logpoplikes_up.apply(lambda x: np.mean(x)).round(5)
    opn_logavgpoplikes_up.rename('Opn_LogAvgPopLikes_UP', inplace=True)
    # ----------- summation -----------
    opn_logsumpoplikes_up = opn_logpoplikes_up.apply(lambda x: np.sum(x)).round(5)
    opn_logsumpoplikes_up.rename('Opn_LogSumPopLikes_UP', inplace=True)
    # ----------- median -----------
    opn_logmedpoplikes_up = opn_logpoplikes_up.apply(lambda x: np.median(x)).round(5)
    opn_logmedpoplikes_up.rename('Opn_LogMedPopLikes_UP', inplace=True)
    # ------- min_max_normalization_time_aware -------
    opn_mmtime_up = target_user_profile.groupby(['User_ID'])['Release_Date'].apply(
        minmax_time_aware_novelty, global_first_release, global_latest_release)
    # ----------- averaged -----------
    opn_mmavgtime_up = opn_mmtime_up.apply(lambda x: np.mean(x)).round(5)
    opn_mmavgtime_up.rename('Opn_MMAvgTime_UP', inplace=True)
    # ----------- summation -----------
    opn_mmsumtime_up = opn_mmtime_up.apply(lambda x: np.sum(x)).round(5)
    opn_mmsumtime_up.rename('Opn_MMSumTime_UP', inplace=True)
    # ----------- median -----------
    opn_mmmedtime_up = opn_mmtime_up.apply(lambda x: np.median(x)).round(5)
    opn_mmmedtime_up.rename('Opn_MMMedTime_UP', inplace=True)
    # ------- log_time_aware -------
    opn_logtime_up = target_user_profile.groupby(['User_ID'])['Release_Date'].apply(
        log_time_aware_novelty, global_first_release)
    # ----------- averaged -----------
    opn_logavgtime_up = opn_logtime_up.apply(lambda x: np.mean(x)).round(5)
    opn_logavgtime_up.rename('Opn_LogAvgTime_UP', inplace=True)
    # ----------- summation -----------
    opn_logsumtime_up = opn_logtime_up.apply(lambda x: np.sum(x)).round(5)
    opn_logsumtime_up.rename('Opn_LogSumTime_UP', inplace=True)
    # ----------- median -----------
    opn_logmedtime_up = opn_logtime_up.apply(lambda x: np.median(x)).round(5)
    opn_logmedtime_up.rename('Opn_LogMedTime_UP', inplace=True)
    # ------- count_user_profile -------
    opn_userprofcount_up = target_user_profile.groupby(['User_ID'])['Song_ID'].nunique()
    opn_userprofcount_up.rename('Opn_UserProfCount_UP', inplace=True)

    # Moderator Variables
    mo_matrix = pd.concat([
        scale_up, div_entropy_up, div_avgsim_up, div_sumsim_up, div_genrecount_up, div_artistcount_up,
        opn_mmavgpopplays_up, opn_mmsumpopplays_up, opn_mmmedpopplays_up,
        opn_logavgpopplays_up, opn_logsumpopplays_up, opn_logmedpopplays_up,
        opn_mmavgpoplikes_up, opn_mmsumpoplikes_up, opn_mmmedpoplikes_up,
        opn_logavgpoplikes_up, opn_logsumpoplikes_up, opn_logmedpoplikes_up,
        opn_mmavgtime_up, opn_mmsumtime_up, opn_mmmedtime_up, opn_logavgtime_up, opn_logsumtime_up, opn_logmedtime_up, 
        opn_userprofcount_up, demog],
        axis=1, sort=False)
    #     user data De-identification & Anonymization
    mo_matrix.index.names = ['User_ID']
    mo_matrix = pd.merge(mo_matrix, user_list, on=['User_ID'], right_index=True, sort=False)
    mo_matrix.set_index('User_SN', inplace=True)
    print("------- User Preference Structure (Moderator Variables) -------")
    mo_matrix.to_csv(pathlib.Path.cwd() / "ModeratorVar.csv", sep='\t', encoding='utf8')
    print(mo_matrix)

    # Recommendation Effectiveness (Dependent Variables)
    # --- Coverage ---
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
    # --- Unknown ---
    unknown_count_re = recommendation_evaluation_profile.groupby(
        ['User_ID', 'Rec_Type'])['Known'].value_counts(dropna=False).unstack(level=-1, fill_value=0)['unknown']
    unknown_count_re.rename('Unknown_Count_RE', inplace=True)
    unknown_ratio_re = (unknown_count_re / 30).round(5)
    unknown_ratio_re.rename('Unknown_Ratio_RE', inplace=True)
    # --- Satisfaction ---
    satisfaction_re = recommendation_evaluation_profile.groupby(['User_ID', 'Rec_Type']).apply(satisfaction).round(5)
    satisfaction_re.rename('Satisfaction_RE', inplace=True)
    # --- Serendipity ---
    # ----------- unknown & satisfaction rated 3~4 -----------
    unknown_satisfy_filter = \
        (recommendation_evaluation_profile['Known'] == 'unknown') & (recommendation_evaluation_profile['Satisfy'] > 2)
    recommendation_evaluation_profile['Unknown_Satisfy'] = recommendation_evaluation_profile['Satisfy']
    recommendation_evaluation_profile.loc[~unknown_satisfy_filter, 'Unknown_Satisfy'] = 0
    serendipity_re = recommendation_evaluation_profile.groupby(['User_ID', 'Rec_Type']).apply(serendipity).round(5)
    serendipity_re.rename('Serendipity_RE', inplace=True)
    # ----------- strict version: unknown & satisfaction rated 4 -----------
    strict_unknown_satisfy_filter = \
        (recommendation_evaluation_profile['Known'] == 'unknown') & (recommendation_evaluation_profile['Satisfy'] > 3)
    recommendation_evaluation_profile['strict_Unknown_Satisfy'] = recommendation_evaluation_profile['Satisfy']
    recommendation_evaluation_profile.loc[~strict_unknown_satisfy_filter, 'strict_Unknown_Satisfy'] = 0
    strict_serendipity_re = recommendation_evaluation_profile.groupby(['User_ID', 'Rec_Type']).apply(
        strict_serendipity).round(5)
    strict_serendipity_re.rename('strict_Serendipity_RE', inplace=True)
    # --- Diversity ---
    # ------- entropy_based -------
    div_entropy_re = recommendation_evaluation_profile.groupby(['User_ID', 'Rec_Type']).apply(
        entropy_based_diversity).round(5)
    div_entropy_re.rename('Div_Entropy_RE', inplace=True)
    # ------- similarity_based -------
    div_sim_re = recommendation_evaluation_profile.groupby(['User_ID', 'Rec_Type']).apply(
        similarity_based_diversity)
    # ----------- averaged -----------
    div_avgsim_re = div_sim_re.apply(lambda x: x.sum() / (x.shape[0] * (x.shape[1] - 1))).round(5)
    div_avgsim_re.rename('Div_AvgSim_RE', inplace=True)
    # ----------- summation -----------
    div_sumsim_re = div_sim_re.apply(lambda x: np.tril(x).sum()).round(5)
    div_sumsim_re.rename('Div_SumSim_RE', inplace=True)
    # --- Novelty ---
    unknown_filter = recommendation_evaluation_profile['Known'] == 'unknown'
    # ------- min_max_normalization_plays_popularity_based(unknown) -------
    recommendation_evaluation_profile['Unknown_Plays_count'] = recommendation_evaluation_profile['Plays_count']
    recommendation_evaluation_profile.loc[~unknown_filter, 'Unknown_Plays_count'] = global_max_plays
    nov_mmpopplays_re = recommendation_evaluation_profile.groupby(
        ['User_ID', 'Rec_Type'])['Unknown_Plays_count'].apply(
        minmax_popularity_based_novelty, global_min_plays, global_max_plays)
    # ----------- averaged -----------
    nov_mmavgpopplays_re = nov_mmpopplays_re.apply(lambda x: np.mean(x)).round(5)
    nov_mmavgpopplays_re.rename('Nov_MMAvgPopPlays_RE', inplace=True)
    # ----------- summation -----------
    nov_mmsumpopplays_re = nov_mmpopplays_re.apply(lambda x: np.sum(x)).round(5)
    nov_mmsumpopplays_re.rename('Nov_MMSumPopPlays_RE', inplace=True)
    # ----------- median -----------
    nov_mmmedpopplays_re = nov_mmpopplays_re.apply(lambda x: np.median(x)).round(5)
    nov_mmmedpopplays_re.rename('Nov_MMMedPopPlays_RE', inplace=True)
    # ------- log_plays_popularity_based(unknown) -------
    nov_logpopplays_re = recommendation_evaluation_profile.groupby(
        ['User_ID', 'Rec_Type'])['Unknown_Plays_count'].apply(
        log_popularity_based_novelty)
    # ----------- averaged -----------
    nov_logavgpopplays_re = nov_logpopplays_re.apply(lambda x: np.mean(x)).round(5)
    nov_logavgpopplays_re.rename('Nov_LogAvgPopPlays_RE', inplace=True)
    # ----------- summation -----------
    nov_logsumpopplays_re = nov_logpopplays_re.apply(lambda x: np.sum(x)).round(5)
    nov_logsumpopplays_re.rename('Nov_LogSumPopPlays_RE', inplace=True)
    # ----------- median -----------
    nov_logmedpopplays_re = nov_logpopplays_re.apply(lambda x: np.median(x)).round(5)
    nov_logmedpopplays_re.rename('Nov_LogMedPopPlays_RE', inplace=True)
    # ------- min_max_normalization_likes_popularity_based(unknown) -------
    recommendation_evaluation_profile['Unknown_Likes_count'] = recommendation_evaluation_profile['Likes_count']
    recommendation_evaluation_profile.loc[~unknown_filter, 'Unknown_Likes_count'] = global_max_likes
    nov_mmpoplikes_re = recommendation_evaluation_profile.groupby(
        ['User_ID', 'Rec_Type'])['Unknown_Likes_count'].apply(
        minmax_popularity_based_novelty, global_min_likes, global_max_likes)
    # ----------- averaged -----------
    nov_mmavgpoplikes_re = nov_mmpoplikes_re.apply(lambda x: np.mean(x)).round(5)
    nov_mmavgpoplikes_re.rename('Nov_MMAvgPopLikes_RE', inplace=True)
    # ----------- summation -----------
    nov_mmsumpoplikes_re = nov_mmpoplikes_re.apply(lambda x: np.sum(x)).round(5)
    nov_mmsumpoplikes_re.rename('Nov_MMSumPopLikes_RE', inplace=True)
    # ----------- median -----------
    nov_mmmedpoplikes_re = nov_mmpoplikes_re.apply(lambda x: np.median(x)).round(5)
    nov_mmmedpoplikes_re.rename('Nov_MMMedPopLikes_RE', inplace=True)
    # ------- log_likes_popularity_based(unknown) -------
    nov_logpoplikes_re = recommendation_evaluation_profile.groupby(
        ['User_ID', 'Rec_Type'])['Unknown_Likes_count'].apply(
        log_popularity_based_novelty)
    # ----------- averaged -----------
    nov_logavgpoplikes_re = nov_logpoplikes_re.apply(lambda x: np.mean(x)).round(5)
    nov_logavgpoplikes_re.rename('Nov_LogAvgPopLikes_RE', inplace=True)
    # ----------- summation -----------
    nov_logsumpoplikes_re = nov_logpoplikes_re.apply(lambda x: np.sum(x)).round(5)
    nov_logsumpoplikes_re.rename('Nov_LogSumPopLikes_RE', inplace=True)
    # ----------- median -----------
    nov_logmedpoplikes_re = nov_logpoplikes_re.apply(lambda x: np.median(x)).round(5)
    nov_logmedpoplikes_re.rename('Nov_LogMedPopLikes_RE', inplace=True)
    # ------- min_max_normalization_time_aware(unknown) -------
    recommendation_evaluation_profile['Unknown_Release_Date'] = recommendation_evaluation_profile['Release_Date']
    recommendation_evaluation_profile.loc[~unknown_filter, 'Unknown_Release_Date'] = \
        global_first_release + np.timedelta64(1, 'D')
    nov_mmtime_re = recommendation_evaluation_profile.groupby(
        ['User_ID', 'Rec_Type'])['Unknown_Release_Date'].apply(
        minmax_time_aware_novelty, global_first_release, global_latest_release)
    # ----------- averaged -----------
    nov_mmavgtime_re = nov_mmtime_re.apply(lambda x: np.mean(x)).round(5)
    nov_mmavgtime_re.rename('Nov_MMAvgTime_RE', inplace=True)
    # ----------- summation -----------
    nov_mmsumtime_re = nov_mmtime_re.apply(lambda x: np.sum(x)).round(5)
    nov_mmsumtime_re.rename('Nov_MMSumTime_RE', inplace=True)
    # ----------- median -----------
    nov_mmmedtime_re = nov_mmtime_re.apply(lambda x: np.median(x)).round(5)
    nov_mmmedtime_re.rename('Nov_MMMedTime_RE', inplace=True)
    # ------- log_time_aware(unknown) -------
    nov_logtime_re = recommendation_evaluation_profile.groupby(
        ['User_ID', 'Rec_Type'])['Unknown_Release_Date'].apply(
        log_time_aware_novelty, global_first_release)
    # ----------- averaged -----------
    nov_logavgtime_re = nov_logtime_re.apply(lambda x: np.mean(x)).round(5)
    nov_logavgtime_re.rename('Nov_LogAvgTime_RE', inplace=True)
    # ----------- summation -----------
    nov_logsumtime_re = nov_logtime_re.apply(lambda x: np.sum(x)).round(5)
    nov_logsumtime_re.rename('Nov_LogSumTime_RE', inplace=True)
    # ----------- median -----------
    nov_logmedtime_re = nov_logtime_re.apply(lambda x: np.median(x)).round(5)
    nov_logmedtime_re.rename('Nov_LogMedTime_RE', inplace=True)
    # ------- distance_based(unknown) -------
    nov_dist_re = recommendation_evaluation_profile[unknown_filter].groupby(['User_ID', 'Rec_Type']).apply(
        distance_based_novelty, target_user_profile)
    # ----------- averaged -----------
    nov_avgdist_re = nov_dist_re.apply(lambda x: x.sum() / (x.shape[0] * 30)).round(5)
    nov_avgdist_re.rename('Nov_AvgDist_RE', inplace=True)
    nov_avgdist_re = nov_avgdist_re.reindex(
        recommendation_evaluation_profile.groupby(['User_ID', 'Rec_Type']).count().index, fill_value=0)
    # ----------- summation -----------
    nov_sumdist_re = nov_dist_re.apply(lambda x: x.sum()).round(5)
    nov_sumdist_re.rename('Nov_SumDist_RE', inplace=True)
    nov_sumdist_re = nov_sumdist_re.reindex(
        recommendation_evaluation_profile.groupby(['User_ID', 'Rec_Type']).count().index, fill_value=0)

    # Dependent Variables
    dv_matrix = pd.concat([
        unknown_count_re, unknown_ratio_re, satisfaction_re, serendipity_re, strict_serendipity_re,
        div_entropy_re, div_avgsim_re, div_sumsim_re,
        nov_mmavgpopplays_re, nov_mmsumpopplays_re, nov_mmmedpopplays_re,
        nov_logavgpopplays_re, nov_logsumpopplays_re, nov_logmedpopplays_re,
        nov_mmavgpoplikes_re, nov_mmsumpoplikes_re, nov_mmmedpoplikes_re,
        nov_logavgpoplikes_re, nov_logsumpoplikes_re, nov_logmedpoplikes_re,
        nov_mmavgtime_re, nov_mmsumtime_re, nov_mmmedtime_re, nov_logavgtime_re, nov_logsumtime_re, nov_logmedtime_re,
        nov_avgdist_re, nov_sumdist_re],
        axis=1, sort=False)
    #     user data De-identification & Anonymization
    dv_matrix = pd.merge(dv_matrix, user_list, on=['User_ID'], right_index=True, sort=False)
    dv_matrix.reset_index(level=0, drop=True, inplace=True)
    dv_matrix.set_index(['User_SN', dv_matrix.index], inplace=True)
    #     get Dummy Variables
    dv_matrix['Rec_Type_SN'] = dv_matrix.rename(
        level='Rec_Type', index={'UserCF': 0, 'CBF': 1, 'TopPlay': 2}).index.get_level_values('Rec_Type')
    rec_type_dummy = \
        pd.get_dummies(dv_matrix.index.get_level_values('Rec_Type')).add_prefix('dummy_').set_index(dv_matrix.index)
    dv_matrix = dv_matrix.join(rec_type_dummy, sort=False)
    #     get Latin Square Order
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

    # Long Format Variables combine
    dv_matrix.reset_index(level=1, drop=False, inplace=True)
    all_matrix_long = pd.merge(mo_matrix, dv_matrix, on=['User_SN'], sort=False)
    print("------- Long Format -------")
    all_matrix_long.to_csv(pathlib.Path.cwd() / "All_LongFormat.csv", sep='\t', encoding='utf8')
    print(all_matrix_long)

    # Wide Format Variables combine
    unknown_count_re_wd = unknown_count_re.unstack().add_prefix('Unknown_Count_RE_')
    unknown_ratio_re_wd = unknown_ratio_re.unstack().add_prefix('Unknown_Ratio_RE_')
    satisfaction_re_wd = satisfaction_re.unstack().add_prefix('Satisfaction_RE_')
    serendipity_re_wd = serendipity_re.unstack().add_prefix('Serendipity_RE_')
    strict_serendipity_re_wd = strict_serendipity_re.unstack().add_prefix('strict_Serendipity_RE_')
    div_entropy_re_wd = div_entropy_re.unstack().add_prefix('Div_Entropy_RE_')
    div_avgsim_re_wd = div_avgsim_re.unstack().add_prefix('Div_AvgSim_RE_')
    div_sumsim_re_wd = div_sumsim_re.unstack().add_prefix('Div_SumSim_RE_')
    nov_mmavgpopplays_re_wd = nov_mmavgpopplays_re.unstack().add_prefix('Nov_MMAvgPopPlays_RE_')
    nov_mmsumpopplays_re_wd = nov_mmsumpopplays_re.unstack().add_prefix('Nov_MMSumPopPlays_RE_')
    nov_mmmedpopplays_re_wd = nov_mmmedpopplays_re.unstack().add_prefix('Nov_MMMedPopPlays_RE_')
    nov_logavgpopplays_re_wd = nov_logavgpopplays_re.unstack().add_prefix('Nov_LogAvgPopPlays_RE_')
    nov_logsumpopplays_re_wd = nov_logsumpopplays_re.unstack().add_prefix('Nov_LogSumPopPlays_RE_')
    nov_logmedpopplays_re_wd = nov_logmedpopplays_re.unstack().add_prefix('Nov_LogMedPopPlays_RE_')
    nov_mmavgpoplikes_re_wd = nov_mmavgpoplikes_re.unstack().add_prefix('Nov_MMAvgPopLikes_RE_')
    nov_mmsumpoplikes_re_wd = nov_mmsumpoplikes_re.unstack().add_prefix('Nov_MMSumPopLikes_RE_')
    nov_mmmedpoplikes_re_wd = nov_mmmedpoplikes_re.unstack().add_prefix('Nov_MMMedPopLikes_RE_')
    nov_logavgpoplikes_re_wd = nov_logavgpoplikes_re.unstack().add_prefix('Nov_LogAvgPopLikes_RE_')
    nov_logsumpoplikes_re_wd = nov_logsumpoplikes_re.unstack().add_prefix('Nov_LogSumPopLikes_RE_')
    nov_logmedpoplikes_re_wd = nov_logmedpoplikes_re.unstack().add_prefix('Nov_LogMedPopLikes_RE_')
    nov_mmavgtime_re_wd = nov_mmavgtime_re.unstack().add_prefix('Nov_MMAvgTime_RE_')
    nov_mmsumtime_re_wd = nov_mmsumtime_re.unstack().add_prefix('Nov_MMSumTime_RE_')
    nov_mmmedtime_re_wd = nov_mmmedtime_re.unstack().add_prefix('Nov_MMMedTime_RE_')
    nov_logavgtime_re_wd = nov_logavgtime_re.unstack().add_prefix('Nov_LogAvgTime_RE_')
    nov_logsumtime_re_wd = nov_logsumtime_re.unstack().add_prefix('Nov_LogSumTime_RE_')
    nov_logmedtime_re_wd = nov_logmedtime_re.unstack().add_prefix('Nov_LogMedTime_RE_')
    nov_avgdist_re_wd = nov_avgdist_re.unstack().add_prefix('Nov_AvgDist_RE_')
    nov_sumdist_re_wd = nov_sumdist_re.unstack().add_prefix('Nov_SumDist_RE_')
    all_matrix_wide = pd.concat([
        scale_up, div_entropy_up, div_avgsim_up, div_sumsim_up, div_genrecount_up, div_artistcount_up,
        opn_mmavgpopplays_up, opn_mmsumpopplays_up, opn_mmmedpopplays_up,
        opn_logavgpopplays_up, opn_logsumpopplays_up, opn_logmedpopplays_up,
        opn_mmavgpoplikes_up, opn_mmsumpoplikes_up, opn_mmmedpoplikes_up,
        opn_logavgpoplikes_up, opn_logsumpoplikes_up, opn_logmedpoplikes_up,
        opn_mmavgtime_up, opn_mmsumtime_up, opn_mmmedtime_up, opn_logavgtime_up, opn_logsumtime_up, opn_logmedtime_up,
        opn_userprofcount_up, demog,
        unknown_count_re_wd, unknown_ratio_re_wd, satisfaction_re_wd, serendipity_re_wd, strict_serendipity_re_wd,
        div_entropy_re_wd, div_avgsim_re_wd, div_sumsim_re_wd,
        nov_mmavgpopplays_re_wd, nov_mmsumpopplays_re_wd, nov_mmmedpopplays_re_wd,
        nov_logavgpopplays_re_wd, nov_logsumpopplays_re_wd, nov_logmedpopplays_re_wd,
        nov_mmavgpoplikes_re_wd, nov_mmsumpoplikes_re_wd, nov_mmmedpoplikes_re_wd,
        nov_logavgpoplikes_re_wd, nov_logsumpoplikes_re_wd, nov_logmedpoplikes_re_wd,
        nov_mmavgtime_re_wd, nov_mmsumtime_re_wd, nov_mmmedtime_re_wd,
        nov_logavgtime_re_wd, nov_logsumtime_re_wd, nov_logmedtime_re_wd,
        nov_avgdist_re_wd, nov_sumdist_re_wd],
        axis=1, sort=False)
    all_matrix_wide.index.names = ['User_ID']
    all_matrix_wide = pd.merge(all_matrix_wide, user_list, on=['User_ID'], right_index=True, sort=False)
    all_matrix_wide.set_index('User_SN', inplace=True)
    print("------- Wide Format -------")
    all_matrix_wide.to_csv(pathlib.Path.cwd() / "All_WideFormat.csv", sep='\t', encoding='utf8')
    print(all_matrix_wide)

    # All Data combine
    #     processing & clean
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
    #     merge Recommendation Song Profile
    all_matrix = pd.merge(recommendation_evaluation_profile, all_matrix_long, on=['User_SN', 'Rec_Type'], sort=False)
    print("------- All Data -------")
    all_matrix.to_csv(pathlib.Path.cwd() / "All_Data.csv", sep='\t', encoding='utf8')
    print(all_matrix)

    # Descriptive Statistics
    # --- Moderator Variables ---
    print("------- Descriptive Statistics: User Preference Structure (Moderator Variables) -------")
    mo_matrix_ds = pd.concat(
        [mo_matrix.describe(), mo_matrix.agg([pd.DataFrame.kurtosis, pd.DataFrame.skew])],
        axis=0, join='outer', sort=False).T
    mo_matrix_ds.to_csv(pathlib.Path.cwd() / "ModeratorVar_StatsDesc.csv", sep='\t', encoding='utf8')
    print(mo_matrix_ds)
    # --- Dependent Variables ---
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

    # Plots Area
    print("------- Plots -------")
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Cambria']
    # --- Long Tail plots ---
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

    # --- Probability Distributions plots ---
    # ------- Moderator Variables plots -------
    fig_scale_up, axs_scale_up = plt.subplots(2, 2, dpi=200, constrained_layout=True)
    scale_up.Diversity_Scale_UP.plot(ax=axs_scale_up[0, 0], kind='kde', title='Diversity_Scale_UP', color='#ff6f61')
    scale_up.Openness_Scale_UP.plot(ax=axs_scale_up[0, 1], kind='kde', title='Openness_Scale_UP', color='#ff6f61')
    scale_up.Identity_Scale_UP.plot(ax=axs_scale_up[1, 0], kind='kde', title='Identity_Scale_UP', color='#ff6f61')
    scale_up.Involvement_Scale_UP.plot(ax=axs_scale_up[1, 1], kind='kde', title='Involvement_Scale_UP', color='#ff6f61')

    fig_div_up, axs_div_up = plt.subplots(2, 2, dpi=200, constrained_layout=True)
    div_entropy_up.plot(ax=axs_div_up[0, 0], kind='kde', title='Div_Entropy_UP', color='#ff6f61')
    div_avgsim_up.plot(ax=axs_div_up[0, 1], kind='kde', title='Div_AvgSim_UP', color='#ff6f61')
    div_sumsim_up.plot(ax=axs_div_up[1, 0], kind='kde', title='Div_SumSim_UP', color='#ff6f61')

    fig_div_up_tag, axs_div_up_tag = plt.subplots(2, 2, dpi=200, constrained_layout=True)
    div_genrecount_up.plot(ax=axs_div_up_tag[0, 0], kind='kde', title='Div_GenreCount_UP', color='#ff6f61')
    div_artistcount_up.plot(ax=axs_div_up_tag[0, 1], kind='kde', title='Div_ArtistCount_UP', color='#ff6f61')

    fig_opn_up_mmavg, axs_opn_up_mmavg = plt.subplots(2, 2, dpi=200, constrained_layout=True)
    opn_mmavgpopplays_up.plot(ax=axs_opn_up_mmavg[0, 0], kind='kde', title='Opn_MMAvgPopPlays_UP', color='#ff6f61')
    opn_mmavgpoplikes_up.plot(ax=axs_opn_up_mmavg[0, 1], kind='kde', title='Opn_MMAvgPopLikes_UP', color='#ff6f61')
    opn_mmavgtime_up.plot(ax=axs_opn_up_mmavg[1, 0], kind='kde', title='Opn_MMAvgTime_UP', color='#ff6f61')
    opn_userprofcount_up.plot(ax=axs_opn_up_mmavg[1, 1], kind='kde', title='Opn_UserProfCount_UP', color='#ff6f61')

    fig_opn_up_logavg, axs_opn_up_logavg = plt.subplots(2, 2, dpi=200, constrained_layout=True)
    opn_logavgpopplays_up.plot(ax=axs_opn_up_logavg[0, 0], kind='kde', title='Opn_LogAvgPopPlays_UP', color='#ff6f61')
    opn_logavgpoplikes_up.plot(ax=axs_opn_up_logavg[0, 1], kind='kde', title='Opn_LogAvgPopLikes_UP', color='#ff6f61')
    opn_logavgtime_up.plot(ax=axs_opn_up_logavg[1, 0], kind='kde', title='Opn_LogAvgTime_UP', color='#ff6f61')
    opn_userprofcount_up.plot(ax=axs_opn_up_logavg[1, 1], kind='kde', title='Opn_UserProfCount_UP', color='#ff6f61')

    fig_opn_up_mmsum, axs_opn_up_mmsum = plt.subplots(2, 2, dpi=200, constrained_layout=True)
    opn_mmsumpopplays_up.plot(ax=axs_opn_up_mmsum[0, 0], kind='kde', title='Opn_MMSumPopPlays_UP', color='#ff6f61')
    opn_mmsumpoplikes_up.plot(ax=axs_opn_up_mmsum[0, 1], kind='kde', title='Opn_MMSumPopLikes_UP', color='#ff6f61')
    opn_mmsumtime_up.plot(ax=axs_opn_up_mmsum[1, 0], kind='kde', title='Opn_MMSumTime_UP', color='#ff6f61')
    opn_userprofcount_up.plot(ax=axs_opn_up_mmsum[1, 1], kind='kde', title='Opn_UserProfCount_UP', color='#ff6f61')

    fig_opn_up_logsum, axs_opn_up_logsum = plt.subplots(2, 2, dpi=200, constrained_layout=True)
    opn_logsumpopplays_up.plot(ax=axs_opn_up_logsum[0, 0], kind='kde', title='Opn_LogSumPopPlays_UP', color='#ff6f61')
    opn_logsumpoplikes_up.plot(ax=axs_opn_up_logsum[0, 1], kind='kde', title='Opn_LogSumPopLikes_UP', color='#ff6f61')
    opn_logsumtime_up.plot(ax=axs_opn_up_logsum[1, 0], kind='kde', title='Opn_LogSumTime_UP', color='#ff6f61')
    opn_userprofcount_up.plot(ax=axs_opn_up_logsum[1, 1], kind='kde', title='Opn_UserProfCount_UP', color='#ff6f61')

    fig_opn_up_mmmed, axs_opn_up_mmmed = plt.subplots(2, 2, dpi=200, constrained_layout=True)
    opn_mmmedpopplays_up.plot(ax=axs_opn_up_mmmed[0, 0], kind='kde', title='Opn_MMMedPopPlays_UP', color='#ff6f61')
    opn_mmmedpoplikes_up.plot(ax=axs_opn_up_mmmed[0, 1], kind='kde', title='Opn_MMMedPopLikes_UP', color='#ff6f61')
    opn_mmmedtime_up.plot(ax=axs_opn_up_mmmed[1, 0], kind='kde', title='Opn_MMMedTime_UP', color='#ff6f61')
    opn_userprofcount_up.plot(ax=axs_opn_up_mmmed[1, 1], kind='kde', title='Opn_UserProfCount_UP', color='#ff6f61')

    fig_opn_up_logmed, axs_opn_up_logmed = plt.subplots(2, 2, dpi=200, constrained_layout=True)
    opn_logmedpopplays_up.plot(ax=axs_opn_up_logmed[0, 0], kind='kde', title='Opn_LogMedPopPlays_UP', color='#ff6f61')
    opn_logmedpoplikes_up.plot(ax=axs_opn_up_logmed[0, 1], kind='kde', title='Opn_LogMedPopLikes_UP', color='#ff6f61')
    opn_logmedtime_up.plot(ax=axs_opn_up_logmed[1, 0], kind='kde', title='Opn_LogMedTime_UP', color='#ff6f61')
    opn_userprofcount_up.plot(ax=axs_opn_up_logmed[1, 1], kind='kde', title='Opn_UserProfCount_UP', color='#ff6f61')

    # ------- Dependent Variables plots -------
    fig_rating_re, axs_rating_re = plt.subplots(2, 2, dpi=200, constrained_layout=True)
    unknown_ratio_re.plot(ax=axs_rating_re[0, 0], kind='kde', title='Unknown_Ratio_RE', color='#3f51b5')
    satisfaction_re.plot(ax=axs_rating_re[0, 1], kind='kde', title='Satisfaction_RE', color='#3f51b5')
    serendipity_re.plot(ax=axs_rating_re[1, 0], kind='kde', title='Serendipity_RE', color='#3f51b5')
    strict_serendipity_re.plot(ax=axs_rating_re[1, 1], kind='kde', title='strict_Serendipity_RE', color='#3f51b5')

    fig_div_re, axs_div_re = plt.subplots(2, 2, dpi=200, constrained_layout=True)
    div_entropy_re.plot(ax=axs_div_re[0, 0], kind='kde', title='Div_Entropy_RE', color='#3f51b5')
    div_avgsim_re.plot(ax=axs_div_re[0, 1], kind='kde', title='Div_AvgSim_RE', color='#3f51b5')
    div_sumsim_re.plot(ax=axs_div_re[1, 0], kind='kde', title='Div_SumSim_RE', color='#3f51b5')

    fig_nov_re_mmavg, axs_nov_re_mmavg = plt.subplots(2, 2, dpi=200, constrained_layout=True)
    nov_mmavgpopplays_re.plot(ax=axs_nov_re_mmavg[0, 0], kind='kde', title='Nov_MMAvgPopPlays_RE', color='#3f51b5')
    nov_mmavgpoplikes_re.plot(ax=axs_nov_re_mmavg[0, 1], kind='kde', title='Nov_MMAvgPopLikes_RE', color='#3f51b5')
    nov_mmavgtime_re.plot(ax=axs_nov_re_mmavg[1, 0], kind='kde', title='Nov_MMAvgTime_RE', color='#3f51b5')
    nov_avgdist_re.plot(ax=axs_nov_re_mmavg[1, 1], kind='kde', title='Nov_AvgDist_RE', color='#3f51b5')

    fig_nov_re_logavg, axs_nov_re_logavg = plt.subplots(2, 2, dpi=200, constrained_layout=True)
    nov_logavgpopplays_re.plot(ax=axs_nov_re_logavg[0, 0], kind='kde', title='Nov_LogAvgPopPlays_RE', color='#3f51b5')
    nov_logavgpoplikes_re.plot(ax=axs_nov_re_logavg[0, 1], kind='kde', title='Nov_LogAvgPopLikes_RE', color='#3f51b5')
    nov_logavgtime_re.plot(ax=axs_nov_re_logavg[1, 0], kind='kde', title='Nov_LogAvgTime_RE', color='#3f51b5')
    nov_avgdist_re.plot(ax=axs_nov_re_logavg[1, 1], kind='kde', title='Nov_AvgDist_RE', color='#3f51b5')

    fig_nov_re_mmsum, axs_nov_re_mmsum = plt.subplots(2, 2, dpi=200, constrained_layout=True)
    nov_mmsumpopplays_re.plot(ax=axs_nov_re_mmsum[0, 0], kind='kde', title='Nov_MMSumPopPlays_RE', color='#3f51b5')
    nov_mmsumpoplikes_re.plot(ax=axs_nov_re_mmsum[0, 1], kind='kde', title='Nov_MMSumPopLikes_RE', color='#3f51b5')
    nov_mmsumtime_re.plot(ax=axs_nov_re_mmsum[1, 0], kind='kde', title='Nov_MMSumTime_RE', color='#3f51b5')
    nov_sumdist_re.plot(ax=axs_nov_re_mmsum[1, 1], kind='kde', title='Nov_SumDist_RE', color='#3f51b5')

    fig_nov_re_logsum, axs_nov_re_logsum = plt.subplots(2, 2, dpi=200, constrained_layout=True)
    nov_logsumpopplays_re.plot(ax=axs_nov_re_logsum[0, 0], kind='kde', title='Nov_LogSumPopPlays_RE', color='#3f51b5')
    nov_logsumpoplikes_re.plot(ax=axs_nov_re_logsum[0, 1], kind='kde', title='Nov_LogSumPopLikes_RE', color='#3f51b5')
    nov_logsumtime_re.plot(ax=axs_nov_re_logsum[1, 0], kind='kde', title='Nov_LogSumTime_RE', color='#3f51b5')
    nov_sumdist_re.plot(ax=axs_nov_re_logsum[1, 1], kind='kde', title='Nov_SumDist_RE', color='#3f51b5')

    fig_nov_re_mmmed, axs_nov_re_mmmed = plt.subplots(2, 2, dpi=200, constrained_layout=True)
    nov_mmmedpopplays_re.plot(ax=axs_nov_re_mmmed[0, 0], kind='kde', title='Nov_MMMedPopPlays_RE', color='#3f51b5')
    nov_mmmedpoplikes_re.plot(ax=axs_nov_re_mmmed[0, 1], kind='kde', title='Nov_MMMedPopLikes_RE', color='#3f51b5')
    nov_mmmedtime_re.plot(ax=axs_nov_re_mmmed[1, 0], kind='kde', title='Nov_MMMedTime_RE', color='#3f51b5')
    nov_avgdist_re.plot(ax=axs_nov_re_mmmed[1, 1], kind='kde', title='Nov_AvgDist_RE', color='#3f51b5')

    fig_nov_re_logmed, axs_nov_re_logmed = plt.subplots(2, 2, dpi=200, constrained_layout=True)
    nov_logmedpopplays_re.plot(ax=axs_nov_re_logmed[0, 0], kind='kde', title='Nov_LogMedPopPlays_RE', color='#3f51b5')
    nov_logmedpoplikes_re.plot(ax=axs_nov_re_logmed[0, 1], kind='kde', title='Nov_LogMedPopLikes_RE', color='#3f51b5')
    nov_logmedtime_re.plot(ax=axs_nov_re_logmed[1, 0], kind='kde', title='Nov_LogMedTime_RE', color='#3f51b5')
    nov_avgdist_re.plot(ax=axs_nov_re_logmed[1, 1], kind='kde', title='Nov_AvgDist_RE', color='#3f51b5')

    #     show plots
    plt.show()
    return


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


def satisfaction(profile):
    return profile['Satisfy'].mean()


def serendipity(profile):
    return profile['Unknown_Satisfy'].mean()


def strict_serendipity(profile):
    return profile['strict_Unknown_Satisfy'].mean()


def entropy_based_diversity(profile):
    label_list = pd.concat([profile['Gen_Idx'], profile['Art_Idx']], axis=0, ignore_index=True, sort=True)
    value, counts = np.unique(list(label_list), return_counts=True)
    return entropy(counts, base=10)


def similarity_based_diversity(profile):
    # Keywords Count matrix
    vectorizer = CountVectorizer(lowercase=False)
    text_matrix = vectorizer.fit_transform(profile['Keywords'])
    text_matrix = pd.DataFrame(text_matrix.toarray(), columns=vectorizer.get_feature_names(), index=profile.index)
    profile_sim_list = cosine_distances(text_matrix)
    return profile_sim_list


def minmax_popularity_based_novelty(profile, min_pop, max_pop):
    return [profile.apply(lambda x: (max_pop - x) / (max_pop - min_pop))]


def log_popularity_based_novelty(profile):
    return [profile.apply(lambda x: -np.log10(x))]


def minmax_time_aware_novelty(profile, first_release, latest_release):
    return [profile.apply(lambda x: (x - first_release) / (latest_release - first_release))]


def log_time_aware_novelty(profile, first_release):
    return [profile.apply(lambda x: np.log10((x - first_release) / np.timedelta64(1, 'D')))]


def distance_based_novelty(profile, user_profile):
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
    return profile_sim_list


def long_tail_plots(candidate_likes):
    candidate_count = pd.DataFrame(candidate_likes.value_counts().to_numpy(), columns=['Likes'])
    candidate_count['Size'] = pd.cut(candidate_count['Likes'].to_numpy(), bins=10, labels=range(1, 11)).codes
    fig_candidate_count, axs_candidate_count = plt.subplots(dpi=200, constrained_layout=True)
    axs_candidate_count.scatter(
        candidate_count.index, candidate_count['Likes'].to_numpy(),
        c=candidate_count['Size'].to_numpy(), s=np.sqrt(candidate_count['Likes'].to_numpy())*3)
    return


if __name__ == "__main__":
    main()
