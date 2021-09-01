# __________ crawler 2.0

import pathlib
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import math
import time

# _global mask crawler
headers_global = {
    "Referer": "https://__________.com/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/77.0.3865.120 Safari/537.36"
}


# project file path management
def path_manage(file_stamp, file_type, op_mode, save_mode):
    path_dict = {
        'song_url': {
            'r': [pathlib.Path.cwd().joinpath("URL List").glob("{0}*_song_url.csv".format(file_stamp))],
            'w': [pathlib.Path.cwd().joinpath("URL List", "{0}_song_url.csv".format(file_stamp))]
        },
        'who_like_list': {
            'r': [pathlib.Path.cwd().joinpath("URL List").glob("{0}*_who_like_list_full.csv".format(file_stamp)),
                  pathlib.Path.cwd().joinpath("URL List").glob("{0}*_who_like_list_batch.csv".format(file_stamp))],
            'w': [pathlib.Path.cwd().joinpath("URL List", "{0}_who_like_list_full.csv".format(file_stamp)),
                  pathlib.Path.cwd().joinpath("URL List", "{0}_who_like_list_batch.csv".format(file_stamp))]
        },
        'target_user_list': {
            'r': [pathlib.Path.cwd().joinpath("URL List").glob("{0}*_target_user_list.csv".format(file_stamp))],
            'w': [pathlib.Path.cwd().joinpath("URL List", "{0}_target_user_list.csv".format(file_stamp))]
        },
        'song_profile': {
            'r': [pathlib.Path.cwd().joinpath("Song Information Profile").glob("{0}*_song_profile_full.csv"
                                                                               .format(file_stamp)),
                  pathlib.Path.cwd().joinpath("Song Information Profile").glob("{0}*_song_profile_batch.csv"
                                                                               .format(file_stamp))],
            'w': [pathlib.Path.cwd().joinpath("Song Information Profile", "{0}_song_profile_full.csv"
                                              .format(file_stamp)),
                  pathlib.Path.cwd().joinpath("Song Information Profile", "{0}_song_profile_batch.csv"
                                              .format(file_stamp))]
        },
        'candidate_user_profile': {
            'r': [pathlib.Path.cwd().joinpath("Candidate User Profile").glob("{0}*_can_user_profile.csv"
                                                                             .format(file_stamp))],
            'w': [pathlib.Path.cwd().joinpath("Candidate User Profile", "{0}_can_user_profile.csv"
                                              .format(file_stamp))]
        },
        'target_user_profile': {
            'r': [pathlib.Path.cwd().joinpath("Target User Profile").glob("{0}*_tar_user_profile.csv"
                                                                          .format(file_stamp))],
            'w': [pathlib.Path.cwd().joinpath("Target User Profile", "{0}_tar_user_profile.csv"
                                              .format(file_stamp))]
        },
        'recommendation_list': {
            'r': [pathlib.Path.cwd().joinpath("Recommendation List").glob("{0}*_rec_list.csv"
                                                                          .format(file_stamp))],
            'w': [pathlib.Path.cwd().joinpath("Recommendation List", "{0}_rec_list.csv"
                                              .format(file_stamp))]
        },
    }
    return path_dict[file_type][op_mode][save_mode]


def main():
    print("------- __________ crawler 2.0 -------")
    function_activate = input(
        "1: get list of Songs from Browse Music\n"
        "   output: Song Url['Browse_Genre', 'URL']\n"
        "2: get Song Information Profile\n"
        "   input:  (ALL_SONG_LIST) Song Url['URL']=/['Artist'](optional)/songs/['Song_ID']/\n"
        "   output: Song Profile['Song_ID', 'Title', 'Artist', 'Genre', 'Release_Date', 'Plays_count', 'Likes_count']\n"
        "3: get list of Users who Like the song\n"
        "   input:  Song Url['URL']=/['Artist'](optional)/songs/['Song_ID']\n"
        "   output: Who Like List['Song_ID', 'User_ID']\n"
        "4: get User Profile\n"
        "   input:  (SAMPLE_CRAWLER) Who Like List['User_ID'] / Target User List['User_ID']\n"
        "   output: User Profile['User_ID', 'Song_ID']\n"
        "5: Sample Candidate User Profile\n"
        "   input:  (ALL) Who Like List['User_ID', 'Song_ID']\n"
        "           Candidate User Profile['User_ID', 'Song_ID']\n"
        "   output: (SAMPLE_CRAWLER) Who Like List['User_ID']\n"
        "           (ALL_USER_PROFILE) Candidate User Profile['User_ID', 'Song_ID', 'Count'])"
        " & (ALL_SONG_LIST) Song Url['Song_ID', 'URL']\n"
        "6: check Song Information Profile\n"
        "   input:  (Target User Profile['Song_ID'] U Candidate User Profile['Song_ID'])"
        " - Song Information Profile['Song_ID']\n"
        "   output: (ADD) Song Profile['URL']=/songs/['Song_ID']/\n"
        "7: concat data\n"
        "   input:  data file dir\n"
        "   output: concat(data file)\n"
        "please input command>>")

    # 1: get list of Songs from Browse Music
    if function_activate == '1':
        print("{0} procedure is preparing...".format("get list of Songs from Browse Music"))

        # Sort Mode
        sort_mode = input("Which Sort mode want to activate?\n"
                          "most_liked: 最多喜歡\nmost_play: 最多播放\nplease input sort mode>>")

        # Genre Mode
        genre_code_dict = {
            0: '全部類型', 1: 'Rock', 2: 'Hip hop / Rap', 3: 'Electronic', 4: 'Pop', 5: 'Folk', 6: 'Singer / Songwriter',
            7: 'Alternative', 8: 'Post rock', 9: 'Metal', 10: 'Punk', 11: 'Reggae / Funk', 12: 'R&B / Soul',
            13: 'Classical', 14: 'Blues', 15: 'Jazz', 16: 'Soundtrack / New age', 17: 'World', 18: 'Religion',
            19: 'ACG', 20: 'Dance'
        }
        genre_code_df = pd.DataFrame.from_dict(genre_code_dict, orient='index', columns=['Music_Genre'])
        print(genre_code_df)
        print('Which Genre want to collect?')
        code_choice_start = input("please input Genre Code.\nrange From>>")
        code_choice_end = input("please input Genre Code.\nrange End>>")
        genre_code = [x for x in range(int(code_choice_start), int(code_choice_end) + 1)]
        print(genre_code_df.iloc[int(code_choice_start): int(code_choice_end) + 1])

        # require Collecting Songs Count
        if len(genre_code) > 1:
            print("Genre Mode: Collect Multiple Genre")
            collect_songs_count = input("How many songs require collecting per music genre?\nplease input integer>>")
        else:
            print("Genre Mode: Collect Single Genre")
            collect_songs_count = input("How many songs require collecting?\nplease input integer>>")

        # output filename stamp
        output_file_stamp = input("please type in output file [filename stamp]>>")

        get_song_list(sort_mode, genre_code, collect_songs_count, genre_code_df, output_file_stamp)

    # 2: get Song Information Profile
    elif function_activate == '2':
        input_file_stamp = input(
            "{0} procedure is preparing...\n"
            "input: [filename stamp]_song_url.csv is preparing to load...\n"
            "please type in input file [filename stamp]>>".format("get Song Information Profile"))

        # read & concat all song_url csv table
        input_song_url = concat_data(input_file_stamp, 'song_url')

        # output filename stamp
        output_file_stamp = input("please type in output file [filename stamp]>>")

        get_song_profile(input_song_url, output_file_stamp)

    # 3: get list of Users who Like the song
    elif function_activate == '3':
        input_file_stamp = input(
            "{0} procedure is preparing...\n"
            "input: [filename stamp]_song_url.csv is preparing to load...\n"
            "please type in input file [filename stamp]>>".format("get list of Users who Like"))

        # read & concat all song_url csv table
        input_song_url = concat_data(input_file_stamp, 'song_url')

        # output filename stamp
        output_file_stamp = input("please type in output file [filename stamp]>>")

        get_song_who_like(input_song_url, output_file_stamp)

    # 4: get User Profile
    elif function_activate == '4':
        file_type = input("{0} procedure is preparing...\n"
                          "who_like_list: read [file name]_who_like_list_full.csv\n"
                          "target_user_list: read [file name]_target_user_list.csv\n"
                          "please input [file type]>>".format("get User Profile"))
        input_file_stamp = input("input: [filename stamp]_{0}.csv is preparing to load...\n"
                                 "please type in input file [filename stamp]>>".format(file_type))

        # read & concat all who_like_list / target_user_list csv table
        input_user_list = concat_data(input_file_stamp, file_type)

        output_file_type = ""
        if file_type == 'who_like_list':
            output_file_type = 'candidate_user_profile'
        elif file_type == 'target_user_list':
            output_file_type = 'target_user_profile'

        # constraint number of songs in User likes(User Profile)
        constraint_user_profile = input(
            "User Profile Collecting Constraint: How many Songs does each User Profile need to contain?\n"
            "please input integer (default suggestion as 20 songs)>>")

        # output filename stamp
        output_file_stamp = input("please type in output file [filename stamp]>>")

        get_user_profile(input_user_list, constraint_user_profile, output_file_type, output_file_stamp)

    # 5: Sample Candidate User Profile
    elif function_activate == '5':
        activate_create_sample_list = input(
            "{0} procedure is preparing...\n"
            "Create Sample List for crawler by type in: {1}\n"
            "or Press any key to continue Sampling Candidate User Profile>>".format(
                "Sample Candidate User Profile", "create"))

        if activate_create_sample_list == 'create':
            # sample list for crawler
            sample_list_size = input(
                "Sample List Size: How many User Profiles are prepared to Crawl?\n"
                "please input integer (default suggestion as 20000 User Profiles)>>")
            # output filename stamp
            output_file_stamp = input("please type in output file [filename stamp]>>")

            who_like_list = concat_data("ALL_TOP_GENRE", 'who_like_list')
            sample_who_like_list = pd.DataFrame(
                pd.unique(who_like_list['User_ID']), columns=['User_ID']).sample(n=int(sample_list_size))
            sample_who_like_list.to_csv(
                path_manage('{0}_SAMPLE_CRAWLER'.format(output_file_stamp), 'who_like_list', 'w', 0),
                header=True, mode='a', sep=',', index=True, encoding='utf8')
            print("\nsample_who_like_list:\n", sample_who_like_list)
            main()

        # sample size of Candidate User Profiles
        sample_size = input(
            "Sample Size: How many User Profiles need to include in Sampling Set?\n"
            "please input integer (default suggestion as 2000 User Profiles)>>")

        # constraint number of songs in User Profile
        constraint_user_profile = input(
            "User Profile Sampling Constraint: How many Songs does each User Profile need to contain?\n"
            "please input integer (default suggestion as 20 songs)>>")

        # read all candidate_user_profile csv table
        candidate_user_profile = concat_data("", 'candidate_user_profile')

        sample_user_profile = sample_candidate_user(candidate_user_profile, sample_size, constraint_user_profile)
        sample_user_profile.to_csv(path_manage("ALL_USER_PROFILE", 'candidate_user_profile', 'w', 0),
                                   header=True, mode='a', sep=',', index=True, encoding='utf8')

        all_song_list = check_song_profile(pd.DataFrame(), sample_user_profile, pd.DataFrame())
        all_song_list.to_csv(path_manage("ALL_SONG_LIST", 'song_url', 'w', 0),
                             header=True, mode='a', sep=',', index=True, encoding='utf8')

    # 6: check Song Information Profile
    elif function_activate == '6':
        input_file_stamp = input(
            "{0} procedure is preparing...\n"
            "input: [filename stamp]_song_profile_full.csv is preparing to load...\n"
            "please type in input file [filename stamp]>>".format("check Song Information Profile"))

        # read & concat all song_profile csv table
        input_song_profile = concat_data(input_file_stamp, 'song_profile')

        # read & concat all candidate_user_profile csv table
        candidate_user_profile = concat_data("ALL_USER_PROFILE", 'candidate_user_profile')

        # output filename stamp
        output_file_stamp = input("please type in output file [filename stamp]>>")

        # read & concat all target_user_profile csv table
        for target_user_profile_file in path_manage("", 'target_user_profile', 'r', 0):
            print(target_user_profile_file)
            user_id = target_user_profile_file.stem.replace("_tar_user_profile", "").split("_", 1)[1]

            # check add song profile file exist or not
            check_add_profile_exist = list(path_manage("*_{0}_add".format(user_id), 'song_profile', 'r', 0))
            if not check_add_profile_exist:
                target_user_profile = pd.read_csv(
                    target_user_profile_file, sep=',', encoding='utf8', dtype={'User_ID': str})

                get_song_profile(
                    check_song_profile(input_song_profile, candidate_user_profile, target_user_profile),
                    "{0}_{1}_{2}".format(output_file_stamp, user_id, "add"))
                print("get user:{0} additional song profile\n".format(user_id))

    # 7: concat data
    elif function_activate == '7':
        file_type = input("{0} procedure is preparing...\n"
                          "song_url: read [file name]_song_url.csv\n"
                          "who_like_list: read [file name]_who_like_list_full.csv\n"
                          "target_user_list: read [file name]_target_user_list.csv\n"
                          "song_profile: read [file name]_song_profile_full.csv\n"
                          "add_song_profile: read [file name]_add_song_profile_full.csv\n"
                          "candidate_user_profile: read [file name]_can_user_profile.csv\n"
                          "target_user_profile: read [file name]_tar_user_profile.csv\n"
                          "recommendation_list: read [file name]_rec_list.csv\n"
                          "please input [file type]>>".format("concat data"))
        input_file_stamp = input("input: [filename stamp]_{0}.csv is preparing to load...\n"
                                 "please type in input file [filename stamp]>>".format(file_type))

        # output filename stamp
        output_file_stamp = input("please type in output file [filename stamp]>>")

        concat_data(input_file_stamp, file_type).to_csv(path_manage(output_file_stamp, file_type, 'w', 0),
                                                        header=True, mode='a', sep=',', index=False, encoding='utf8')


# 1: get list of Songs from Browse Music
# __________ > 發現音樂 > 所有音樂 & 不限時間 > 20 songs/page
def get_song_list(sort_mode, genre_code, collect_songs_count, genre_code_df, output_file_stamp):
    # variable initialization
    count_songs = 1

    # list of Songs URL & Genre DataFrame
    song_url_df = pd.DataFrame(columns=['Browse_Genre', 'URL'])

    # collect Browse Music URL & check songs count: 20 songs/page
    url_browse_list = []
    for code in genre_code:
        url_browse = "https://__________.com/music/browse/{0}/all/{1}/1/".format(str(code), sort_mode)
        url_browse_list.append(url_browse)
        if int(collect_songs_count) > 20:

            # split 20 songs/page
            pages_browse = int(collect_songs_count) / 20
            for page in range(2, math.ceil(pages_browse) + 1):
                url_browse = "https://__________.com/music/browse/{0}/all/{1}/1/?page={2}".format(
                    str(code), sort_mode, str(page))
                url_browse_list.append(url_browse)

    # read Browse Music URL
    for url in url_browse_list:
        html_browse = requests.get(url, headers=headers_global)
        soup_browse = BeautifulSoup(html_browse.text, 'lxml')

        # get song Browse_Genre
        url_split = int(url.split("/")[5])
        browse_genre = genre_code_df['Music_Genre'][url_split]

        # get song URL
        page_limit_count = 1
        for song in soup_browse.find_all('div', {'class': 'cover-block with-button img-lg mr-3'}):
            # limit in require songs count and limit in 20 songs per page
            if count_songs < int(collect_songs_count) * int(len(genre_code)) + 1 \
                    and page_limit_count < int(collect_songs_count) + 1:
                song_url = song.find('a')['href']

                # write in DataFrame
                row = pd.DataFrame({'Browse_Genre': [browse_genre], 'URL': [song_url]})
                song_url_df = song_url_df.append(row, ignore_index=True, sort=False)

                # Counter
                print("第 <{2}> 首歌曲之 URL: <{1}> 擷取完成！ from Browse_Genre: <{0}>"
                      .format(browse_genre, song_url, count_songs))
                count_songs += 1
                page_limit_count += 1

    # DataFrame to CSV
    song_url_df.to_csv(path_manage("{0}_{1}_{2}".format(output_file_stamp, sort_mode, collect_songs_count),
                                   'song_url', 'w', 0), header=True, mode='a', sep=',', index=False, encoding='utf8')
    return


# 2: get Song Information Profile
# __________ > 歌曲資訊
def get_song_profile(input_song_url, output_file_stamp):
    # variable initialization
    count_songs = 1

    # Song Information Profile DataFrame
    song_profile_df = pd.DataFrame(columns=['Song_ID', 'Title', 'Artist', 'Genre', 'Release_Date',
                                            'Plays_count', 'Likes_count'])

    # batch ver. DataFrame header
    song_profile_df.to_csv(path_manage(output_file_stamp, 'song_profile', 'w', 1),
                           header=True, mode='a', sep=',', index=False, encoding='utf8')

    # read songs URL List
    for index, rows in input_song_url.iterrows():
        url_song = "https://__________.com{0}".format(rows['URL'])

        # read song URL
        html_song = requests.get(url_song, headers=headers_global)
        soup_song = BeautifulSoup(html_song.text, 'lxml')

        # song authorization check
        if soup_song.find('button', {'class': 'btn btn-play btn-circle btn-primary btn-xl'}) is not None:

            # get song ID
            song_id = soup_song.find('button', {'class': 'btn btn-play btn-circle btn-primary btn-xl'})['data-id']

            # var DjangoPrerender
            sv_django_prerender = soup_song.find('script', string=re.compile("DjangoPrerender")).text
            # get song Plays_count
            sv_play_count = re.search(re.compile(r"play_count: \'.*.\'"), sv_django_prerender)
            song_plays = sv_play_count.group().split()[1][1:-1]
            # get song Likes_count
            sv_like_count = re.search(re.compile(r"like_count: \'.*.\'"), sv_django_prerender)
            song_likes = sv_like_count.group().split()[1][1:-1]

            # get song title
            song_title = soup_song.find('h1', {'class': 'text-white'}).text
            # get song artist
            song_artist = soup_song.find('h3', {'class': 'mb-0 text-break'}).text
            # get song genre
            song_genre = soup_song.find('h3', {'class': 'text-truncate text-white opacity-72'}).text.split(" ・ ")[0]
            # get song date
            song_date = soup_song.find('p', {'class': 'text-gray-light mb-2'}).text.split()[1]

            # write in DataFrame & batch ver. DataFrame to CSV
            row = pd.DataFrame(
                {'Song_ID': [song_id], 'Title': [song_title], 'Artist': [song_artist], 'Genre': [song_genre],
                 'Release_Date': [song_date], 'Plays_count': [song_plays], 'Likes_count': [song_likes]})
            row.to_csv(path_manage(output_file_stamp, 'song_profile', 'w', 1),
                       header=False, mode='a', sep=',', index=False, encoding='utf8')
            song_profile_df = song_profile_df.append(row, ignore_index=True, sort=False)

            # Counter
            print("   Song_ID: <{0}>, Title: <{1}>, Artist: <{2}>, \n   Genre: <{3}>, Release_Date: <{4}>, "
                  "Plays_count: <{5}>, Likes_count: <{6}>\n第 <{7}> 首歌曲之 Song Profile 擷取完成！"
                  .format(song_id, song_title, song_artist, song_genre, song_date, song_plays, song_likes, count_songs))
            count_songs += 1

        else:
            print("Song <{0}> is not open to the public".format(rows['URL']))

    # full ver. DataFrame to CSV
    song_profile_df.to_csv(path_manage(output_file_stamp, 'song_profile', 'w', 0),
                           header=True, mode='a', sep=',', index=False, encoding='utf8')
    return


# 3: get list of Users who Like the song
# __________ > 歌曲資訊 > 誰喜歡
def get_song_who_like(input_song_url, output_file_stamp):
    # variable initialization
    count_songs = 1

    # read songs URL List
    for index, rows in input_song_url.iterrows():
        url_song = "https://__________.com{0}".format(rows['URL'])

        # read song information URL
        html_song = requests.get(url_song, headers=headers_global)
        soup_song = BeautifulSoup(html_song.text, 'lxml')

        # get X-CSRFToken to access ______who_like
        csrf_token = soup_song.find('input', {'name': 'csrfmiddlewaretoken'})['value']

        # get song ID & genre
        song_id = soup_song.find('button', {'class': 'btn btn-play btn-circle btn-primary btn-xl'})['data-id']
        song_genre = soup_song.find(
            'h3', {'class': 'text-truncate text-white opacity-72'}).text.split(" ・ ")[0].replace("/", "&")

        # var DjangoPrerender
        sv_django_prerender = soup_song.find('script', string=re.compile("DjangoPrerender")).text
        # get song Likes_count
        sv_like_count = re.search(re.compile(r"like_count: \'.*.\'"), sv_django_prerender)
        song_likes = sv_like_count.group().split()[1][1:-1]

        # variable initialization
        count_users = 1

        # list of Users who Like the song DataFrame
        who_like_list_df = pd.DataFrame(columns=['Song_ID', 'User_ID'])

        # batch ver. DataFrame header
        who_like_list_df.to_csv(
            path_manage("{0}_{1}{2}".format(output_file_stamp, song_genre, song_id), 'who_like_list', 'w', 1),
            header=True, mode='a', sep=',', index=False, encoding='utf8')

        # ______ mask crawler
        headers______ = {
            "Accept": "text/html, */*; q=0.01",
            "Accept-Encoding": "gzip, deflate, sdch",
            "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6",
            "Referer": url_song,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/77.0.3865.120 Safari/537.36",
            "X-CSRFToken": csrf_token,
            "X-Requested-With": "XMLHttpRequest"
        }

        # collect song who_like URL
        url_who_like_list = []
        url_who_like = "https://__________.com/_____/music.song/{0}/who_like/".format(song_id)
        url_who_like_list.append(url_who_like)

        # split 10 users/page
        pages_who_like = int(song_likes) / 10
        for page in range(2, math.ceil(pages_who_like) + 1):
            url_who_like = "https://__________.com/_____/music.song/{0}/who_like/?page={1}".format(song_id, str(page))
            url_who_like_list.append(url_who_like)

        # read song who_like URL
        for url in url_who_like_list:
            html_song_who_like = requests.get(url, headers=headers______)
            soup_song_who_like = BeautifulSoup(html_song_who_like.text, 'lxml')
            for each_user in soup_song_who_like.find_all(
                    'li', {'class': 'list-group-item align-items-center justify-content-between work-item item_box'}):
                # get user ID
                user_id = each_user.find('a')['href'].split("/")[1]

                # write in DataFrame & batch ver. DataFrame to CSV
                row = pd.DataFrame({'Song_ID': [song_id], 'User_ID': [user_id]})
                row.to_csv(
                    path_manage("{0}_{1}{2}".format(output_file_stamp, song_genre, song_id), 'who_like_list', 'w', 1),
                    header=False, mode='a', sep=',', index=False, encoding='utf8')
                who_like_list_df = who_like_list_df.append(row, ignore_index=True, sort=False)

                # Counter
                print("第 <{2}> 位使用者 User_ID: <{1}> 擷取完成！ from Song_ID: <{0}>".format(song_id, user_id, count_users))
                count_users += 1
            time.sleep(0.2)

        # Counter
        print("第 <{1}> 首歌曲 Song_ID: <{0}> 之 Who Like 擷取完成！".format(song_id, count_songs))
        count_songs += 1

        # full ver. DataFrame to CSV
        who_like_list_df.to_csv(
            path_manage("{0}_{1}{2}".format(output_file_stamp, song_genre, song_id), 'who_like_list', 'w', 0),
            header=True, mode='a', sep=',', index=False, encoding='utf8')
    return


# 4: get User Profile
# __________ > 使用者個人頁面 > 喜歡歌單(使用者興趣檔)
def get_user_profile(input_user_list, constraint_user_profile, output_file_type, output_file_stamp):
    # variable initialization
    count_users = 1

    # read User ID list
    for index, rows in input_user_list.iterrows():
        user_id = rows['User_ID']

        # collect user likes URL
        url_user_list = []
        url_user = "https://__________.com/{0}/likes/".format(user_id)
        url_user_list.append(url_user)

        # _user_likes mask crawler
        headers_user_likes = {
            "Referer": url_user,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/77.0.3865.120 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest"
        }

        html_user = requests.get(url_user, headers=headers_user_likes)
        soup_user = BeautifulSoup(html_user.text, 'lxml')

        # number of songs in user likes(User Profile)
        if soup_user.find('h2', {'class': 'mb-3'}, string=re.compile("首歌曲")) is not None:
            profile_song_number = soup_user.find('h2', {'class': 'mb-3'}, string=re.compile("首歌曲")).text.split()[0]
            if int(profile_song_number) < int(constraint_user_profile):
                pass
            else:
                # variable initialization
                count_likes = 1

                # list of Songs in user likes(User Profile) DataFrame
                user_profile_df = pd.DataFrame(columns=['User_ID', 'Song_ID'])

                # split 8 songs/page
                pages_profile = int(profile_song_number) / 8
                for page in range(2, math.ceil(pages_profile) + 1):
                    url_user = "https://__________.com/{0}/likes/?page={1}".format(user_id, str(page))
                    url_user_list.append(url_user)

                # read user likes URL
                for url in url_user_list:
                    html_user = requests.get(url, headers=headers_user_likes)
                    soup_user = BeautifulSoup(html_user.text, 'lxml')
                    for each_song in soup_user.find_all(
                            'button', {'class': 'btn btn-play btn-circle btn-outline-white btn-xl js-likes'}):
                        # get song ID
                        song_id = each_song['data-id']

                        # write in DataFrame
                        row = pd.DataFrame({'User_ID': [user_id], 'Song_ID': [song_id]})
                        user_profile_df = user_profile_df.append(row, ignore_index=True, sort=False)

                        # Counter
                        print("第 <{2}> 首喜歡歌曲 Song_ID: <{1}> 擷取完成！ from User_ID: <{0}>"
                              .format(user_id, song_id, count_likes))
                        count_likes += 1

                # Counter
                print("第 <{1}> 位使用者 User_ID: <{0}> 之 User Profile 擷取完成！".format(user_id, count_users))
                count_users += 1
                time.sleep(0.2)

                # DataFrame to CSV
                user_profile_df.to_csv(
                    path_manage("{0}_{1}".format(output_file_stamp, user_id), output_file_type, 'w', 0),
                    header=True, mode='w', sep=',', index=False, encoding='utf8')
    return


# 5: Sample Candidate User Profile
def sample_candidate_user(candidate_user_profile, sample_size, constraint_user_profile):
    candidate_user_profile.set_index(['User_ID'], inplace=True)

    # count songs of each User Profile
    candidate_user_profile['Count'] = candidate_user_profile.groupby(['User_ID']).transform('count')

    # constraint number of songs in User Profile
    sample_user_profile = candidate_user_profile[candidate_user_profile['Count'] > int(constraint_user_profile) - 1]

    # User Profile sampling
    sample_user_list = pd.DataFrame(pd.unique(sample_user_profile.index), columns=['User_ID']).sample(
        n=int(sample_size))
    sample_user_profile = sample_user_profile.loc[sample_user_list['User_ID'], :]
    sample_user_profile.sort_values(by='Count', ascending=False, inplace=True)
    print(sample_user_profile)

    return sample_user_profile


# 6: check Song Information Profile
# __________ > 歌曲資訊
def check_song_profile(input_song_profile, candidate_user_profile, target_user_profile):
    if not input_song_profile.empty:
        # Target User Profile['Song_ID'] U Candidate User Profile['Song_ID']
        all_user_profile = pd.concat([target_user_profile, candidate_user_profile], axis=0, join='outer', sort=False)
        all_user_profile_song_list = pd.DataFrame(sorted(pd.unique(all_user_profile['Song_ID'])), columns=['Song_ID'])

        # All User Profile['Song_ID'] - Song Information Profile['Song_ID']
        add_song_profile = pd.DataFrame(
            pd.Index(all_user_profile_song_list['Song_ID']).difference(
                pd.Index(input_song_profile['Song_ID']), sort=False), columns=['Song_ID'])
        add_song_profile['URL'] = add_song_profile['Song_ID'].map(lambda x: r"/songs/{0}/".format(x))

    # while song_profile_full.csv is not exist
    else:
        add_song_profile = pd.DataFrame(pd.unique(candidate_user_profile['Song_ID']), columns=['Song_ID'])
        add_song_profile['URL'] = add_song_profile['Song_ID'].map(lambda x: r"/songs/{0}/".format(x))

    # random index
    add_song_profile = add_song_profile.sample(n=add_song_profile.shape[0], replace=False)
    print(add_song_profile)

    return add_song_profile


# 7: concat data
def concat_data(input_file_stamp, file_type):
    # check if file path exists or not
    if list(path_manage(input_file_stamp, file_type, 'r', 0)):
        print("{0}:\n{1}".format(file_type, "\n".join(map(str, path_manage(input_file_stamp, file_type, 'r', 0)))))
        # read & concat all csv table
        data_append = []
        for file in path_manage(input_file_stamp, file_type, 'r', 0):
            data_file = pd.read_csv(file, sep=',', encoding='utf8', dtype={'User_ID': str})
            data_append.append(data_file)
        data_append = pd.concat(data_append, sort=False)
    else:
        print("{0} csv:\n{1}".format(file_type, "Not Found"))
        data_append = pd.DataFrame()

    return data_append


if __name__ == '__main__':
    main()
