from django.shortcuts import render, redirect, HttpResponse
from django.urls import reverse

from SongEvalWeb.settings import BASE_DIR
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup


# top-n of recommendation list will present
top_n = 30

# user desktop location
# desktop_location = os.path.join(os.environ["HOMEDRIVE"], os.environ["HOMEPATH"], "Desktop")
# desktop_location = BASE_DIR

# read User list DataFrame
user_list = pd.read_csv(os.path.join(BASE_DIR, 'static', 'user_list.csv'),
                        sep=',', encoding='utf8', dtype={'User_ID': str, 'User_PSW': str})
user_list.set_index('User_ID', inplace=True)

# read Recommendation list DataFrame
rec_list = pd.read_csv(os.path.join(BASE_DIR, 'static', 'rec_list.csv'),
                       sep=',', encoding='utf8', dtype={'User_ID': str})
rec_list.set_index(['User_ID', 'Strategy_Type'], inplace=True)
rec_list.sort_index(inplace=True)

# read Google Form id DataFrame
form_id = pd.read_csv(os.path.join(BASE_DIR, 'static', 'googleform_id.csv'), sep=',', encoding='utf8')
known_val = form_id['known_val']
satisfy_val = form_id['satisfy_val']
pref_val = form_id['pref_val']

# read Scale question DataFrame
scale_list = pd.read_csv(os.path.join(BASE_DIR, 'static', 'scale_list.csv'), sep=',', encoding='utf8')

# Google Form headers
headers_form_strategy = {
    "Referer": "https://docs.google.com/forms/d/e/__________/viewform",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/76.0.3809.100 Safari/537.36"
}
headers_form_scale = {
    "Referer": "https://docs.google.com/forms/d/e/__________/viewform",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/76.0.3809.100 Safari/537.36"
}


def get_index(request):
    title_nav = '歡迎您參與本研究'
    return render(request, 'index.html', locals())


def post_login(request):
    user_id = request.POST.get('user_id')
    user_psw = request.POST.get('user_psw')
    if (user_list.loc[user_list.index == user_id, 'User_PSW'] == user_psw).any():
        return redirect(reverse('get_strategy', kwargs={'user_id': user_id, 'strategy_page': 1}), locals())
    else:
        return render(request, 'index.html', locals())


def get_strategy(request, user_id, strategy_page):
    if strategy_page < 4:
        title_nav = '第一階段 - {0}'.format(strategy_page)
        # user authorization
        user_auth = True

        # get recommendation list of user_id
        strategy_type = user_list.loc[user_id, 'strategy_0{0}'.format(strategy_page)]
        song_list = rec_list.loc[(user_id, strategy_type), 'Song_ID'][:top_n]
        # random recommendation list by df.sample()
        song_list = song_list.sample(n=song_list.shape[0], replace=False, random_state=99)
        song_list = list(song_list.to_numpy())
        # locals()
        data = {
            'rec_type': strategy_type,
            'form_data': zip(song_list, known_val, satisfy_val),
        }
        return render(request, 'strategy.html', locals())
    else:
        return redirect(reverse('get_scale', kwargs={'user_id': user_id}), locals())


def post_strategy(request, user_id, strategy_page):
    url_google_form = \
        "https://docs.google.com/forms/d/e/__________/formResponse"
    google_form_submit(request, user_id, strategy_page, url_google_form, headers_form_strategy)

    # next page count
    strategy_page += 1
    return redirect(reverse('get_strategy', kwargs={'user_id': user_id, 'strategy_page': strategy_page}), locals())


def get_scale(request, user_id):
    title_nav = '第二階段'
    user_auth = True

    # random Scale question by df.sample()
    scale_list_random = scale_list.sample(n=scale_list.shape[0], replace=False)
    pref_id = list(scale_list_random['Pref_ID'].to_numpy())
    scale_question = list(scale_list_random['Question'].to_numpy())
    # locals()
    data = {
        'form_data': zip(pref_id, scale_question, pref_val),
    }
    return render(request, 'scale.html', locals())


def post_scale(request, user_id):
    user_auth = True
    url_google_form = \
        "https://docs.google.com/forms/d/e/__________/formResponse"
    google_form_submit(request, user_id, 'scale', url_google_form, headers_form_strategy)
    return redirect(reverse('get_done', kwargs={'user_id': user_id}), locals())


def get_done(request, user_id):
    title_nav = '本實驗結束，誠摯感謝您的參與'
    user_auth = True
    return render(request, 'done.html', locals())


def google_form_submit(request, file_stamp1, file_stamp2, url_google_form, headers_google_form):
    # save to desktop
    form_value = pd.DataFrame(request.POST, index=[0])
    # form_value.to_csv(os.path.join(desktop_location, '{0}_{1}.csv'.format(file_stamp1, file_stamp2)), header = True, mode = 'a', sep = ',', index = False, encoding = 'utf8')
    # response = HttpResponse(form_value, content_type='text/csv')
    # response['Content-Disposition'] = 'attachment; filename={0}_{1}.txt'.format(file_stamp1, file_stamp2)
    # writer = csv.writer(response)
    # writer.writerow([form_value.to_csv(header=True, mode='a', sep=',', index=False, encoding='utf8')])

    # post to Google Form
    html_google_form = requests.get(url_google_form, headers=headers_google_form)
    soup_google_form = BeautifulSoup(html_google_form.text, 'lxml')
    tag_fbzx = soup_google_form.find('input', {'name': 'fbzx'})['value']
    form_value.drop(['csrfmiddlewaretoken'], axis=1, inplace=True)
    form_value['fvv'] = "1"
    form_value['draftResponse'] = str("[null,null,'{0}']").format(tag_fbzx)
    form_value['pageHistory'] = "0"
    form_value['fbzx'] = tag_fbzx
    form_value = form_value.T.to_dict()[0]
    requests.post(url_google_form, data=form_value, headers=headers_google_form)
    return


# mobile version
def get_mobile_index(request):
    title_nav = '歡迎您參與本研究'
    return render(request, 'mobile_index.html', locals())


def post_mobile_login(request):
    user_id = request.POST.get('user_id')
    user_psw = request.POST.get('user_psw')
    if (user_list.loc[user_list.index == user_id, 'User_PSW'] == user_psw).any():
        return redirect(reverse('get_mobile_strategy', kwargs={'user_id': user_id, 'strategy_page': 1}), locals())
    else:
        return render(request, 'mobile_index.html', locals())


def get_mobile_strategy(request, user_id, strategy_page):
    if strategy_page < 4:
        title_nav = '第一階段 - {0}'.format(strategy_page)
        # user authorization
        user_auth = True

        # get recommendation list of user_id
        strategy_type = user_list.loc[user_id, 'strategy_0{0}'.format(strategy_page)]
        song_list = rec_list.loc[(user_id, strategy_type), 'Song_ID'][:top_n]
        # random recommendation list by df.sample()
        song_list = song_list.sample(n=song_list.shape[0], replace=False, random_state=99)
        song_list = list(song_list.to_numpy())
        # locals()
        data = {
            'song_data': song_list,
        }
        # next page count
        strategy_page += 1
        return render(request, 'mobile_strategy.html', locals())
    else:
        return redirect(reverse('get_mobile_done', kwargs={'user_id': user_id}), locals())


def get_mobile_done(request, user_id):
    title_nav = '請繼續於電腦填答問卷'
    user_auth = True
    return render(request, 'mobile_done.html', locals())