"""SongEvalWeb URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# from django.contrib import admin
from django.urls import path
from CoreApp.views import get_index, post_login, get_strategy, post_strategy, get_scale, post_scale, get_done, \
    get_mobile_index, post_mobile_login, get_mobile_strategy, get_mobile_done

urlpatterns = [
    # path('admin/', admin.site.urls),
    path('', get_index, name='index'),
    path('login/post/', post_login, name='post_login'),
    path('<str:user_id>/strategy/<int:strategy_page>', get_strategy, name='get_strategy'),
    path('<str:user_id>/strategy/<int:strategy_page>/post', post_strategy, name='post_strategy'),
    path('<str:user_id>/scale', get_scale, name='get_scale'),
    path('<str:user_id>/scale/post', post_scale, name='post_scale'),
    path('<str:user_id>/done', get_done, name='get_done'),
    # mobile version
    path('m', get_mobile_index, name='mobile_index'),
    path('m/login/post/', post_mobile_login, name='post_mobile_login'),
    path('m/<str:user_id>/strategy/<int:strategy_page>', get_mobile_strategy, name='get_mobile_strategy'),
    path('m/<str:user_id>/done', get_mobile_done, name='get_mobile_done'),
]
