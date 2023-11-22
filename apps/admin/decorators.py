# encoding:utf-8
from functools import wraps
from flask import session,redirect,url_for,render_template
from .views import bp

import config

#登录限制装饰器
def login_required(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        if session.get(config.ADMIN_USER_ID):
            return func(*args,**kwargs)
        else:
            return redirect(url_for('admin.login'))
    return wrapper

#如果没有登录而直接访问管理员后台首页，会跳转到登录界面
@bp.route("/")
@login_required
def index():#定义视图函数
    return render_template('admin/index.html') #返回响应，其地址相对于tempaltes文件夹
