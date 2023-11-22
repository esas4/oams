from flask import Blueprint,render_template,request,session,redirect,url_for

import config
#从flask导入blueprint模块，用于创建蓝图对象，以组织和管理应用的路由和视图
#render_template函数用于实现渲染模板，通常用于将动态数据嵌入到html页面中
#request允许访问客户端发出的http请求信息
#session用于在应用中存储和管理会话数据，例如用户的登录状态或其他会话相关信息
#redirect用于执行重定向操作，将用户导航到其他url或视图
#url_for用于生成url，通常与路由函数的名称和参数一起使用，以便在应用中动态构建url
from .models import Admin_Users
from .forms import LoginForm #从forms.py中导入好的用户登录验证类
from flask import make_response
from utils.captcha import create_validate_code
from io import BytesIO
from datetime import timedelta #从datetime模块导入timedelta,timedelta是datetime的一个对象，该对象表示两个时间的差值
from functools import wraps #admin网页不能直接访问，使用wraps自定义装饰器

#设置蓝图的相关信息
bp=Blueprint("admin",__name__,url_prefix='/admin')#创建蓝图对象bp，必须指定两个参数。admin是蓝图的名称，__name__表示蓝图所在模块,url_prefix='/admin'是蓝图的url前缀，定义由这个蓝图定义的基本路由为admin

@bp.route("/login/",methods=['GET','POST']) #定义登录的路由是'/login'
def login(): #定义登录处理函数login()
    error=None #声明出错信息
    if request.method=='GET':
        return render_template('admin/login.html')
    else:
        form=LoginForm(request.form) #将表单元素通过定义好的表单验证类LoginForm进行验证
        if form.validate(): #如果表单验证成功，则进行下一步的数据库用户名和密码的验证
            captcha=request.form.get('captcha') #获取用户输入的验证码
            if session.get('image').lower()!=captcha.lower():
                return render_template('admin/login.html',message="验证码错误！")
            user=request.form.get('username') #获取用户输入的用户名
            pwd=request.form.get('password') #获取用户输入的密码
            online=request.form.get('online') #接收checkbox按钮传过来的值
            users=Admin_Users.query.filter_by(username=user).first() #在数据库中查询是否有此用户
            if users: #确认用户输入的用户名和密码是否正确，如果正确就认为用户登录成功
                if user==users.username and users.check_password(pwd):
                    session[config.ADMIN_USER_ID]=users.uid #用户id存于session
                    #print(session['user_id'])
                    print("密码正确！")
                    # 如果选择了记住我
                    if online:
                        session.permanent=True
                        #bp.permanent_session_lifetime=timedelta(days=14)
                        bp.permanent_session_lifetime=timedelta(minutes=10)
                    return redirect(url_for('admin.index'))
                else:
                    #print("用户名或密码错误！")
                    error="用户名或密码错误！"
                    return render_template('admin/login.html',message=error)
            else: #提示用户不存在
                return render_template('admin/login.html',message="该用户不存在！")
        else:
            return render_template('admin/login.html',message=form.errors)

#调用验证码
@bp.route('/code/')
def get_code():
    #把strs发给前段，或者在后台使用session保存
    code_img,strs=create_validate_code()
    buf=BytesIO() #允许在内存中创建一个BytesIO类的对象buf，像操作文件一样进行读写操作
    code_img.save(buf,'JPEG',quality=70)
    buf_str=buf.getvalue()
    #buf.seek(0)
    response=make_response(buf_str) #使用make_response函数创建一个响应对象，并将buf_str中的内容作为响应的主体进行设置，使得其可以作为HTTP响应的一部分返回给客户端
    response.headers['Content-Type']='image/jpeg'
    #将验证码字符串储存在session中
    session['image']=strs
    return response

@bp.route('/logout/')
def logout():
    del session[config.ADMIN_USER_ID]
    return redirect(url_for('admin.login'))