from wtforms import Form #从wtforms组建中导入Form表单基类
from wtforms import StringField,BooleanField #导入用到的字段
from wtforms.validators import InputRequired,Length,Email #导入用到的验证器

class LoginForm(Form): #定义登录用表单验证类LoginForm类
    username=StringField(
        label='用户名',
        validators=[
            InputRequired('用户名为必填项'),
            Length(4,20,'用户名长度为4到20')
        ]
    )
    password=StringField(
        label='密码',
        validators=[
            InputRequired('密码为必填项'),
            Length(6,9,'密码长度为6到9')
        ]
    )