from wtforms import Form #从wtforms组建中导入Form表单基类
from wtforms import StringField, BooleanField #导入要用到的字段
# from wtforms.validators import InputRequired, Length#导入要用到的验证器

class Test_Dataset_Path(Form):#定义接收数据集路径的表单类
    path=StringField(
        label='ImagePath',
    )