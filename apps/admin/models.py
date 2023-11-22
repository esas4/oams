from exts import db #导入db数据库链接对象
from datetime import datetime #导入时间函数
from werkzeug.security import generate_password_hash,check_password_hash #导入加密函数和对比函数
class Admin_Users(db.Model): #创建一名称为Admin_Users的类
    __tablename__='admin_users' #创建一名称为admin_users的表
    uid = db.Column(db.Integer, primary_key=True, autoincrement=True) #名称为uid的主键，整型，主键，自增长
    #gid=db.Column(db.Integer,nullable=True)
    username = db.Column(db.String(50), nullable=False)  # 用户名 username字段,varcha型，长度为50，不允许为空
    _password = db.Column(db.String(256), nullable=False)  # 密码 password字段，varcha型，长度为128，不允许为空
    email = db.Column(db.String(50), nullable=False, unique=True)  # email字段，varchar型，长度为50,不允许为空，键值唯一
    def __init__(self,username,password,email): #类的构造函数
        self.username=username
        self.password=password #此处使用到的password即为property将_password转化得到的
        self.email=email
        # print("password是"+self.password)
        # print("_password是"+self._password)
    ##获取密码
    @property #property装饰器用于将一个方法转化为属性，可以像访问属性一样访问该方法，而不需要使用函数调用的括号，从而提供更加友好的访问方式
    def password(self):
        return self._password
    ##设置密码
    @password.setter
    def password(self,raw_password):
        self._password=generate_password_hash(raw_password) #密码加密
    ##检查数据库中的密码与原始密码是否一样
    def check_password(self,raw_password):
        result=check_password_hash(self.password,raw_password)
        return result