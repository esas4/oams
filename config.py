import os

SECRET_KEY=os.urandom(24)

DEBUG=True #开启调试模式，显示更加详细的错误信息
ADMIN_USER_ID='HEBOANHEHE'
DB_USERNAME='root'#数据库登录账号用户名
DB_PASSWORD='090807'#数据库登录账号密码
DB_HOST='localhost'#数据库服务器地址
DB_PORT='3306'#数据库的端口
DB_NAME='oams_database'#数据库名称
DB_URI='mysql+pymysql://%s:%s@%s:%s/%s?charset=utf8' % (DB_USERNAME,DB_PASSWORD,DB_HOST,DB_PORT,DB_NAME) #指定数据库链接格式
#数据库链接格式：数据库+驱动：//用户名：密码@数据库主机地址：端口/数据库名称
SQLALCHEMY_DATABASE_URI=DB_URI#数据库URL必须保存到Flask配置对象的SQLALCHEMY_DATABASE_URI键中
SQLALCHEMY_TRACK_MODIFICATIONS=False#设置跟踪对象的修改，本oams中用不到调高运行效率，所以设置为False