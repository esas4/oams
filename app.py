DEBUG=True #在开发过程中进入调试模式，方便显示更多信息
           #不能在生产过程中使用，生产过程中应调为DEBUG=False

from flask import Flask
#导入各个模块的蓝图
from apps.admin import bp as admin_bp
from apps.common import bp as common_bp
from apps.front import bp as front_bp
#导入db数据库链接对象
from exts import db

# @app.route('/') #定义路由 #蓝图中的路由函数优于应用程序全局的视图函数
# def hello_world():  # put application's code here
#     return 'Hello World!'

def create_app():
    # 创建flask类的对象app（应用实例）。接收自客户端的所有请求都转交给这个对象处理
    app=Flask(__name__) #__name__是应用主模块或包的名称。flask用这个参数确定应用的位置，进而找到应用中其他文件的位置
    #注册蓝图
    app.register_blueprint(admin_bp)
    app.register_blueprint(front_bp)
    app.register_blueprint(common_bp)
    app.config.from_object('config') #调用在config中的配置
    # app.config['STATIC_FOLDER']='static'
    db.init_app(app)
    return app

if __name__ == '__main__':
    app=create_app()
    app.run(host='127.0.0.1',port=8000,debug=True)
