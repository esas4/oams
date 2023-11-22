from flask_script import Manager #从Flask Script中导入Manager模块
from flask_migrate import Migrate,MigrateCommand #导入Migrate的相关模块
from app import create_app #导入创建应用实例
from exts import db #导入db对象
from apps.admin import models as admin_models #导入需要迁移的数据库模型
app=create_app() #注册蓝图
manager=Manager(app) #初始化manager模块，让Python支持命令行工作
Migrate(app,db) #使用Migrate绑定app和db
manager.add_command('db',MigrateCommand) #添加迁移脚本的命令到manager中
#在终端使用命令，使用option装饰之后可以传递参数
@manager.option('-u','--username',dest='username')
@manager.option('-p','--password',dest='password')
@manager.option('-e','--email',dest='email')
#接受命令行参数username，password，email，将其作为user表中对应字段的内容
def create_user(username,password,email):
    user=admin_models.Admin_Users(username=username,password=password,email=email)
    db.session.add(user)
    db.session.commit()
    print("用户添加成功！")
#运行服务器
if __name__=='__main__':
    manager.run()
