from flask import Blueprint#从flask导入blueprint模块

#设置蓝图的相关信息
bp=Blueprint("front",__name__)#创建蓝图对象，必须指定两个参数。bp是蓝图的名称，__name__表示蓝图所在模块
#前台访问不需要前缀（首页）
@bp.route('/')#定义蓝图路由
def index():#定义视图函数
    return "这是前台首页！"#返回响应