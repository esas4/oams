from flask import Blueprint#从flask导入blueprint模块

#设置蓝图的相关信息
bp=Blueprint("common",__name__)#创建蓝图对象，必须指定两个参数。bp是蓝图的名称，__name__表示蓝图所在模块
@bp.route("/common")#定义蓝图路由
def index():#定义视图函数
    return "这是公共部分首页！"#返回响应