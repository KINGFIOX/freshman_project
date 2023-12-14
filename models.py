from exts import db
from datetime import datetime


# class UserModel(db.Model):
#     __tablename__ = "user_data"
#     # id、用户名、密码、创建时间
#     id = db.Column(db.Integer, primary_key=True, autoincrement=True)
#     username = db.Column(db.String(100), nullable=True)
#     password = db.Column(db.String(100), nullable=True)
#     creatime = db.Column(db.DateTime, default=datetime.now)


# 数据库 ORM 模型，通知
class NotificationModel(db.Model):
    __tablename__ = "notification_data"
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    # 提取日期
    date = db.Column(db.DateTime, nullable=True)
    # 摘要
    summary = db.Column(db.Text, nullable=True)

    creator_id = db.Column(db.Integer)
    # 创建时间
    created_date = db.Column(db.DateTime, default=datetime.now)
    
    # 这是一个关系，允许从UserModel的实例访问其所有的通知
    # user = db.relationship('UserModel', back_populates='notifications')
