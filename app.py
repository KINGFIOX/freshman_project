from flask import Flask, render_template  # 引入模板插件
import config  # 数据库的配置文件
from exts import db

# from blueprints import user_bp
from blueprints import notice_bp
from flask_cors import CORS  # Flask的跨域问题
from flask_migrate import Migrate

# from models import UserModel as User  没有用到用户系统

# 应用配置信息
app = Flask(
    __name__,
    static_folder="./static",  # 设置静态文件夹目录
    template_folder="./templates",
    # static_url_path=""
)
app.config.from_object(config)
db.init_app(app)

CORS(app, supports_credentials=True)
# app.config['SECRET_KEY'] = '********'

# 数据库 的 加载
migrate = Migrate(app, db)

# 注册蓝图，用户系统，没有用到
# app.register_blueprint(user_bp)

# 注册蓝图，数据库
app.register_blueprint(notice_bp)


@app.route("/")
def index():
    # 使用模板插件，引入index.html。此处会自动Flask模板文件目录寻找index.html文件。
    return render_template("categories.html", name="index")


@app.route("/categories")
def categories():
    # 使用模板插件，引入index.html。此处会自动Flask模板文件目录寻找index.html文件。
    return render_template("categories.html", name="categories")


@app.route("/calendar")
def calendar():
    # 使用模板插件，引入index.html。此处会自动Flask模板文件目录寻找index.html文件。
    return render_template("calendar.html", name="calendar")


@app.route("/search")
def search():
    # 使用模板插件，引入index.html。此处会自动Flask模板文件目录寻找index.html文件。
    return render_template("search.html", name="search")


@app.route("/add")
def add_notice():
    # 使用模板插件，引入index.html。此处会自动Flask模板文件目录寻找index.html文件。
    return render_template("add.html", name="add")


if __name__ == "__main__":
    # lsof -i :5000

    # with app.app_context():
    #     db.create_all()

    #     # Check if the admin user already exists
    #     admin = User.query.filter_by(username='admin').first()

    #     # If the admin user does not exist, create it
    #     if admin is None:
    #         admin = User(username='admin', password='admin')
    #         db.session.add(admin)
    #         db.session.commit()

    # app.run(host="0.0.0.0", port=7891, debug=True)
    app.run(debug=True)
