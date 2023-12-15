import re
from flask import Blueprint, request, jsonify
from models import NotificationModel
from exts import db
import datetime
from sqlalchemy import or_

# 引入模型的依赖
import torch
from transformers import PegasusForConditionalGeneration
import tokenizers_pegasus as tkg
from transformers import BertForSequenceClassification
from transformers import BertTokenizer

bp = Blueprint("notice", __name__, url_prefix="/Notice")

# 打印调试信息
print("开始加载模型，请稍等...")
model_summarize = PegasusForConditionalGeneration.from_pretrained(
    "IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese-V1"
)
tokenizer_summarize = tkg.PegasusTokenizer.from_pretrained(
    "IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese-V1"
)

model_save_path = "./model/"
# 分类器加载
model_classify = BertForSequenceClassification.from_pretrained(model_save_path)
# tokenizer 加载
tokenizer_classify = BertTokenizer.from_pretrained(model_save_path)

# 通知的分类
label_to_id = {
    "升学": 0,
    "志愿": 1,
    "教务": 2,
    "思政": 3,
    "心理": 4,
    "灾害": 5,
    "作业与考试": 6,
    "竞赛与机会": 7,
    "企业参观与就业": 8,
    "生活": 9,
    "重要通知": 10,
    "垃圾与乐子": 11,
}

# 将label_to_id进行反转
id_to_label = {value: key for key, value in label_to_id.items()}


def predict_label(text):
    # 对文本进行编码
    inputs = tokenizer_classify(
        text, return_tensors="pt", truncation=True, padding="max_length", max_length=128
    )

    # 将输入移到模型所在的设备上
    inputs = {key: val.to(model_classify.device) for key, val in inputs.items()}

    # 使用模型进行预测，这里不会进行训练
    with torch.no_grad():
        outputs = model_classify(**inputs)
        logits = outputs.logits
        predicted_label_id = logits.argmax(-1).item()

    # 获取预测的类别名
    predicted_label = id_to_label[predicted_label_id]

    return predicted_label


def summarize_text(
    text, max_length=128, num_beams=4, length_penalty=2.0, max_length_output=150
):
    """
    用输入的模型和分词器生成输入文本的摘要。
    参数：
        - text (str): 要生成摘要的输入文本。
    """

    inputs = tokenizer_summarize(
        text, max_length=max_length, truncation=True, return_tensors="pt"
    )

    # beam search 生成摘要
    summary_ids = model_summarize.generate(
        inputs["input_ids"],
        num_beams=num_beams,
        length_penalty=length_penalty,  # 惩罚系数是2.0
        max_length=max_length_output,
        no_repeat_ngram_size=3,  # 生成文本时避免重复的n元组的大小
    )

    return tokenizer_summarize.batch_decode(
        summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def find_date(text, patterns):
    """
    使用正则表达式从文本中查找日期

    Args:
        text (str): 输入文本
        patterns (list of str): 日期的正则表达式列表

    Returns:
        str: 找到的第一个日期字符串；如果未找到，则为None
    """
    for pattern in patterns:
        # 通过正则表达式找日期
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


# 添加新通知
@bp.route("/Add", methods=["POST"])
def AddNotice():
    data = request.json

    # 分类
    new_notice = NotificationModel(
        title=data.get("title"),
        content=data.get("content"),
        date=datetime.datetime.strptime(data.get("date"), "%Y年%m月%d日")
        if data.get("date")
        else None,
        summary=data.get("summary"),
        creator_id=data.get("creator_id"),
    )

    # 提交数据库
    db.session.add(new_notice)
    db.session.commit()

    data = {
        "title": data.get("title"),
        "content": data.get("content"),
        "date": new_notice.date,
        "summary": data.get("summary"),
        "creator_id": data.get("creator_id"),
    }

    return jsonify({"code": 200, "data": data, "msg": "Notice added successfully"})


# 查询通知
@bp.route("/Search", methods=["GET"])
def SearchNotice():
    keyword = request.args.get("keyword")
    title = request.args.get("title")
    date = request.args.get("date")

    query = NotificationModel.query

    # 模糊查询
    if keyword:
        keyword_search = or_(
            NotificationModel.content.like(f"%{keyword}%"),
            NotificationModel.summary.like(f"%{keyword}%"),
        )
        query = query.filter(keyword_search)

    # 根据类别查询
    if title:
        query = query.filter_by(title=title)

    # 根据日期查询
    if date:
        target_date = datetime.datetime.strptime(date, "%Y-%m-%d")
        query = query.filter_by(date=target_date)
    else:
        # 按created_date降序排列
        query = query.order_by(NotificationModel.created_date.desc())

    results = query.all()

    # 搜索到的结果
    notices = [
        {
            "id": notice.id,
            "title": notice.title,
            "content": notice.content,
            "date": notice.date.strftime("%Y-%m-%d %H:%M:%S") if notice.date else None,
            "summary": notice.summary,
            "creator_id": notice.creator_id,
            "created_date": notice.created_date.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for notice in results
    ]

    return jsonify({"code": 200, "data": notices})


@bp.route("/AutoAdd", methods=["POST"])
def AutoAddNotice():
    data = request.json

    content = data.get("content")
    # 去除前后空格
    content = content.strip()

    # 自动生成摘要
    summary = summarize_text(content)

    # 自动生成类别标签
    title = predict_label(content)

    # 自动提取日期
    patterns = [
        r"\d{4}年\d{1,2}月\d{1,2}日",  # YYYY年MM月DD日
    ]
    date_str = find_date(content, patterns)
    date = datetime.datetime.strptime(date_str, "%Y年%m月%d日") if date_str else None

    # new_notice = NotificationModel(
    #     title=title,
    #     content=content,
    #     date=date,
    #     summary=summary,
    #     creator_id=data.get("creator_id")
    # )

    data = {
        "title": title,
        "content": content,
        "date": date_str,
        "summary": summary,
        "creator_id": data.get("creator_id"),
    }

    # db.session.add(new_notice)
    # db.session.commit()

    return jsonify(
        {
            "code": 200,
            "data": data,
            "msg": "Notice added successfully with auto features",
        }
    )


# if __name__ == '__main__':
#     from flask import Flask, request

#     app = Flask(__name__)

#     test_data = {
#         "content": "各位同学注意，由于疫情原因，2023年6月15日学校将暂停开放，恢复时间另行通知。",
#         "creator_id": 1  # 使用任何适合的creator_id
#     }

#     with app.test_request_context(json=test_data):
#         response = AutoAddNotice()
#         print(response.get_json())  # 这将显示函数的输出
