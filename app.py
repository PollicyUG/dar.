import os
import json
from flask import Flask, render_template, request, jsonify, url_for
from flask_bootstrap import Bootstrap
from gevent.pywsgi import WSGIServer

import wikipedia

from celery import subtask, chord

from config import ProdConfig

# import backgroud tasks here
from answer_policy_question import process_question, get_DOAJ_articles,\
    get_Crossref_articles, get_CORE_articles, answer_question

app = Flask(__name__)

app.config.from_object(ProdConfig)

Bootstrap(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the home page."""
    return render_template('index.html')


@app.route('/backgroundtask', methods=['POST', 'GET'])
def backgroundtask():
    """Start the background tasks."""
    question = request.json['question']
    keywords = process_question(question)

    # use a chord here
    callback = answer_question.subtask(kwargs={'keywords': keywords})
    header = [
        get_DOAJ_articles.subtask(args=(keywords, )),
        get_Crossref_articles.subtask(args=(keywords, )),
        get_CORE_articles.subtask(args=(keywords, ))
    ]

    task = chord(header)(callback)

    return jsonify(
        {}), 202, {
        'Location': url_for(
            'taskstatus', task_id=task.id)}


@app.route('/status/<task_id>', methods=['GET'])
def taskstatus(task_id):
    """Check on the status of the background tasks."""
    # remember answer_question is the callback so we use its task id
    task = answer_question.AsyncResult(task_id)
    response_data = {'task_status': task.status, 'task_id': task.id}

    if task.status == 'SUCCESS':
        response_data['results'] = task.get()
    return jsonify(response_data)


@app.route('/result', methods=['POST'])
def returnanswer():
    """Return an answer to the user."""
    question = request.json['replies']['question']
    answer = request.json['results']

    try:
        summary_policy = wikipedia.summary(request.json['replies']['policy'])
    except (DisambiguationError, HTTPTimeOutError, PageError, RedirectError):
        summary_policy = ''

    try:
        summary_consequence = wikipedia.summary(
            request.json['replies']['phenomenon'])
    except (DisambiguationError, HTTPTimeOutError, PageError, RedirectError):
        summary_consequence = ''

    return render_template(
        'answer.html',
        question=question,
        answer=answer,
        summary_policy=summary_policy,
        summary_consequence=summary_consequence)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'
    http_server = WSGIServer((host, port), app)
    print("Starting server on port {}".format(port))
    http_server.serve_forever()
