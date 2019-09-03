web: python clear_redis.py && python app.py
worker: celery worker -A answer_policy_question.celery -O fair --loglevel=INFO --concurrency=2 --max-tasks-per-child=1
