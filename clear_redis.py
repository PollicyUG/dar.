# This short program is used to flush the redis instance so that we do
# not run out of memory since the Heroku redis add-on only gives a few MB

import os
import redis
from config import CeleryConfig

redis_url = CeleryConfig.broker_url
r = redis.from_url(redis_url)
r.flushdb()
