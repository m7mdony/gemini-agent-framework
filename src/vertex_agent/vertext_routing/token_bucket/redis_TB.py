import time
import json
from threading import Lock
try:
    import redis
except ImportError:
    redis = None

# -------- Redis Token Bucket --------
class RedisTokenBucket:
    def __init__(self, redis_client, key, capacity, refill_rate):
        self.redis_client = redis_client
        self.key = key
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.lock = Lock()

        if not self.redis_client.exists(self.key):
            bucket_data = {
                'tokens': capacity,
                'timestamp': time.time(),
                'capacity': capacity,
                'refill_rate': refill_rate
            }
            self.redis_client.set(self.key, json.dumps(bucket_data))

    def consume(self, tokens):
        with self.lock:
            pipe = self.redis_client.pipeline()
            try:
                pipe.watch(self.key)
                bucket_data_str = pipe.get(self.key)

                if bucket_data_str is None:
                    bucket_data = {
                        'tokens': self.capacity,
                        'timestamp': time.time(),
                        'capacity': self.capacity,
                        'refill_rate': self.refill_rate
                    }
                else:
                    bucket_data = json.loads(bucket_data_str)

                now = time.time()
                elapsed = now - bucket_data['timestamp']
                current_tokens = min(
                    bucket_data['capacity'],
                    bucket_data['tokens'] + elapsed * bucket_data['refill_rate']
                )

                if current_tokens >= tokens:
                    new_tokens = current_tokens - tokens
                    bucket_data['tokens'] = new_tokens
                    bucket_data['timestamp'] = now
                    pipe.multi()
                    pipe.set(self.key, json.dumps(bucket_data))
                    pipe.execute()
                    return True
                else:
                    bucket_data['tokens'] = current_tokens
                    bucket_data['timestamp'] = now
                    pipe.multi()
                    pipe.set(self.key, json.dumps(bucket_data))
                    pipe.execute()
                    return False

            except redis.WatchError:
                return self.consume(tokens)

    def refund(self, tokens):
        """Refund tokens back to the bucket, capped at bucket capacity"""
        with self.lock:
            pipe = self.redis_client.pipeline()
            try:
                pipe.watch(self.key)
                bucket_data_str = pipe.get(self.key)

                if bucket_data_str is None:
                    bucket_data = {
                        'tokens': self.capacity,
                        'timestamp': time.time(),
                        'capacity': self.capacity,
                        'refill_rate': self.refill_rate
                    }
                else:
                    bucket_data = json.loads(bucket_data_str)

                now = time.time()
                elapsed = now - bucket_data['timestamp']
                current_tokens = min(
                    bucket_data['capacity'],
                    bucket_data['tokens'] + elapsed * bucket_data['refill_rate']
                )

                # Add refunded tokens, but don't exceed capacity
                new_tokens = min(bucket_data['capacity'], current_tokens + tokens)
                bucket_data['tokens'] = new_tokens
                bucket_data['timestamp'] = now
                
                pipe.multi()
                pipe.set(self.key, json.dumps(bucket_data))
                pipe.execute()
                return True

            except redis.WatchError:
                return self.refund(tokens)

    def get_balance(self):
        bucket_data_str = self.redis_client.get(self.key)
        if bucket_data_str is None:
            return self.capacity

        bucket_data = json.loads(bucket_data_str)
        now = time.time()
        elapsed = now - bucket_data['timestamp']
        current_tokens = min(
            bucket_data['capacity'],
            bucket_data['tokens'] + elapsed * bucket_data['refill_rate']
        )
        return int(current_tokens)

