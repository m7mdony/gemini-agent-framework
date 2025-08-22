import time
from threading import Lock

# -------- Local Token Bucket --------
class LocalTokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.timestamp = time.time()
        self.lock = Lock()

    def consume(self, tokens):
        with self.lock:
            now = time.time()
            elapsed = now - self.timestamp
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.timestamp = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def refund(self, tokens):
        """Refund tokens back to the bucket, capped at bucket capacity"""
        with self.lock:
            now = time.time()
            elapsed = now - self.timestamp
            # Update current tokens with refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.timestamp = now
            
            # Add refunded tokens, but don't exceed capacity
            self.tokens = min(self.capacity, self.tokens + tokens)
            return True

    def get_balance(self):
        now = time.time()
        elapsed = now - self.timestamp
        current_tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        return int(current_tokens)