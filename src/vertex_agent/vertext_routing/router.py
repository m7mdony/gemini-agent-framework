try:
    import redis
    use_redis_import = True
except ImportError:
    use_redis_import = False
from .router_memory import routerMemory
from .token_bucket.local_TB import LocalTokenBucket
from .token_bucket.redis_TB import RedisTokenBucket
# -------- Router with Option --------
class RegionRouter:
    def __init__(self, use_redis=True, redis_host='localhost', redis_port=6379, redis_db=0, key_prefix='token_bucket'):
        if not use_redis_import:
            use_redis = False
        self.use_redis = use_redis
        self.key_prefix = key_prefix

        if self.use_redis:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
            self.round_robin_key = f"{self.key_prefix}:round_robin_index"
            if not self.redis_client.exists(self.round_robin_key):
                self.redis_client.set(self.round_robin_key, 0)
        else:
            self.redis_client = None
            self.round_robin_index = 0

        self.region_buckets = {}
        self.region_list = []

        for region, tpm in routerMemory.GEMINI_FLASH_TPM.value.items():
            if self.use_redis:
                bucket = RedisTokenBucket(
                    redis_client=self.redis_client,
                    key=f"{self.key_prefix}:{region}",
                    capacity=tpm,
                    refill_rate=tpm / 60
                )
            else:
                bucket = LocalTokenBucket(
                    capacity=tpm,
                    refill_rate=tpm / 60
                )

            self.region_buckets[region] = bucket
            self.region_list.append(region)

    def pick_region(self, tokens_needed=1000):
        candidates = [r for r in self.region_list if tokens_needed <= self.region_buckets[r].capacity]
        if not candidates:
            return None, None

        attempts = 0
        max_attempts = len(candidates) * 2

        while attempts < max_attempts:
            if self.use_redis:
                current_index = self.redis_client.incr(self.round_robin_key) - 1
            else:
                self.round_robin_index += 1
                current_index = self.round_robin_index - 1

            region_index = current_index % len(candidates)
            region = candidates[region_index]
            bucket = self.region_buckets[region]

            if bucket.get_balance() >= tokens_needed and bucket.consume(tokens_needed):
                return region, bucket

            attempts += 1

        return None, None

    def refund_tokens(self, region, tokens):
        """Refund tokens to a specific region"""
        if region not in self.region_buckets:
            return False
        
        return self.region_buckets[region].refund(tokens)

    def get_all_balances(self):
        return {region: bucket.get_balance() for region, bucket in self.region_buckets.items()}

    def close(self):
        if self.use_redis and self.redis_client:
            self.redis_client.close()

