"""One-shot transfer of an original backend's already initialized SDK store.

No replacement allocation/copy method is provided. The consumer resumes its
original registration, descriptor, transfer and close implementation.
"""


class EarlyStoreHandoff:
    def __init__(self, backend_type, owner, barrier, pause):
        if owner.store is None or not owner._init_bm:
            raise ValueError('a real initialized worker store is required')
        self.backend_type, self.owner = backend_type, owner
        self.original = backend_type._setup_store
        self.claimed = False
        self.barrier, self.pause = barrier, pause

        def setup(consumer):
            if not consumer._init_bm:
                return self.original(consumer)
            if self.claimed or consumer.device_id != owner.device_id:
                raise RuntimeError('early SDK store consumer identity differs')
            consumer.set_device()
            if consumer._dp_init_barrier:
                self.barrier()
            store = owner.store
            # No second owner may close the same actual SDK client.
            owner.store = None
            owner._store_initialized = False
            self.claimed = True
            self.pause()
            if backend_type._setup_store is not setup:
                raise RuntimeError('early store factory ownership changed')
            backend_type._setup_store = self.original
            return store

        self.wrapper = setup
        backend_type._setup_store = setup
