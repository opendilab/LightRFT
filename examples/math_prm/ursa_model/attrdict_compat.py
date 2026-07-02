try:
    from attrdict import AttrDict  # type: ignore
except ImportError:
    try:
        from easydict import EasyDict as AttrDict  # type: ignore
    except ImportError:
        class AttrDict(dict):
            """Minimal AttrDict fallback for URSA config objects."""

            def __getattr__(self, item):
                try:
                    return self[item]
                except KeyError as exc:
                    raise AttributeError(item) from exc

            def __setattr__(self, key, value):
                self[key] = value

            def __delattr__(self, item):
                try:
                    del self[item]
                except KeyError as exc:
                    raise AttributeError(item) from exc
