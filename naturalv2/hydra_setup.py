from omegaconf import OmegaConf


def _register_resolvers():
    """Registers all custom OmegaConf resolvers."""

    def coalesce(*keys, _root_):
        for key in keys:
            value = OmegaConf.select(_root_, str(key))
            if value is not None:
                return value
        return None

    if not OmegaConf.has_resolver("coalesce"):
        OmegaConf.register_new_resolver("coalesce", coalesce, use_cache=False)


# Call the function so the registration happens when the module is imported.
_register_resolvers()
