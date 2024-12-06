from .EIM import EIM
from .ImageImageMatcher import ImageImageMatcher


def build_model(config, device, logger):
    if config.name == 'EIM':
        model = EIM(config, device, logger)
    elif config.name == 'ImageImageMatcher':
        model = ImageImageMatcher(config, device, logger)
    else:
        raise NotImplementedError(f'Unsupported model: {config.name}')
    return model
