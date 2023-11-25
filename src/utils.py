def equivalent_dict(prev_config, configs):
    for key in prev_config.keys():
        if prev_config[key] != configs[key]:
            return False

    return True