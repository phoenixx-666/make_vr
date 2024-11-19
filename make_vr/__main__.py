if __name__ == '__main__':
    from .config import Config
    cfg = Config.from_args()
    from .fs import validate_input_files
    validate_input_files(cfg)

    if cfg.do_image:
        from .image import make_image
        make_image(cfg)
    else:
        from .video import make_video
        make_video(cfg)
