def main():
    from .task import Task
    task = Task.from_args()

    if task.do_image:
        from .image import make_image
        make_image(task)
    else:
        from .video import make_video
        make_video(task)


if __name__ == '__main__':
    main()
