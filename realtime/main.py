import multiprocessing

def main():
    multiprocessing.freeze_support()
    from realtime.main_window import main as window_main

    window_main()

if __name__ == "__main__":
    main()
