try:
    from .training import main
except ImportError:
    from training import main

if __name__ == '__main__':
    main('ranker')
