import streaming

def clean_stale_memory():
    try:
        streaming.base.util.clean_stale_shared_memory()
    except FileNotFoundError as e:
        pass

if __name__ == "__main__":
    clean_stale_memory()