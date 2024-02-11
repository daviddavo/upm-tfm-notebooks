import time

def remove_unused_categories(df):
    for c in df.select_dtypes(include=['category']):
        df[c] = df[c].cat.remove_unused_categories()
    return df

class Timer:
    def __init__(self, get_time = time.perf_counter, print=False):
        self.get_time = get_time
        self.print = print
    
    def __enter__(self):
        self.start = self.get_time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time = self.get_time() - self.start
        if self.print:
            print(self.time)
        