from datetime import datetime
"""
start_time = timer(None)
timer(start_time)
"""
def timer(start_time=None, title=""):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n'+ title +' Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
