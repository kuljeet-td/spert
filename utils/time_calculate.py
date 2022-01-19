import time


def time_this_function(method):
    """
    Decorate any function with this definition and time the methods
    Example would be like
    >> @time_this_function
       def foo(x):
          time.sleep(3)
          return x
    'foo' -- 3.002 seconds
    :param method:
    :return:
    """

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print('%r -- %2.3f seconds' % (method.__name__, (te - ts)))
        return result
    return timed