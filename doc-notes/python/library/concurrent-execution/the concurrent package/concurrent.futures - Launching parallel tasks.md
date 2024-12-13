Added in version 3.2.

**Source code:** [Lib/concurrent/futures/thread.py](https://github.com/python/cpython/tree/3.13/Lib/concurrent/futures/thread.py) and [Lib/concurrent/futures/process.py](https://github.com/python/cpython/tree/3.13/Lib/concurrent/futures/process.py)

---

The [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures "concurrent.futures: Execute computations concurrently using threads or processes.") module provides a high-level interface for asynchronously executing callables.
>  `concurrent.futures` 模块为异步执行的可调用对象提供了高级接口

The asynchronous execution can be performed with threads, using [`ThreadPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor "concurrent.futures.ThreadPoolExecutor"), or separate processes, using [`ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor "concurrent.futures.ProcessPoolExecutor"). Both implement the same interface, which is defined by the abstract [`Executor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor "concurrent.futures.Executor") class.
>  异步执行可以使用 `ThreadPoolExecutor` 多线程执行，也可以使用 `ProcessPoolExecutor` 多进程执行
>  这两个类的接口一致，都由抽象类 `Executor` 定义


[Availability](https://docs.python.org/3/library/intro.html#availability): not WASI.

This module does not work or is not available on WebAssembly. See [WebAssembly platforms](https://docs.python.org/3/library/intro.html#wasm-availability) for more information.

## Executor Objects
_class_ `concurrent.futures.Executor`
An abstract class that provides methods to execute calls asynchronously. It should not be used directly, but through its concrete subclasses.

`submit(fn, /, *args, **kwargs) `
Schedules the callable, _fn_, to be executed as `fn(*args, **kwargs)` and returns a [`Future`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future "concurrent.futures.Future") object representing the execution of the callable.

```python
with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(pow, 323, 1235)
    print(future.result())
```

`map(fn, *iterables, timeout=None, chunksize=1)`

Similar to [`map(fn, *iterables)`](https://docs.python.org/3/library/functions.html#map "map") except:

- the _iterables_ are collected immediately rather than lazily;
- _fn_ is executed asynchronously and several calls to _fn_ may be made concurrently.

The returned iterator raises a [`TimeoutError`](https://docs.python.org/3/library/exceptions.html#TimeoutError "TimeoutError") if [`__next__()`](https://docs.python.org/3/library/stdtypes.html#iterator.__next__ "iterator.__next__") is called and the result isn’t available after _timeout_ seconds from the original call to [`Executor.map()`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor.map "concurrent.futures.Executor.map"). _timeout_ can be an int or a float. If _timeout_ is not specified or `None`, there is no limit to the wait time.

If a _fn_ call raises an exception, then that exception will be raised when its value is retrieved from the iterator.

When using [`ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor "concurrent.futures.ProcessPoolExecutor"), this method chops _iterables_ into a number of chunks which it submits to the pool as separate tasks. The (approximate) size of these chunks can be specified by setting _chunksize_ to a positive integer. For very long iterables, using a large value for _chunksize_ can significantly improve performance compared to the default size of 1. With [`ThreadPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor "concurrent.futures.ThreadPoolExecutor"), _chunksize_ has no effect.

Changed in version 3.5: Added the _chunksize_ argument.

`shutdown(wait=True, *, cancel_futures=False)``
Signal the executor that it should free any resources that it is using when the currently pending futures are done executing. Calls to [`Executor.submit()`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor.submit "concurrent.futures.Executor.submit") and [`Executor.map()`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor.map "concurrent.futures.Executor.map") made after shutdown will raise [`RuntimeError`](https://docs.python.org/3/library/exceptions.html#RuntimeError "RuntimeError").

If _wait_ is `True` then this method will not return until all the pending futures are done executing and the resources associated with the executor have been freed. If _wait_ is `False` then this method will return immediately and the resources associated with the executor will be freed when all pending futures are done executing. Regardless of the value of _wait_, the entire Python program will not exit until all pending futures are done executing.

If _cancel_futures_ is `True`, this method will cancel all pending futures that the executor has not started running. Any futures that are completed or running won’t be cancelled, regardless of the value of _cancel_futures_.

If both _cancel_futures_ and _wait_ are `True`, all futures that the executor has started running will be completed prior to this method returning. The remaining futures are cancelled.

You can avoid having to call this method explicitly if you use the [`with`](https://docs.python.org/3/reference/compound_stmts.html#with) statement, which will shutdown the [`Executor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor "concurrent.futures.Executor") (waiting as if [`Executor.shutdown()`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor.shutdown "concurrent.futures.Executor.shutdown") were called with _wait_ set to `True`):

```python
import shutil
with ThreadPoolExecutor(max_workers=4) as e:
    e.submit(shutil.copy, 'src1.txt', 'dest1.txt')
    e.submit(shutil.copy, 'src2.txt', 'dest2.txt')
    e.submit(shutil.copy, 'src3.txt', 'dest3.txt')
    e.submit(shutil.copy, 'src4.txt', 'dest4.txt')
```

Changed in version 3.9: Added _cancel_futures_.

## ThreadPoolExecutor
[`ThreadPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor "concurrent.futures.ThreadPoolExecutor") is an [`Executor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor "concurrent.futures.Executor") subclass that uses a pool of threads to execute calls asynchronously.

Deadlocks can occur when the callable associated with a [`Future`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future "concurrent.futures.Future") waits on the results of another [`Future`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future "concurrent.futures.Future"). For example:

import time
def wait_on_b():
    time.sleep(5)
    print(b.result())  # b will never complete because it is waiting on a.
    return 5

def wait_on_a():
    time.sleep(5)
    print(a.result())  # a will never complete because it is waiting on b.
    return 6

executor = ThreadPoolExecutor(max_workers=2)
a = executor.submit(wait_on_b)
b = executor.submit(wait_on_a)

And:

def wait_on_future():
    f = executor.submit(pow, 5, 2)
    # This will never complete because there is only one worker thread and
    # it is executing this function.
    print(f.result())

executor = ThreadPoolExecutor(max_workers=1)
executor.submit(wait_on_future)

_class_ concurrent.futures.ThreadPoolExecutor(_max_workers=None_, _thread_name_prefix=''_, _initializer=None_, _initargs=()_)[](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor "Link to this definition")

An [`Executor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor "concurrent.futures.Executor") subclass that uses a pool of at most _max_workers_ threads to execute calls asynchronously.

All threads enqueued to `ThreadPoolExecutor` will be joined before the interpreter can exit. Note that the exit handler which does this is executed _before_ any exit handlers added using `atexit`. This means exceptions in the main thread must be caught and handled in order to signal threads to exit gracefully. For this reason, it is recommended that `ThreadPoolExecutor` not be used for long-running tasks.

_initializer_ is an optional callable that is called at the start of each worker thread; _initargs_ is a tuple of arguments passed to the initializer. Should _initializer_ raise an exception, all currently pending jobs will raise a [`BrokenThreadPool`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.thread.BrokenThreadPool "concurrent.futures.thread.BrokenThreadPool"), as well as any attempt to submit more jobs to the pool.

Changed in version 3.5: If _max_workers_ is `None` or not given, it will default to the number of processors on the machine, multiplied by `5`, assuming that [`ThreadPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor "concurrent.futures.ThreadPoolExecutor") is often used to overlap I/O instead of CPU work and the number of workers should be higher than the number of workers for [`ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor "concurrent.futures.ProcessPoolExecutor").

Changed in version 3.6: Added the _thread_name_prefix_ parameter to allow users to control the [`threading.Thread`](https://docs.python.org/3/library/threading.html#threading.Thread "threading.Thread") names for worker threads created by the pool for easier debugging.

Changed in version 3.7: Added the _initializer_ and _initargs_ arguments.

Changed in version 3.8: Default value of _max_workers_ is changed to `min(32, os.cpu_count() + 4)`. This default value preserves at least 5 workers for I/O bound tasks. It utilizes at most 32 CPU cores for CPU bound tasks which release the GIL. And it avoids using very large resources implicitly on many-core machines.

ThreadPoolExecutor now reuses idle worker threads before starting _max_workers_ worker threads too.

Changed in version 3.13: Default value of _max_workers_ is changed to `min(32, (os.process_cpu_count() or 1) + 4)`.

### ThreadPoolExecutor Example[](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor-example "Link to this heading")

import concurrent.futures
import urllib.request

URLS = ['http://www.foxnews.com/',
        'http://www.cnn.com/',
        'http://europe.wsj.com/',
        'http://www.bbc.co.uk/',
        'http://nonexistent-subdomain.python.org/']

# Retrieve a single page and report the URL and contents
def load_url(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as conn:
        return conn.read()

# We can use a with statement to ensure threads are cleaned up promptly
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # Start the load operations and mark each future with its URL
    future_to_url = {executor.submit(load_url, url, 60): url for url in URLS}
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
        except Exception as exc:
            print('%r generated an exception: %s' % (url, exc))
        else:
            print('%r page is %d bytes' % (url, len(data)))

## ProcessPoolExecutor[](https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor "Link to this heading")

The [`ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor "concurrent.futures.ProcessPoolExecutor") class is an [`Executor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor "concurrent.futures.Executor") subclass that uses a pool of processes to execute calls asynchronously. [`ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor "concurrent.futures.ProcessPoolExecutor") uses the [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing "multiprocessing: Process-based parallelism.") module, which allows it to side-step the [Global Interpreter Lock](https://docs.python.org/3/glossary.html#term-global-interpreter-lock) but also means that only picklable objects can be executed and returned.

The `__main__` module must be importable by worker subprocesses. This means that [`ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor "concurrent.futures.ProcessPoolExecutor") will not work in the interactive interpreter.

Calling [`Executor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor "concurrent.futures.Executor") or [`Future`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future "concurrent.futures.Future") methods from a callable submitted to a [`ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor "concurrent.futures.ProcessPoolExecutor") will result in deadlock.

_class_ concurrent.futures.ProcessPoolExecutor(_max_workers=None_, _mp_context=None_, _initializer=None_, _initargs=()_, _max_tasks_per_child=None_)[](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor "Link to this definition")

An [`Executor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor "concurrent.futures.Executor") subclass that executes calls asynchronously using a pool of at most _max_workers_ processes. If _max_workers_ is `None` or not given, it will default to [`os.process_cpu_count()`](https://docs.python.org/3/library/os.html#os.process_cpu_count "os.process_cpu_count"). If _max_workers_ is less than or equal to `0`, then a [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError "ValueError") will be raised. On Windows, _max_workers_ must be less than or equal to `61`. If it is not then [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError "ValueError") will be raised. If _max_workers_ is `None`, then the default chosen will be at most `61`, even if more processors are available. _mp_context_ can be a [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing "multiprocessing: Process-based parallelism.") context or `None`. It will be used to launch the workers. If _mp_context_ is `None` or not given, the default [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing "multiprocessing: Process-based parallelism.") context is used. See [Contexts and start methods](https://docs.python.org/3/library/multiprocessing.html#multiprocessing-start-methods).

_initializer_ is an optional callable that is called at the start of each worker process; _initargs_ is a tuple of arguments passed to the initializer. Should _initializer_ raise an exception, all currently pending jobs will raise a [`BrokenProcessPool`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.process.BrokenProcessPool "concurrent.futures.process.BrokenProcessPool"), as well as any attempt to submit more jobs to the pool.

_max_tasks_per_child_ is an optional argument that specifies the maximum number of tasks a single process can execute before it will exit and be replaced with a fresh worker process. By default _max_tasks_per_child_ is `None` which means worker processes will live as long as the pool. When a max is specified, the “spawn” multiprocessing start method will be used by default in absence of a _mp_context_ parameter. This feature is incompatible with the “fork” start method.

Changed in version 3.3: When one of the worker processes terminates abruptly, a [`BrokenProcessPool`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.process.BrokenProcessPool "concurrent.futures.process.BrokenProcessPool") error is now raised. Previously, behaviour was undefined but operations on the executor or its futures would often freeze or deadlock.

Changed in version 3.7: The _mp_context_ argument was added to allow users to control the start_method for worker processes created by the pool.

Added the _initializer_ and _initargs_ arguments.

Note

 

The default [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing "multiprocessing: Process-based parallelism.") start method (see [Contexts and start methods](https://docs.python.org/3/library/multiprocessing.html#multiprocessing-start-methods)) will change away from _fork_ in Python 3.14. Code that requires _fork_ be used for their [`ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor "concurrent.futures.ProcessPoolExecutor") should explicitly specify that by passing a `mp_context=multiprocessing.get_context("fork")` parameter.

Changed in version 3.11: The _max_tasks_per_child_ argument was added to allow users to control the lifetime of workers in the pool.

Changed in version 3.12: On POSIX systems, if your application has multiple threads and the [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing "multiprocessing: Process-based parallelism.") context uses the `"fork"` start method: The [`os.fork()`](https://docs.python.org/3/library/os.html#os.fork "os.fork") function called internally to spawn workers may raise a [`DeprecationWarning`](https://docs.python.org/3/library/exceptions.html#DeprecationWarning "DeprecationWarning"). Pass a _mp_context_ configured to use a different start method. See the [`os.fork()`](https://docs.python.org/3/library/os.html#os.fork "os.fork") documentation for further explanation.

Changed in version 3.13: _max_workers_ uses [`os.process_cpu_count()`](https://docs.python.org/3/library/os.html#os.process_cpu_count "os.process_cpu_count") by default, instead of [`os.cpu_count()`](https://docs.python.org/3/library/os.html#os.cpu_count "os.cpu_count").

### ProcessPoolExecutor Example[](https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor-example "Link to this heading")

import concurrent.futures
import math

PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True

def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
            print('%d is prime: %s' % (number, prime))

if __name__ == '__main__':
    main()

## Future Objects
The [`Future`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future "concurrent.futures.Future") class encapsulates the asynchronous execution of a callable. [`Future`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future "concurrent.futures.Future") instances are created by [`Executor.submit()`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor.submit "concurrent.futures.Executor.submit").
>  `Future` 类封装了可调用对象的异步执行
>  `Future` 实例由 `Executor.submit()` 创建

_class_ `concurrent.futures.Future`
Encapsulates the asynchronous execution of a callable. [`Future`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future "concurrent.futures.Future") instances are created by [`Executor.submit()`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor.submit "concurrent.futures.Executor.submit") and should not be created directly except for testing.

`cancel()`
Attempt to cancel the call. If the call is currently being executed or finished running and cannot be cancelled then the method will return `False`, otherwise the call will be cancelled and the method will return `True`.

`cancelled()`
Return `True` if the call was successfully cancelled.

`running()`
Return `True` if the call is currently being executed and cannot be cancelled.

`done()`
Return `True` if the call was successfully cancelled or finished running.

`result(timeout=None)`
Return the value returned by the call. If the call hasn’t yet completed then this method will wait up to _timeout_ seconds. If the call hasn’t completed in _timeout_ seconds, then a [`TimeoutError`](https://docs.python.org/3/library/exceptions.html#TimeoutError "TimeoutError") will be raised. _timeout_ can be an int or float. If _timeout_ is not specified or `None`, there is no limit to the wait time.

If the future is cancelled before completing then [`CancelledError`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.CancelledError "concurrent.futures.CancelledError") will be raised.

If the call raised an exception, this method will raise the same exception.

`exception(timeout=None)`
Return the exception raised by the call. If the call hasn’t yet completed then this method will wait up to _timeout_ seconds. If the call hasn’t completed in _timeout_ seconds, then a [`TimeoutError`](https://docs.python.org/3/library/exceptions.html#TimeoutError "TimeoutError") will be raised. _timeout_ can be an int or float. If _timeout_ is not specified or `None`, there is no limit to the wait time.

If the future is cancelled before completing then [`CancelledError`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.CancelledError "concurrent.futures.CancelledError") will be raised.

If the call completed without raising, `None` is returned.

add_done_callback(_fn_)[](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future.add_done_callback "Link to this definition")

Attaches the callable _fn_ to the future. _fn_ will be called, with the future as its only argument, when the future is cancelled or finishes running.

Added callables are called in the order that they were added and are always called in a thread belonging to the process that added them. If the callable raises an [`Exception`](https://docs.python.org/3/library/exceptions.html#Exception "Exception") subclass, it will be logged and ignored. If the callable raises a [`BaseException`](https://docs.python.org/3/library/exceptions.html#BaseException "BaseException") subclass, the behavior is undefined.

If the future has already completed or been cancelled, _fn_ will be called immediately.

The following [`Future`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future "concurrent.futures.Future") methods are meant for use in unit tests and [`Executor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor "concurrent.futures.Executor") implementations.

set_running_or_notify_cancel()[](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future.set_running_or_notify_cancel "Link to this definition")

This method should only be called by [`Executor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor "concurrent.futures.Executor") implementations before executing the work associated with the [`Future`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future "concurrent.futures.Future") and by unit tests.

If the method returns `False` then the [`Future`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future "concurrent.futures.Future") was cancelled, i.e. [`Future.cancel()`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future.cancel "concurrent.futures.Future.cancel") was called and returned `True`. Any threads waiting on the [`Future`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future "concurrent.futures.Future") completing (i.e. through [`as_completed()`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.as_completed "concurrent.futures.as_completed") or [`wait()`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.wait "concurrent.futures.wait")) will be woken up.

If the method returns `True` then the [`Future`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future "concurrent.futures.Future") was not cancelled and has been put in the running state, i.e. calls to [`Future.running()`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future.running "concurrent.futures.Future.running") will return `True`.

This method can only be called once and cannot be called after [`Future.set_result()`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future.set_result "concurrent.futures.Future.set_result") or [`Future.set_exception()`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future.set_exception "concurrent.futures.Future.set_exception") have been called.

set_result(_result_)[](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future.set_result "Link to this definition")

Sets the result of the work associated with the [`Future`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future "concurrent.futures.Future") to _result_.

This method should only be used by [`Executor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor "concurrent.futures.Executor") implementations and unit tests.

Changed in version 3.8: This method raises [`concurrent.futures.InvalidStateError`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.InvalidStateError "concurrent.futures.InvalidStateError") if the [`Future`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future "concurrent.futures.Future") is already done.

set_exception(_exception_)[](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future.set_exception "Link to this definition")

Sets the result of the work associated with the [`Future`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future "concurrent.futures.Future") to the [`Exception`](https://docs.python.org/3/library/exceptions.html#Exception "Exception") _exception_.

This method should only be used by [`Executor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor "concurrent.futures.Executor") implementations and unit tests.

Changed in version 3.8: This method raises [`concurrent.futures.InvalidStateError`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.InvalidStateError "concurrent.futures.InvalidStateError") if the [`Future`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future "concurrent.futures.Future") is already done.

## Module Functions[](https://docs.python.org/3/library/concurrent.futures.html#module-functions "Link to this heading")

concurrent.futures.wait(_fs_, _timeout=None_, _return_when=ALL_COMPLETED_)[](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.wait "Link to this definition")

Wait for the [`Future`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future "concurrent.futures.Future") instances (possibly created by different [`Executor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor "concurrent.futures.Executor") instances) given by _fs_ to complete. Duplicate futures given to _fs_ are removed and will be returned only once. Returns a named 2-tuple of sets. The first set, named `done`, contains the futures that completed (finished or cancelled futures) before the wait completed. The second set, named `not_done`, contains the futures that did not complete (pending or running futures).

_timeout_ can be used to control the maximum number of seconds to wait before returning. _timeout_ can be an int or float. If _timeout_ is not specified or `None`, there is no limit to the wait time.

_return_when_ indicates when this function should return. It must be one of the following constants:

|Constant|Description|
|---|---|
|concurrent.futures.FIRST_COMPLETED[](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.FIRST_COMPLETED "Link to this definition")|The function will return when any future finishes or is cancelled.|
|concurrent.futures.FIRST_EXCEPTION[](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.FIRST_EXCEPTION "Link to this definition")|The function will return when any future finishes by raising an exception. If no future raises an exception then it is equivalent to [`ALL_COMPLETED`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ALL_COMPLETED "concurrent.futures.ALL_COMPLETED").|
|concurrent.futures.ALL_COMPLETED[](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ALL_COMPLETED "Link to this definition")|The function will return when all futures finish or are cancelled.|

concurrent.futures.as_completed(_fs_, _timeout=None_)[](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.as_completed "Link to this definition")

Returns an iterator over the [`Future`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future "concurrent.futures.Future") instances (possibly created by different [`Executor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor "concurrent.futures.Executor") instances) given by _fs_ that yields futures as they complete (finished or cancelled futures). Any futures given by _fs_ that are duplicated will be returned once. Any futures that completed before [`as_completed()`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.as_completed "concurrent.futures.as_completed") is called will be yielded first. The returned iterator raises a [`TimeoutError`](https://docs.python.org/3/library/exceptions.html#TimeoutError "TimeoutError") if [`__next__()`](https://docs.python.org/3/library/stdtypes.html#iterator.__next__ "iterator.__next__") is called and the result isn’t available after _timeout_ seconds from the original call to [`as_completed()`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.as_completed "concurrent.futures.as_completed"). _timeout_ can be an int or float. If _timeout_ is not specified or `None`, there is no limit to the wait time.

See also

[**PEP 3148**](https://peps.python.org/pep-3148/) – futures - execute computations asynchronously

The proposal which described this feature for inclusion in the Python standard library.

## Exception classes[](https://docs.python.org/3/library/concurrent.futures.html#exception-classes "Link to this heading")

_exception_ concurrent.futures.CancelledError[](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.CancelledError "Link to this definition")

Raised when a future is cancelled.

_exception_ concurrent.futures.TimeoutError[](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.TimeoutError "Link to this definition")

A deprecated alias of [`TimeoutError`](https://docs.python.org/3/library/exceptions.html#TimeoutError "TimeoutError"), raised when a future operation exceeds the given timeout.

Changed in version 3.11: This class was made an alias of [`TimeoutError`](https://docs.python.org/3/library/exceptions.html#TimeoutError "TimeoutError").

_exception_ concurrent.futures.BrokenExecutor[](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.BrokenExecutor "Link to this definition")

Derived from [`RuntimeError`](https://docs.python.org/3/library/exceptions.html#RuntimeError "RuntimeError"), this exception class is raised when an executor is broken for some reason, and cannot be used to submit or execute new tasks.

Added in version 3.7.

_exception_ concurrent.futures.InvalidStateError[](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.InvalidStateError "Link to this definition")

Raised when an operation is performed on a future that is not allowed in the current state.

Added in version 3.8.

_exception_ concurrent.futures.thread.BrokenThreadPool[](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.thread.BrokenThreadPool "Link to this definition")

Derived from [`BrokenExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.BrokenExecutor "concurrent.futures.BrokenExecutor"), this exception class is raised when one of the workers of a [`ThreadPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor "concurrent.futures.ThreadPoolExecutor") has failed initializing.

Added in version 3.7.

_exception_ concurrent.futures.process.BrokenProcessPool[](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.process.BrokenProcessPool "Link to this definition")

Derived from [`BrokenExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.BrokenExecutor "concurrent.futures.BrokenExecutor") (formerly [`RuntimeError`](https://docs.python.org/3/library/exceptions.html#RuntimeError "RuntimeError")), this exception class is raised when one of the workers of a [`ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor "concurrent.futures.ProcessPoolExecutor") has terminated in a non-clean fashion (for example, if it was killed from the outside).

Added in version 3.3.