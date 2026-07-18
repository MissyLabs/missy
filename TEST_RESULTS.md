# TEST_RESULTS

- Timestamp: 2026-07-07 22:02:09
- Command: pytest -q

```
Traceback (most recent call last):
  File "/home/missy/.local/bin/pytest", line 8, in <module>
    sys.exit(console_main())
             ^^^^^^^^^^^^^^
  File "/home/missy/.local/lib/python3.12/site-packages/_pytest/config/__init__.py", line 223, in console_main
    code = main()
           ^^^^^^
  File "/home/missy/.local/lib/python3.12/site-packages/_pytest/config/__init__.py", line 193, in main
    config = _prepareconfig(new_args, plugins)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/missy/.local/lib/python3.12/site-packages/_pytest/config/__init__.py", line 361, in _prepareconfig
    config: Config = pluginmanager.hook.pytest_cmdline_parse(
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/missy/.local/lib/python3.12/site-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/missy/.local/lib/python3.12/site-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/missy/.local/lib/python3.12/site-packages/pluggy/_callers.py", line 167, in _multicall
    raise exception
  File "/home/missy/.local/lib/python3.12/site-packages/pluggy/_callers.py", line 139, in _multicall
    teardown.throw(exception)
  File "/home/missy/.local/lib/python3.12/site-packages/_pytest/helpconfig.py", line 124, in pytest_cmdline_parse
    config = yield
             ^^^^^
  File "/home/missy/.local/lib/python3.12/site-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/missy/.local/lib/python3.12/site-packages/_pytest/config/__init__.py", line 1186, in pytest_cmdline_parse
    self.parse(args)
  File "/home/missy/.local/lib/python3.12/site-packages/_pytest/config/__init__.py", line 1539, in parse
    self.pluginmanager.load_setuptools_entrypoints("pytest11")
  File "/home/missy/.local/lib/python3.12/site-packages/pluggy/_manager.py", line 416, in load_setuptools_entrypoints
    plugin = ep.load()
             ^^^^^^^^^
  File "/usr/lib/python3.12/importlib/metadata/__init__.py", line 205, in load
    module = import_module(match.group('module'))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1310, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "/home/missy/.local/lib/python3.12/site-packages/_pytest/assertion/rewrite.py", line 197, in exec_module
    exec(co, module.__dict__)
  File "/home/missy/.local/lib/python3.12/site-packages/pytest_asyncio/__init__.py", line 7, in <module>
    from .plugin import fixture, is_async_test
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "/home/missy/.local/lib/python3.12/site-packages/_pytest/assertion/rewrite.py", line 197, in exec_module
    exec(co, module.__dict__)
  File "/home/missy/.local/lib/python3.12/site-packages/pytest_asyncio/plugin.py", line 5, in <module>
    import asyncio
  File "/usr/lib/python3.12/asyncio/__init__.py", line 17, in <module>
    from .streams import *
  File "/usr/lib/python3.12/asyncio/streams.py", line 418, in <module>
    class StreamReader:
KeyboardInterrupt
