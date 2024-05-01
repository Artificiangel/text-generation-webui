import asyncio
import traceback

async def _coro_on_error(coro, exc, error_data=None):
    print(f'Error in coro {coro}')
    if error_data:
        print(error_data)
    
    traceback_str = ''.join(traceback.format_tb(exc.__traceback__))
    print(traceback_str)
    print('')
    print(exc)
    print('')
    print('')
        
async def run_wrapped_coro(coro, error_data=None, error_handler=None, return_stat=False):
    if error_handler is None:
        error_handler = _coro_on_error
    
    try:
        resp = await coro 
        if return_stat:
            return resp, True
        return resp
    
    except asyncio.CancelledError:
        if return_stat:
            return None, False
        return None
    
    except Exception as e:
        try:
            await error_handler(coro, e, error_data=error_data)
        except asyncio.CancelledError:
            pass
        
        if return_stat:
            return None, False
        return None

def dispatch_func(coro, error_data=None, loop=None, error_handler=None):
    loop = loop or asyncio.get_event_loop()
    wrapped = run_wrapped_coro(coro, error_handler=error_handler, error_data=error_data)
    return asyncio.run_coroutine_threadsafe(wrapped, loop)
