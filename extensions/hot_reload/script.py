from threading import Thread
import asyncio
from modules.text_generation import gen_reply_function
from extensions.hot_reload.utils import dispatch_func
import os
file = 'extensions/hot_reload/exec.py'

import importlib.util
import sys


# TODO Current issues:
# tokenizer(batch, padding=..) - Doesn't work because most models don't come with a padding token

# ---
# Create a mixin that injects into the _sample function into the PreTrainedModel class.
# This would have to be imported before models load
# Extensions load after model as loaded.

# from there create a function that generates input ids for process
# if no changes, it will continue adding tokens without regenerating padding.
# If a new prompt is added, it gets padded if smaller, else redo them all.
# When a prompt finishes it will be removed and padding is trimmed.
# implement caching

# TODO ExllamaV2_HF doesn't support batching, but batch tests with GTPQ worked.

# built off this idea https://github.com/LowinLi/transformers-stream-generator/blob/main/transformers_stream_generator/main.py#L464

        
class Main:
    def __init__(self, loop:asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.q = asyncio.Queue()
        # self.q = []
        
        self.last_mod = 0
        
    def modified(self):
        return os.stat(file).st_mtime
    
    def callback(self, inputs):
        pass
        
    def reload(self):
        spec = importlib.util.spec_from_file_location("gen", file)
        gen = importlib.util.module_from_spec(spec)
        sys.modules["gen"] = gen
        spec.loader.exec_module(gen)

        # gen_reply_function[0] = gen.create_func(self.callback)
        gen_reply_function[0] = gen.generate_reply_HF
        print('Done')
    
    async def import_loop(self):
        while True:
            await asyncio.sleep(2)
            m = self.modified()
            delta = m-self.last_mod
            
            if delta >= 0.5:
                print('Reloading')
                
                try:
                    self.reload()
                    self.last_mod = self.modified()
                except Exception as e:
                    print(e)
                
    async def gen_loop(self):
        while True:
            task = await self.q.get()
        
    async def on_start(self):
        dispatch_func(self.import_loop())
        # dispatch_func(self.gen_loop())
    
    def start(self):
        asyncio.ensure_future(self.on_start(), loop=self.loop)
        self.loop.run_forever()


loop = asyncio.new_event_loop()
main = Main(loop)

def setup():
    Thread(target=main.start, daemon=True).start()
