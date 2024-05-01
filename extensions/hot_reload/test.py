from au import setup
setup(debug=True)
from au.ext.ai.model_text_gen import TextGen

async def main():
    async with TextGen() as tg:
        res = await tg.complete('Test', max_tokens=10)
        print(res.text)
        print(len(res.choices))

import asyncio
loop = asyncio.get_event_loop()
loop.run_until_complete(main())