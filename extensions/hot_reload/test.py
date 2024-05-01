from au import setup
setup(debug=True)
from au.ext.ai.model_text_gen import TextGen

async with TextGen() as tg:
    res = await tg.complete('Test', max_tokens=10)
    print(res.text)
    print(len(res.choices))
