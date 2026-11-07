"""Stack the two panels into Figure2.png, as in the manuscript: the top image is scaled to the width of the
bottom image (2864 px at 400 dpi), with a white gap of 39 px between them."""
import os
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
top, bot = Image.open('fig2_top_structure.png').convert('RGB'), Image.open('fig2_bottom_sequence_map.png').convert('RGB')
W = 2864
def fit(im): return im if im.width == W else im.resize((W, round(im.height * W / im.width)), Image.LANCZOS)
top, bot = fit(top), fit(bot)
out = Image.new('RGB', (W, top.height + 39 + bot.height), 'white')
out.paste(top, (0, 0)); out.paste(bot, (0, top.height + 39))
out.save('Figure2.png', dpi=(400, 400)); print('wrote Figure2.png', out.size)
