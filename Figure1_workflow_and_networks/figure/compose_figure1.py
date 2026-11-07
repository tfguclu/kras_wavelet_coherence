"""Stack the two panels into Figure1.png, as in the manuscript: the top image is scaled to the width of the
bottom image (2864 px at 400 dpi), with a white gap of 71 px between them."""
import os
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
top, bot = Image.open('fig1_top_workflow.png').convert('RGB'), Image.open('fig1_bottom_networks.png').convert('RGB')
W = 2864
def fit(im): return im if im.width == W else im.resize((W, round(im.height * W / im.width)), Image.LANCZOS)
top, bot = fit(top), fit(bot)
out = Image.new('RGB', (W, top.height + 71 + bot.height), 'white')
out.paste(top, (0, 0)); out.paste(bot, (0, top.height + 71))
out.save('Figure1.png', dpi=(400, 400)); print('wrote Figure1.png', out.size)
